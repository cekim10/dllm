"""
Run a trace-driven canvas-queue serving prototype with real LLaDA forward calls.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/serve_canvas_queues.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --workload mix75 \
    --num_requests 64 \
    --arrival_rate_rps 2 \
    --refinement_steps 16 \
    --max_batch_size 16 \
    --policies arrival_dense,exact_canvas_queue,exact_canvas_queue_wait,exact_canvas_queue_bounded \
    --output_prefix artifacts/elastic_canvas/serve_canvas_queues_mix75
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    if any(item <= 0 for item in values):
        raise ValueError(f"Values must be positive: {values}")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalize_inputs(inputs: Any) -> list[int]:
    if isinstance(inputs, torch.Tensor):
        if inputs.dim() == 1:
            return inputs.tolist()
        return inputs[0].tolist()
    if inputs and isinstance(inputs[0], int):
        return inputs
    return inputs[0]


def _round_canvas_length(useful_length: int, canvas_classes: list[int]) -> int:
    for canvas in sorted(canvas_classes):
        if useful_length <= canvas:
            return canvas
    return max(canvas_classes)


def _load_trace_lengths(path: str, canvas_classes: list[int]) -> list[int]:
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    if trace_path.suffix == ".json":
        data = json.loads(trace_path.read_text())
        if "per_request_elastic_lengths" in data:
            return [int(value) for value in data["per_request_elastic_lengths"]]
    lengths = []
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "canvas_length" in record:
            lengths.append(int(record["canvas_length"]))
        elif "useful_generated_tokens" in record:
            lengths.append(
                _round_canvas_length(
                    int(record["useful_generated_tokens"]),
                    canvas_classes,
                )
            )
    if not lengths:
        raise ValueError(f"Could not extract canvas lengths from {trace_path}")
    return lengths


def _workload_lengths(
    *,
    workload: str,
    num_requests: int,
    canvas_classes: list[int],
    trace_lengths_path: str | None,
    rng: random.Random,
) -> list[int]:
    classes = sorted(canvas_classes)
    min_canvas = min(classes)
    max_canvas = max(classes)
    if workload == "short":
        base = [min_canvas]
    elif workload == "medium":
        base = [128 if 128 in classes else classes[len(classes) // 2]]
    elif workload == "long":
        base = [max_canvas]
    elif workload == "mixed":
        base = classes
    elif workload == "mix50":
        return rng.choices([min_canvas, max_canvas], weights=[0.5, 0.5], k=num_requests)
    elif workload == "mix75":
        return rng.choices([min_canvas, max_canvas], weights=[0.75, 0.25], k=num_requests)
    elif workload == "mix90":
        return rng.choices([min_canvas, max_canvas], weights=[0.90, 0.10], k=num_requests)
    elif workload == "rare":
        base = classes if len(classes) >= 4 else [min_canvas, 64, 128, max_canvas]
        weights = [0.70, 0.20, 0.08, 0.02][: len(base)]
        return rng.choices(base, weights=weights, k=num_requests)
    elif workload == "shift":
        first = rng.choices([min_canvas, max_canvas], weights=[0.90, 0.10], k=num_requests // 2)
        second = rng.choices([min_canvas, max_canvas], weights=[0.35, 0.65], k=num_requests - len(first))
        return first + second
    elif workload == "trace":
        if trace_lengths_path is None:
            raise ValueError("workload=trace requires --trace_lengths_path")
        trace = _load_trace_lengths(trace_lengths_path, canvas_classes)
        return [trace[index % len(trace)] for index in range(num_requests)]
    elif workload.startswith("lengths:"):
        base = _parse_int_list(workload.split(":", 1)[1])
    else:
        raise ValueError(f"Unknown workload: {workload}")
    return [base[index % len(base)] for index in range(num_requests)]


@dataclass
class Request:
    request_id: int
    arrival_ms: float
    canvas: int
    remaining_steps: int
    deadline_ms: float | None = None
    service_class: str = "default"
    first_start_ms: float | None = None
    completion_ms: float | None = None
    service_ms: float = 0.0
    dispatches: int = 0


@dataclass
class Operation:
    policy: str
    operation_index: int
    start_ms: float
    end_ms: float
    latency_ms: float
    mode: str
    physical_canvas: int
    request_ids: list[int]
    canvas_lengths: list[int]
    token_waste: int
    attention_waste: int


def _make_requests(
    *,
    canvas_lengths: list[int],
    arrival_rate_rps: float,
    arrival_process: str,
    burst_size: int,
    burst_interval_ms: float,
    refinement_steps: int,
    slo_policy: str,
    slo_scale: float,
    rng: random.Random,
) -> list[Request]:
    now_ms = 0.0
    requests = []
    for request_id, canvas in enumerate(canvas_lengths):
        if arrival_process == "poisson":
            if request_id > 0 and arrival_rate_rps > 0:
                now_ms += rng.expovariate(arrival_rate_rps) * 1000.0
        elif arrival_process == "bursty":
            if request_id > 0 and request_id % max(burst_size, 1) == 0:
                now_ms += burst_interval_ms
        else:
            raise ValueError(f"Unknown arrival_process: {arrival_process}")

        if slo_policy == "none":
            deadline_ms = None
            service_class = "default"
        elif slo_policy == "by_canvas":
            if canvas <= 32:
                deadline_ms = now_ms + 2_000.0 * slo_scale
                service_class = "interactive_short"
            elif canvas <= 128:
                deadline_ms = now_ms + 5_000.0 * slo_scale
                service_class = "interactive_medium"
            else:
                deadline_ms = now_ms + 10_000.0 * slo_scale
                service_class = "interactive_long"
        else:
            raise ValueError(f"Unknown slo_policy: {slo_policy}")

        requests.append(
            Request(
                request_id=request_id,
                arrival_ms=now_ms,
                canvas=int(canvas),
                remaining_steps=refinement_steps,
                deadline_ms=deadline_ms,
                service_class=service_class,
            )
        )
    return requests


def _clone_requests(requests: list[Request]) -> list[Request]:
    return [
        Request(
            request_id=request.request_id,
            arrival_ms=request.arrival_ms,
            canvas=request.canvas,
            remaining_steps=request.remaining_steps,
            deadline_ms=request.deadline_ms,
            service_class=request.service_class,
        )
        for request in requests
    ]


def _oldest(requests: list[Request]) -> list[Request]:
    return sorted(requests, key=lambda request: (request.arrival_ms, request.request_id))


def _ready_requests(requests: list[Request], now_ms: float) -> list[Request]:
    return [
        request
        for request in requests
        if request.arrival_ms <= now_ms and request.remaining_steps > 0
    ]


def _next_arrival_ms(requests: list[Request], now_ms: float) -> float | None:
    future = [
        request.arrival_ms
        for request in requests
        if request.arrival_ms > now_ms and request.remaining_steps > 0
    ]
    return min(future) if future else None


def _estimate_single_step_ms(observed_ms: dict[int, list[float]], canvas: int, default_ms: float) -> float:
    values = observed_ms.get(canvas)
    return statistics.mean(values) if values else default_ms


def _deadline_dispatch_time(
    request: Request,
    *,
    now_ms: float,
    observed_ms: dict[int, list[float]],
    default_step_ms: float,
    deadline_safety_margin_ms: float,
) -> float | None:
    if request.deadline_ms is None:
        return None
    remaining_service_ms = (
        _estimate_single_step_ms(observed_ms, request.canvas, default_step_ms)
        * request.remaining_steps
    )
    dispatch_by = request.deadline_ms - remaining_service_ms - deadline_safety_margin_ms
    return max(now_ms, dispatch_by)


def _plan_arrival_dense(
    ready: list[Request],
    *,
    max_batch_size: int,
    **_: Any,
) -> tuple[str, list[Request], int] | None:
    group = _oldest(ready)[:max_batch_size]
    if not group:
        return None
    return "dense", group, max(request.canvas for request in group)


def _plan_exact_canvas_queue(
    ready: list[Request],
    *,
    max_batch_size: int,
    **_: Any,
) -> tuple[str, list[Request], int] | None:
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in ready:
        buckets[request.canvas].append(request)
    if not buckets:
        return None
    candidates = []
    for canvas, bucket in buckets.items():
        ordered = _oldest(bucket)
        candidates.append((ordered[0].arrival_ms, -len(ordered), canvas, ordered[:max_batch_size]))
    _, _, canvas, group = min(candidates)
    return "exact_canvas_queue", group, canvas


def _plan_exact_canvas_queue_wait(
    ready: list[Request],
    *,
    max_batch_size: int,
    min_bucket_size: int,
    max_bucket_wait_ms: float,
    now_ms: float,
    **_: Any,
) -> tuple[str, list[Request], int] | None:
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in ready:
        buckets[request.canvas].append(request)
    candidates = []
    for canvas, bucket in buckets.items():
        ordered = _oldest(bucket)
        oldest_age_ms = now_ms - ordered[0].arrival_ms
        deadline_ready = any(
            request.deadline_ms is not None and request.deadline_ms <= now_ms
            for request in ordered
        )
        if len(ordered) >= min_bucket_size or oldest_age_ms >= max_bucket_wait_ms or deadline_ready:
            candidates.append((ordered[0].arrival_ms, -len(ordered), canvas, ordered[:max_batch_size]))
    if not candidates:
        return None
    _, _, canvas, group = min(candidates)
    return "exact_canvas_queue_wait", group, canvas


def _plan_exact_canvas_queue_bounded(
    ready: list[Request],
    *,
    max_batch_size: int,
    target_bucket_size: int,
    max_bucket_wait_ms: float,
    now_ms: float,
    observed_ms: dict[int, list[float]],
    default_step_ms: float,
    deadline_safety_margin_ms: float,
    **_: Any,
) -> tuple[str, list[Request], int] | None:
    target = min(max(target_bucket_size, 1), max_batch_size)
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in ready:
        buckets[request.canvas].append(request)
    candidates = []
    for canvas, bucket in buckets.items():
        ordered = _oldest(bucket)
        oldest_age_ms = now_ms - ordered[0].arrival_ms
        dispatch_times = [
            _deadline_dispatch_time(
                request,
                now_ms=now_ms,
                observed_ms=observed_ms,
                default_step_ms=default_step_ms,
                deadline_safety_margin_ms=deadline_safety_margin_ms,
            )
            for request in ordered
        ]
        deadline_ready = any(
            value is not None and value <= now_ms for value in dispatch_times
        )
        if len(ordered) >= target or oldest_age_ms >= max_bucket_wait_ms or deadline_ready:
            earliest_deadline = min(
                (value for value in dispatch_times if value is not None),
                default=float("inf"),
            )
            candidates.append(
                (
                    0 if deadline_ready else 1,
                    earliest_deadline,
                    ordered[0].arrival_ms,
                    -len(ordered),
                    canvas,
                    ordered[:max_batch_size],
                )
            )
    if not candidates:
        return None
    _, _, _, _, canvas, group = min(candidates)
    return "exact_canvas_queue_bounded", group, canvas


def _build_batch(
    *,
    prompt_ids: list[int],
    group: list[Request],
    physical_canvas: int,
    mask_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_len = len(prompt_ids)
    total_len = prompt_len + physical_canvas
    input_ids = torch.full(
        (len(group), total_len),
        eos_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(group), total_len),
        dtype=torch.long,
        device=device,
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    for row, request in enumerate(group):
        input_ids[row, :prompt_len] = prompt_tensor
        input_ids[row, prompt_len : prompt_len + request.canvas] = mask_token_id
        attention_mask[row, : prompt_len + request.canvas] = 1
    return input_ids, attention_mask


def _dense_waste(prompt_tokens: int, canvases: list[int], physical_canvas: int) -> tuple[int, int]:
    token_waste = len(canvases) * physical_canvas - sum(canvases)
    physical_seq = prompt_tokens + physical_canvas
    attention_waste = len(canvases) * physical_seq * physical_seq - sum(
        (prompt_tokens + canvas) ** 2 for canvas in canvases
    )
    return int(token_waste), int(attention_waste)


@torch.inference_mode()
def _run_forward_ms(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    device = model.device
    _sync_device(device)
    start = time.perf_counter()
    _ = model(input_ids=input_ids, attention_mask=attention_mask).logits
    _sync_device(device)
    return (time.perf_counter() - start) * 1000.0


def _next_wait_event_ms(
    *,
    policy: str,
    ready: list[Request],
    now_ms: float,
    max_bucket_wait_ms: float,
    observed_ms: dict[int, list[float]],
    default_step_ms: float,
    deadline_safety_margin_ms: float,
) -> float | None:
    events = [
        request.arrival_ms + max_bucket_wait_ms
        for request in ready
        if policy in ("exact_canvas_queue_wait", "exact_canvas_queue_bounded")
        and request.arrival_ms + max_bucket_wait_ms > now_ms
    ]
    if policy == "exact_canvas_queue_bounded":
        for request in ready:
            event = _deadline_dispatch_time(
                request,
                now_ms=now_ms,
                observed_ms=observed_ms,
                default_step_ms=default_step_ms,
                deadline_safety_margin_ms=deadline_safety_margin_ms,
            )
            if event is not None and event > now_ms:
                events.append(event)
    return min(events) if events else None


def _simulate_policy(
    *,
    policy: str,
    base_requests: list[Request],
    model,
    prompt_ids: list[int],
    mask_token_id: int,
    eos_token_id: int,
    max_batch_size: int,
    min_bucket_size: int,
    target_bucket_size: int,
    max_bucket_wait_ms: float,
    default_step_ms: float,
    deadline_safety_margin_ms: float,
) -> tuple[list[Request], list[Operation]]:
    planners = {
        "arrival_dense": _plan_arrival_dense,
        "exact_canvas_queue": _plan_exact_canvas_queue,
        "exact_canvas_queue_wait": _plan_exact_canvas_queue_wait,
        "exact_canvas_queue_bounded": _plan_exact_canvas_queue_bounded,
    }
    if policy not in planners:
        raise ValueError(f"Unknown policy: {policy}")

    requests = _clone_requests(base_requests)
    operations = []
    observed_ms: dict[int, list[float]] = defaultdict(list)
    now_ms = min(request.arrival_ms for request in requests) if requests else 0.0

    while any(request.remaining_steps > 0 for request in requests):
        ready = _ready_requests(requests, now_ms)
        if not ready:
            next_arrival = _next_arrival_ms(requests, now_ms)
            if next_arrival is None:
                break
            now_ms = next_arrival
            ready = _ready_requests(requests, now_ms)

        plan = planners[policy](
            ready=ready,
            max_batch_size=max_batch_size,
            min_bucket_size=min_bucket_size,
            target_bucket_size=target_bucket_size,
            max_bucket_wait_ms=max_bucket_wait_ms,
            now_ms=now_ms,
            observed_ms=observed_ms,
            default_step_ms=default_step_ms,
            deadline_safety_margin_ms=deadline_safety_margin_ms,
        )
        if plan is None:
            next_arrival = _next_arrival_ms(requests, now_ms)
            next_wait = _next_wait_event_ms(
                policy=policy,
                ready=ready,
                now_ms=now_ms,
                max_bucket_wait_ms=max_bucket_wait_ms,
                observed_ms=observed_ms,
                default_step_ms=default_step_ms,
                deadline_safety_margin_ms=deadline_safety_margin_ms,
            )
            candidates = [
                value
                for value in (next_arrival, next_wait)
                if value is not None and value > now_ms
            ]
            if candidates:
                now_ms = min(candidates)
                continue
            plan = _plan_exact_canvas_queue(ready=ready, max_batch_size=max_batch_size)
            if plan is None:
                break

        mode, group, physical_canvas = plan
        input_ids, attention_mask = _build_batch(
            prompt_ids=prompt_ids,
            group=group,
            physical_canvas=physical_canvas,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            device=model.device,
        )
        latency_ms = _run_forward_ms(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        start_ms = now_ms
        end_ms = now_ms + latency_ms
        canvases = [request.canvas for request in group]
        token_waste, attention_waste = _dense_waste(
            len(prompt_ids), canvases, physical_canvas
        )
        operation = Operation(
            policy=policy,
            operation_index=len(operations),
            start_ms=start_ms,
            end_ms=end_ms,
            latency_ms=latency_ms,
            mode=mode,
            physical_canvas=physical_canvas,
            request_ids=[request.request_id for request in group],
            canvas_lengths=canvases,
            token_waste=token_waste,
            attention_waste=attention_waste,
        )
        operations.append(operation)
        observed_ms[physical_canvas].append(latency_ms)

        for request in group:
            if request.first_start_ms is None:
                request.first_start_ms = start_ms
            request.remaining_steps -= 1
            request.dispatches += 1
            request.service_ms += latency_ms
            if request.remaining_steps == 0:
                request.completion_ms = end_ms
        now_ms = end_ms

    return requests, operations


def _summarize(
    *,
    policy: str,
    requests: list[Request],
    operations: list[Operation],
    prompt_tokens: int,
    target_bucket_size: int,
) -> dict[str, Any]:
    completed = [request for request in requests if request.completion_ms is not None]
    latencies = [
        request.completion_ms - request.arrival_ms
        for request in completed
        if request.completion_ms is not None
    ]
    waits = [
        request.first_start_ms - request.arrival_ms
        for request in completed
        if request.first_start_ms is not None
    ]
    services = [request.service_ms for request in completed]
    inter_iteration_delays = [
        (request.completion_ms - request.arrival_ms) - (request.first_start_ms - request.arrival_ms) - request.service_ms
        for request in completed
        if request.completion_ms is not None and request.first_start_ms is not None
    ]
    first_arrival = min(request.arrival_ms for request in requests) if requests else 0.0
    last_completion = max((request.completion_ms or first_arrival) for request in requests)
    makespan_ms = max(last_completion - first_arrival, 1e-9)
    total_gpu_ms = sum(operation.latency_ms for operation in operations)
    physical_tokens = sum(
        len(operation.canvas_lengths) * operation.physical_canvas
        for operation in operations
    )
    useful_tokens = sum(sum(operation.canvas_lengths) for operation in operations)
    physical_attention = sum(
        len(operation.canvas_lengths) * (prompt_tokens + operation.physical_canvas) ** 2
        for operation in operations
    )
    useful_attention = sum(
        sum((prompt_tokens + canvas) ** 2 for canvas in operation.canvas_lengths)
        for operation in operations
    )
    slo_requests = [
        request
        for request in completed
        if request.deadline_ms is not None and request.completion_ms is not None
    ]
    slo_misses = [
        request
        for request in slo_requests
        if request.completion_ms is not None
        and request.deadline_ms is not None
        and request.completion_ms > request.deadline_ms
    ]
    batch_sizes = [len(operation.request_ids) for operation in operations]
    partial_ops = [
        operation for operation in operations if len(operation.request_ids) < target_bucket_size
    ]

    per_canvas = {}
    for canvas in sorted({request.canvas for request in requests}):
        canvas_requests = [request for request in completed if request.canvas == canvas]
        canvas_latencies = [
            request.completion_ms - request.arrival_ms
            for request in canvas_requests
            if request.completion_ms is not None
        ]
        canvas_waits = [
            request.first_start_ms - request.arrival_ms
            for request in canvas_requests
            if request.first_start_ms is not None
        ]
        per_canvas[str(canvas)] = {
            "num_requests": len(canvas_requests),
            "p95_latency_ms": _percentile(canvas_latencies, 0.95),
            "p99_latency_ms": _percentile(canvas_latencies, 0.99),
            "p95_first_wait_ms": _percentile(canvas_waits, 0.95),
            "mean_service_ms": statistics.mean([r.service_ms for r in canvas_requests])
            if canvas_requests
            else 0.0,
        }

    return {
        "policy": policy,
        "completed_requests": len(completed),
        "makespan_ms": makespan_ms,
        "throughput_rps": len(completed) / (makespan_ms / 1000.0),
        "gpu_busy_fraction": total_gpu_ms / makespan_ms,
        "total_gpu_ms": total_gpu_ms,
        "num_dispatches": len(operations),
        "mean_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0.0,
        "p50_batch_size": _percentile([float(x) for x in batch_sizes], 0.50),
        "p95_batch_size": _percentile([float(x) for x in batch_sizes], 0.95),
        "partial_dispatch_ratio": len(partial_ops) / len(operations) if operations else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_first_wait_ms": statistics.mean(waits) if waits else 0.0,
        "p95_first_wait_ms": _percentile(waits, 0.95),
        "mean_service_ms": statistics.mean(services) if services else 0.0,
        "p95_service_ms": _percentile(services, 0.95),
        "mean_inter_iteration_delay_ms": statistics.mean(inter_iteration_delays)
        if inter_iteration_delays
        else 0.0,
        "p95_inter_iteration_delay_ms": _percentile(inter_iteration_delays, 0.95),
        "slo_requests": len(slo_requests),
        "slo_miss_rate": len(slo_misses) / len(slo_requests) if slo_requests else 0.0,
        "token_coupling_waste_ratio": (
            (physical_tokens - useful_tokens) / physical_tokens
            if physical_tokens
            else 0.0
        ),
        "attention_coupling_waste_ratio": (
            (physical_attention - useful_attention) / physical_attention
            if physical_attention
            else 0.0
        ),
        "mode_counts": dict(Counter(operation.mode for operation in operations)),
        "per_canvas": per_canvas,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    prompt: str = "Explain canvas-aware dLLM serving in one sentence."
    output_prefix: str = "artifacts/elastic_canvas/serve_canvas_queues"
    workload: str = "mix75"
    trace_lengths_path: str | None = None
    num_requests: int = 64
    arrival_rate_rps: float = 2.0
    arrival_process: str = "poisson"
    burst_size: int = 16
    burst_interval_ms: float = 5_000.0
    refinement_steps: int = 16
    max_batch_size: int = 16
    canvas_classes: str = "32,64,128,256"
    policies: str = "arrival_dense,exact_canvas_queue,exact_canvas_queue_wait,exact_canvas_queue_bounded"
    min_bucket_size: int = 4
    target_bucket_size: int = 8
    max_bucket_wait_ms: float = 20.0
    default_step_ms: float = 100.0
    slo_policy: str = "none"
    slo_scale: float = 10.0
    deadline_safety_margin_ms: float = 250.0
    warmup: int = 2
    seed: int = 42

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


parser = transformers.HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
transformers.set_seed(script_args.seed)
rng = random.Random(script_args.seed)

config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
    script_args.model_name_or_path
)
model = dllm.utils.get_model(model_args=script_args, config=config).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)

inputs = tokenizer.apply_chat_template(
    [[{"role": "user", "content": script_args.prompt}]],
    add_generation_prompt=True,
    tokenize=True,
)
prompt_ids = _normalize_inputs(inputs)
mask_token_id = tokenizer.mask_token_id
eos_token_id = tokenizer.eos_token_id
if mask_token_id is None:
    raise ValueError("tokenizer.mask_token_id is required")
if eos_token_id is None:
    raise ValueError("tokenizer.eos_token_id is required")

canvas_classes = _parse_int_list(script_args.canvas_classes)
canvas_lengths = _workload_lengths(
    workload=script_args.workload,
    num_requests=script_args.num_requests,
    canvas_classes=canvas_classes,
    trace_lengths_path=script_args.trace_lengths_path,
    rng=rng,
)
base_requests = _make_requests(
    canvas_lengths=canvas_lengths,
    arrival_rate_rps=script_args.arrival_rate_rps,
    arrival_process=script_args.arrival_process,
    burst_size=script_args.burst_size,
    burst_interval_ms=script_args.burst_interval_ms,
    refinement_steps=script_args.refinement_steps,
    slo_policy=script_args.slo_policy,
    slo_scale=script_args.slo_scale,
    rng=rng,
)

if script_args.warmup > 0:
    warmup_group = [
        Request(index, 0.0, max(canvas_classes), 1)
        for index in range(min(script_args.max_batch_size, max(1, len(base_requests))))
    ]
    warmup_ids, warmup_mask = _build_batch(
        prompt_ids=prompt_ids,
        group=warmup_group,
        physical_canvas=max(canvas_classes),
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        device=model.device,
    )
    for _ in range(script_args.warmup):
        _run_forward_ms(model=model, input_ids=warmup_ids, attention_mask=warmup_mask)

summaries = []
operation_rows = []
request_rows = []
wall_start = time.perf_counter()
for policy in [part.strip() for part in script_args.policies.split(",") if part.strip()]:
    print(f"Running policy: {policy}", flush=True)
    requests, operations = _simulate_policy(
        policy=policy,
        base_requests=base_requests,
        model=model,
        prompt_ids=prompt_ids,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        max_batch_size=script_args.max_batch_size,
        min_bucket_size=script_args.min_bucket_size,
        target_bucket_size=script_args.target_bucket_size,
        max_bucket_wait_ms=script_args.max_bucket_wait_ms,
        default_step_ms=script_args.default_step_ms,
        deadline_safety_margin_ms=script_args.deadline_safety_margin_ms,
    )
    summary = _summarize(
        policy=policy,
        requests=requests,
        operations=operations,
        prompt_tokens=len(prompt_ids),
        target_bucket_size=script_args.target_bucket_size,
    )
    summaries.append(summary)
    for operation in operations:
        operation_rows.append(
            {
                "policy": operation.policy,
                "operation_index": operation.operation_index,
                "start_ms": operation.start_ms,
                "end_ms": operation.end_ms,
                "latency_ms": operation.latency_ms,
                "mode": operation.mode,
                "batch_size": len(operation.request_ids),
                "physical_canvas": operation.physical_canvas,
                "max_canvas": max(operation.canvas_lengths),
                "sum_canvas": sum(operation.canvas_lengths),
                "token_waste": operation.token_waste,
                "attention_waste": operation.attention_waste,
                "canvas_lengths": ",".join(str(value) for value in operation.canvas_lengths),
            }
        )
    for request in requests:
        request_rows.append(
            {
                "policy": policy,
                "request_id": request.request_id,
                "arrival_ms": request.arrival_ms,
                "first_start_ms": request.first_start_ms,
                "completion_ms": request.completion_ms,
                "latency_ms": (
                    request.completion_ms - request.arrival_ms
                    if request.completion_ms is not None
                    else None
                ),
                "first_wait_ms": (
                    request.first_start_ms - request.arrival_ms
                    if request.first_start_ms is not None
                    else None
                ),
                "service_ms": request.service_ms,
                "inter_iteration_delay_ms": (
                    (request.completion_ms - request.arrival_ms)
                    - (request.first_start_ms - request.arrival_ms)
                    - request.service_ms
                    if request.completion_ms is not None
                    and request.first_start_ms is not None
                    else None
                ),
                "dispatches": request.dispatches,
                "canvas": request.canvas,
                "deadline_ms": request.deadline_ms,
                "service_class": request.service_class,
                "slo_miss": (
                    request.completion_ms > request.deadline_ms
                    if request.completion_ms is not None
                    and request.deadline_ms is not None
                    else None
                ),
            }
        )
wall_elapsed_s = time.perf_counter() - wall_start

baseline = summaries[0]
for summary in summaries:
    summary["throughput_speedup_vs_" + baseline["policy"]] = (
        summary["throughput_rps"] / baseline["throughput_rps"]
        if baseline["throughput_rps"]
        else 0.0
    )
    summary["p95_latency_ratio_vs_" + baseline["policy"]] = (
        summary["p95_latency_ms"] / baseline["p95_latency_ms"]
        if baseline["p95_latency_ms"]
        else 0.0
    )
    summary["gpu_time_ratio_vs_" + baseline["policy"]] = (
        summary["total_gpu_ms"] / baseline["total_gpu_ms"]
        if baseline["total_gpu_ms"]
        else 0.0
    )

output = {
    "model_name_or_path": script_args.model_name_or_path,
    "workload": script_args.workload,
    "canvas_distribution": dict(Counter(canvas_lengths)),
    "num_requests": script_args.num_requests,
    "arrival_rate_rps": script_args.arrival_rate_rps,
    "arrival_process": script_args.arrival_process,
    "refinement_steps": script_args.refinement_steps,
    "max_batch_size": script_args.max_batch_size,
    "canvas_classes": canvas_classes,
    "prompt_tokens": len(prompt_ids),
    "slo_policy": script_args.slo_policy,
    "slo_scale": script_args.slo_scale,
    "target_bucket_size": script_args.target_bucket_size,
    "max_bucket_wait_ms": script_args.max_bucket_wait_ms,
    "wall_elapsed_s": wall_elapsed_s,
    "summaries": summaries,
}

prefix = Path(script_args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
summary_path = prefix.with_name(prefix.name + "_summary.json")
operations_path = prefix.with_name(prefix.name + "_operations.jsonl")
requests_path = prefix.with_name(prefix.name + "_requests.csv")
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=True, indent=2)
_write_jsonl(operations_path, operation_rows)
_write_csv(requests_path, request_rows)

print("Saved summary:", summary_path)
print("Saved operations:", operations_path)
print("Saved requests:", requests_path)
print(
    "\t".join(
        [
            "policy",
            "throughput",
            "p95_ms",
            "p95_wait",
            "p95_service",
            "p95_inter",
            "gpu_busy",
            "gpu_ms",
            "avg_batch",
            "partial",
            "tok_waste",
            "attn_waste",
            "slo_miss",
        ]
    )
)
for summary in summaries:
    print(
        "\t".join(
            [
                summary["policy"],
                f"{summary['throughput_rps']:.3f}",
                f"{summary['p95_latency_ms']:.1f}",
                f"{summary['p95_first_wait_ms']:.1f}",
                f"{summary['p95_service_ms']:.1f}",
                f"{summary['p95_inter_iteration_delay_ms']:.1f}",
                f"{summary['gpu_busy_fraction']:.3f}",
                f"{summary['total_gpu_ms']:.1f}",
                f"{summary['mean_batch_size']:.2f}",
                f"{summary['partial_dispatch_ratio']:.3f}",
                f"{summary['token_coupling_waste_ratio']:.3f}",
                f"{summary['attention_coupling_waste_ratio']:.3f}",
                f"{summary['slo_miss_rate']:.3f}",
            ]
        )
    )
