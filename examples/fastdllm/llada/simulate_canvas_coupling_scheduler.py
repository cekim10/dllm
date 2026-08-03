"""
Simulate persistent response-canvas coupling under online dLLM scheduling.

Run from repo root after collecting forward microbenchmarks:
  python examples/fastdllm/llada/simulate_canvas_coupling_scheduler.py \
    --latency_table artifacts/elastic_canvas/forward_bench_sweep.json \
    --workload mix75 \
    --num_requests 500 \
    --arrival_rate_rps 12 \
    --max_batch_size 16 \
    --output_prefix artifacts/elastic_canvas/canvas_scheduler_mix75
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _round_canvas_length(
    useful_length: int,
    *,
    canvas_classes: list[int],
) -> int:
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
        if "results" in data:
            lengths = []
            for result in data["results"]:
                lengths.extend(int(value) for value in result.get("canvas_lengths", []))
            if lengths:
                return lengths
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
                    canvas_classes=canvas_classes,
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
    max_canvas = max(canvas_classes)
    min_canvas = min(canvas_classes)
    classes = sorted(canvas_classes)

    if workload == "short":
        base = [min_canvas]
    elif workload == "medium":
        base = [128 if 128 in classes else classes[len(classes) // 2]]
    elif workload == "long":
        base = [max_canvas]
    elif workload == "mixed":
        base = classes
    elif workload == "mix50":
        base = [min_canvas, max_canvas]
        weights = [0.5, 0.5]
        return rng.choices(base, weights=weights, k=num_requests)
    elif workload == "mix75":
        base = [min_canvas, max_canvas]
        weights = [0.75, 0.25]
        return rng.choices(base, weights=weights, k=num_requests)
    elif workload == "mix90":
        base = [min_canvas, max_canvas]
        weights = [0.90, 0.10]
        return rng.choices(base, weights=weights, k=num_requests)
    elif workload == "rare":
        base = classes
        if len(base) < 4:
            base = [min_canvas, 64, 128, max_canvas]
        weights = [0.70, 0.20, 0.08, 0.02][: len(base)]
        return rng.choices(base, weights=weights, k=num_requests)
    elif workload == "lowvar":
        base = [length for length in classes if 64 <= length <= 192] or classes
    elif workload == "highvar":
        base = classes
        weights = [0.30, 0.20, 0.20, 0.30][: len(base)]
        if len(weights) == len(base):
            return rng.choices(base, weights=weights, k=num_requests)
    elif workload == "shift":
        first = rng.choices([min_canvas, max_canvas], weights=[0.90, 0.10], k=num_requests // 2)
        second = rng.choices([min_canvas, max_canvas], weights=[0.35, 0.65], k=num_requests - len(first))
        return first + second
    elif workload == "adversarial":
        period = max(2, min(8, num_requests))
        return [
            max_canvas if (index + 1) % period == 0 else min_canvas
            for index in range(num_requests)
        ]
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


class ForwardCostModel:
    def __init__(self, latency_table_path: str):
        self.path = Path(latency_table_path)
        data = json.loads(self.path.read_text())
        self.prompt_tokens = 0
        self.observations: list[tuple[int, int, float]] = []
        self.exact: dict[tuple[int, int], float] = {}
        for result in data["results"]:
            batch_size = int(result["batch_size"])
            self.prompt_tokens = max(self.prompt_tokens, int(result["prompt_tokens"]))
            fixed_canvas = int(result["fixed_canvas"])
            max_elastic_canvas = int(result["max_elastic_canvas"])
            self._add(batch_size, fixed_canvas, float(result["dense_fixed"]["avg_ms"]))
            self._add(
                batch_size,
                max_elastic_canvas,
                float(result["elastic_dense"]["avg_ms"]),
            )
            if int(result["bucketed_shape_decoupled"]["num_calls"]) == 1:
                unique_lengths = set(int(length) for length in result["canvas_lengths"])
                if len(unique_lengths) == 1:
                    self._add(
                        batch_size,
                        unique_lengths.pop(),
                        float(result["bucketed_shape_decoupled"]["avg_ms"]),
                    )
        if not self.observations:
            raise ValueError(f"No latency observations found in {self.path}")
        self.min_ms = min(ms for _, _, ms in self.observations)
        self.max_batch = max(batch for batch, _, _ in self.observations)
        self.max_canvas = max(canvas for _, canvas, _ in self.observations)
        self.coefficients: list[float] | None = None
        self._fit()

    def _add(self, batch_size: int, canvas: int, ms: float) -> None:
        key = (batch_size, canvas)
        if key in self.exact:
            self.exact[key] = min(self.exact[key], ms)
            return
        self.exact[key] = ms
        self.observations.append((batch_size, canvas, ms))

    def _features(self, batch_size: int, canvas: int) -> list[float]:
        b = batch_size / max(self.max_batch, 1)
        l = (self.prompt_tokens + canvas) / max(self.prompt_tokens + self.max_canvas, 1)
        return [1.0, b, l, b * l, b * l * l]

    def _fit(self) -> None:
        try:
            import numpy as np

            x = np.array(
                [self._features(batch, canvas) for batch, canvas, _ in self.observations],
                dtype=float,
            )
            y = np.array([ms for _, _, ms in self.observations], dtype=float)
            coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
            self.coefficients = [float(value) for value in coeffs]
        except Exception:
            self.coefficients = None

    def predict_ms(self, batch_size: int, canvas: int) -> float:
        if batch_size <= 0:
            return 0.0
        key = (batch_size, canvas)
        if key in self.exact:
            return self.exact[key]
        if self.coefficients is not None:
            estimate = sum(
                coefficient * feature
                for coefficient, feature in zip(
                    self.coefficients, self._features(batch_size, canvas)
                )
            )
            return max(float(estimate), self.min_ms * 0.25)

        nearest_batch, nearest_canvas, nearest_ms = min(
            self.observations,
            key=lambda obs: abs(obs[0] - batch_size) + abs(obs[1] - canvas) / 32.0,
        )
        scale = (batch_size / nearest_batch) * (
            (self.prompt_tokens + canvas) / (self.prompt_tokens + nearest_canvas)
        ) ** 2
        return max(nearest_ms * scale, self.min_ms * 0.25)


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


@dataclass
class Operation:
    policy: str
    start_ms: float
    end_ms: float
    request_ids: list[int]
    canvas_lengths: list[int]
    mode: str
    physical_canvas: int
    latency_ms: float
    token_waste: int
    attention_waste: int


@dataclass
class SimulationResult:
    policy: str
    requests: list[Request]
    operations: list[Operation] = field(default_factory=list)


def _arrivals(
    *,
    num_requests: int,
    arrival_rate_rps: float,
    canvas_lengths: list[int],
    refinement_steps: int,
    arrival_process: str,
    burst_size: int,
    burst_interval_ms: float,
    slo_policy: str,
    slo_scale: float,
    rng: random.Random,
) -> list[Request]:
    now_ms = 0.0
    requests = []
    for request_id in range(num_requests):
        if arrival_process == "poisson":
            if request_id > 0 and arrival_rate_rps > 0:
                now_ms += rng.expovariate(arrival_rate_rps) * 1000.0
        elif arrival_process == "bursty":
            if request_id > 0 and request_id % max(burst_size, 1) == 0:
                now_ms += burst_interval_ms
        else:
            raise ValueError(f"Unknown arrival_process: {arrival_process}")

        canvas = int(canvas_lengths[request_id])
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
        elif slo_policy == "mixed_priority":
            if rng.random() < 0.35:
                deadline_ms = now_ms + 2_000.0 * slo_scale
                service_class = "interactive"
            else:
                deadline_ms = now_ms + 15_000.0 * slo_scale
                service_class = "background"
        else:
            raise ValueError(f"Unknown slo_policy: {slo_policy}")

        requests.append(
            Request(
                request_id=request_id,
                arrival_ms=now_ms,
                canvas=canvas,
                remaining_steps=refinement_steps,
                deadline_ms=deadline_ms,
                service_class=service_class,
            )
        )
    return requests


def _dense_waste(prompt_tokens: int, canvases: list[int], physical_canvas: int) -> tuple[int, int]:
    token_waste = len(canvases) * physical_canvas - sum(canvases)
    seq_physical = prompt_tokens + physical_canvas
    attention_waste = len(canvases) * seq_physical * seq_physical - sum(
        (prompt_tokens + canvas) ** 2 for canvas in canvases
    )
    return int(token_waste), int(attention_waste)


def _run_call(
    *,
    policy: str,
    mode: str,
    now_ms: float,
    group: list[Request],
    physical_canvas: int,
    cost_model: ForwardCostModel,
    operations: list[Operation],
) -> float:
    if not group:
        return now_ms
    canvases = [request.canvas for request in group]
    latency_ms = cost_model.predict_ms(len(group), physical_canvas)
    token_waste, attention_waste = _dense_waste(
        cost_model.prompt_tokens, canvases, physical_canvas
    )
    start_ms = now_ms
    end_ms = now_ms + latency_ms
    for request in group:
        if request.first_start_ms is None:
            request.first_start_ms = start_ms
        request.remaining_steps -= 1
        if request.remaining_steps == 0:
            request.completion_ms = end_ms
    operations.append(
        Operation(
            policy=policy,
            start_ms=start_ms,
            end_ms=end_ms,
            request_ids=[request.request_id for request in group],
            canvas_lengths=canvases,
            mode=mode,
            physical_canvas=physical_canvas,
            latency_ms=latency_ms,
            token_waste=token_waste,
            attention_waste=attention_waste,
        )
    )
    return end_ms


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


def _oldest(requests: list[Request]) -> list[Request]:
    return sorted(requests, key=lambda request: (request.arrival_ms, request.request_id))


def _plan_arrival_dense(
    ready: list[Request],
    max_batch_size: int,
    **_: Any,
) -> list[tuple[str, list[Request], int]]:
    group = _oldest(ready)[:max_batch_size]
    return [("dense", group, max(request.canvas for request in group))]


def _plan_exact_bucket(
    ready: list[Request],
    max_batch_size: int,
    **_: Any,
) -> list[tuple[str, list[Request], int]]:
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in ready:
        buckets[request.canvas].append(request)
    candidates = []
    for canvas, bucket in buckets.items():
        ordered = _oldest(bucket)
        oldest_arrival = ordered[0].arrival_ms
        candidates.append((oldest_arrival, -len(ordered), canvas, ordered[:max_batch_size]))
    _, _, canvas, group = min(candidates)
    return [("exact_bucket", group, canvas)]


def _plan_exact_bucket_wait(
    ready: list[Request],
    max_batch_size: int,
    now_ms: float,
    min_bucket_size: int,
    max_bucket_wait_ms: float,
    **_: Any,
) -> list[tuple[str, list[Request], int]]:
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in ready:
        buckets[request.canvas].append(request)

    eligible = []
    for canvas, bucket in buckets.items():
        ordered = _oldest(bucket)
        oldest_age_ms = now_ms - ordered[0].arrival_ms
        deadline_ready = any(
            request.deadline_ms is not None and request.deadline_ms <= now_ms
            for request in ordered
        )
        if (
            len(ordered) >= min_bucket_size
            or oldest_age_ms >= max_bucket_wait_ms
            or deadline_ready
        ):
            eligible.append(
                (ordered[0].arrival_ms, -len(ordered), canvas, ordered[:max_batch_size])
            )

    if not eligible:
        return []
    _, _, canvas, group = min(eligible)
    return [("exact_bucket_wait", group, canvas)]


def _deadline_dispatch_time(
    request: Request,
    *,
    now_ms: float,
    cost_model: ForwardCostModel,
    deadline_safety_margin_ms: float,
) -> float | None:
    if request.deadline_ms is None:
        return None
    single_call_ms = cost_model.predict_ms(1, request.canvas)
    remaining_service_ms = single_call_ms * request.remaining_steps
    dispatch_by = request.deadline_ms - remaining_service_ms - deadline_safety_margin_ms
    return max(now_ms, dispatch_by)


def _plan_exact_bucket_bounded(
    ready: list[Request],
    max_batch_size: int,
    now_ms: float,
    cost_model: ForwardCostModel,
    target_bucket_size: int,
    max_bucket_wait_ms: float,
    deadline_safety_margin_ms: float,
    **_: Any,
) -> list[tuple[str, list[Request], int]]:
    target = min(max(target_bucket_size, 1), max_batch_size)
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in ready:
        buckets[request.canvas].append(request)

    eligible = []
    for canvas, bucket in buckets.items():
        ordered = _oldest(bucket)
        oldest_age_ms = now_ms - ordered[0].arrival_ms
        dispatch_deadlines = [
            _deadline_dispatch_time(
                request,
                now_ms=now_ms,
                cost_model=cost_model,
                deadline_safety_margin_ms=deadline_safety_margin_ms,
            )
            for request in ordered
        ]
        deadline_ready = any(
            dispatch_time is not None and dispatch_time <= now_ms
            for dispatch_time in dispatch_deadlines
        )
        if (
            len(ordered) >= target
            or oldest_age_ms >= max_bucket_wait_ms
            or deadline_ready
        ):
            # Prioritize urgent buckets, then oldest buckets, then fuller batches.
            earliest_dispatch = min(
                (time for time in dispatch_deadlines if time is not None),
                default=float("inf"),
            )
            eligible.append(
                (
                    0 if deadline_ready else 1,
                    earliest_dispatch,
                    ordered[0].arrival_ms,
                    -len(ordered),
                    canvas,
                    ordered[:max_batch_size],
                )
            )

    if not eligible:
        return []
    _, _, _, _, canvas, group = min(eligible)
    return [("exact_bucket_bounded", group, canvas)]


def _split_by_exact_canvas(group: list[Request]) -> list[tuple[str, list[Request], int]]:
    buckets: dict[int, list[Request]] = defaultdict(list)
    for request in group:
        buckets[request.canvas].append(request)
    return [
        ("split_bucket", _oldest(bucket), canvas)
        for canvas, bucket in sorted(buckets.items())
    ]


def _plan_split_oldest(
    ready: list[Request],
    max_batch_size: int,
    **_: Any,
) -> list[tuple[str, list[Request], int]]:
    return _split_by_exact_canvas(_oldest(ready)[:max_batch_size])


def _partition_cost_ms(
    groups: list[tuple[str, list[Request], int]],
    cost_model: ForwardCostModel,
) -> float:
    return sum(
        cost_model.predict_ms(len(group), physical_canvas)
        for _, group, physical_canvas in groups
    )


def _plan_canvas_aware(
    ready: list[Request],
    max_batch_size: int,
    now_ms: float,
    cost_model: ForwardCostModel,
    planner_window: int,
    split_min_speedup: float,
    age_weight: float,
    **_: Any,
) -> list[tuple[str, list[Request], int]]:
    ordered = _oldest(ready)[: max(max_batch_size, planner_window)]
    best_score = -1.0
    best_plan: list[tuple[str, list[Request], int]] | None = None
    for size in range(1, min(max_batch_size, len(ordered)) + 1):
        group = ordered[:size]
        dense_plan = [("aware_dense", group, max(request.canvas for request in group))]
        split_plan = _split_by_exact_canvas(group)
        dense_ms = _partition_cost_ms(dense_plan, cost_model)
        split_ms = _partition_cost_ms(split_plan, cost_model)
        if split_ms * split_min_speedup < dense_ms:
            chosen = split_plan
            chosen_ms = split_ms
        else:
            chosen = dense_plan
            chosen_ms = dense_ms
        oldest_age_ms = max(0.0, now_ms - group[0].arrival_ms)
        score = (size / max(chosen_ms, 1e-9)) * (1.0 + age_weight * oldest_age_ms / 1000.0)
        if score > best_score:
            best_score = score
            best_plan = chosen
    assert best_plan is not None
    return best_plan


def simulate_policy(
    *,
    policy: str,
    base_requests: list[Request],
    cost_model: ForwardCostModel,
    max_batch_size: int,
    planner_window: int,
    split_min_speedup: float,
    age_weight: float,
    min_bucket_size: int,
    target_bucket_size: int,
    max_bucket_wait_ms: float,
    deadline_safety_margin_ms: float,
) -> SimulationResult:
    requests = [
        Request(
            request_id=request.request_id,
            arrival_ms=request.arrival_ms,
            canvas=request.canvas,
            remaining_steps=request.remaining_steps,
            deadline_ms=request.deadline_ms,
            service_class=request.service_class,
        )
        for request in base_requests
    ]
    operations: list[Operation] = []
    now_ms = min(request.arrival_ms for request in requests) if requests else 0.0
    planners = {
        "arrival_dense": _plan_arrival_dense,
        "exact_bucket": _plan_exact_bucket,
        "exact_bucket_wait": _plan_exact_bucket_wait,
        "exact_bucket_bounded": _plan_exact_bucket_bounded,
        "split_oldest": _plan_split_oldest,
        "canvas_aware": _plan_canvas_aware,
    }
    if policy not in planners:
        raise ValueError(f"Unknown policy: {policy}")

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
            now_ms=now_ms,
            cost_model=cost_model,
            planner_window=planner_window,
            split_min_speedup=split_min_speedup,
            age_weight=age_weight,
            min_bucket_size=min_bucket_size,
            target_bucket_size=target_bucket_size,
            max_bucket_wait_ms=max_bucket_wait_ms,
            deadline_safety_margin_ms=deadline_safety_margin_ms,
        )
        if not plan:
            next_arrival = _next_arrival_ms(requests, now_ms)
            wait_deadlines = [
                request.arrival_ms + max_bucket_wait_ms
                for request in ready
                if request.arrival_ms + max_bucket_wait_ms > now_ms
            ]
            slack_deadlines = [
                dispatch_time
                for request in ready
                for dispatch_time in [
                    _deadline_dispatch_time(
                        request,
                        now_ms=now_ms,
                        cost_model=cost_model,
                        deadline_safety_margin_ms=deadline_safety_margin_ms,
                    )
                ]
                if dispatch_time is not None and dispatch_time > now_ms
            ]
            next_wait_deadline = min(wait_deadlines) if wait_deadlines else None
            next_slack_deadline = min(slack_deadlines) if slack_deadlines else None
            candidates = [
                value
                for value in (next_arrival, next_wait_deadline, next_slack_deadline)
                if value is not None and value > now_ms
            ]
            if candidates:
                now_ms = min(candidates)
                continue
            plan = _plan_exact_bucket(
                ready=ready,
                max_batch_size=max_batch_size,
            )
        for mode, group, physical_canvas in plan:
            now_ms = _run_call(
                policy=policy,
                mode=mode,
                now_ms=now_ms,
                group=group,
                physical_canvas=physical_canvas,
                cost_model=cost_model,
                operations=operations,
            )

    return SimulationResult(policy=policy, requests=requests, operations=operations)


def _summarize(result: SimulationResult, prompt_tokens: int) -> dict[str, Any]:
    completed = [request for request in result.requests if request.completion_ms is not None]
    latencies = [
        float(request.completion_ms - request.arrival_ms)
        for request in completed
        if request.completion_ms is not None
    ]
    first_waits = [
        float(request.first_start_ms - request.arrival_ms)
        for request in completed
        if request.first_start_ms is not None
    ]
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
    first_arrival = min(request.arrival_ms for request in result.requests)
    last_completion = max(request.completion_ms or first_arrival for request in result.requests)
    makespan_s = max((last_completion - first_arrival) / 1000.0, 1e-9)
    total_gpu_ms = sum(operation.latency_ms for operation in result.operations)
    useful_token_units = sum(
        sum(operation.canvas_lengths) for operation in result.operations
    )
    physical_token_units = sum(
        len(operation.canvas_lengths) * operation.physical_canvas
        for operation in result.operations
    )
    useful_attention_units = sum(
        sum((prompt_tokens + canvas) ** 2 for canvas in operation.canvas_lengths)
        for operation in result.operations
    )
    physical_attention_units = sum(
        len(operation.canvas_lengths)
        * (prompt_tokens + operation.physical_canvas) ** 2
        for operation in result.operations
    )

    per_canvas = {}
    for canvas in sorted({request.canvas for request in result.requests}):
        group = [
            float(request.completion_ms - request.arrival_ms)
            for request in completed
            if request.canvas == canvas and request.completion_ms is not None
        ]
        waits = [
            float(request.first_start_ms - request.arrival_ms)
            for request in completed
            if request.canvas == canvas and request.first_start_ms is not None
        ]
        canvas_slo = [
            request
            for request in completed
            if request.canvas == canvas
            and request.deadline_ms is not None
            and request.completion_ms is not None
        ]
        canvas_slo_miss = [
            request
            for request in canvas_slo
            if request.completion_ms is not None
            and request.deadline_ms is not None
            and request.completion_ms > request.deadline_ms
        ]
        per_canvas[str(canvas)] = {
            "num_requests": len(group),
            "mean_latency_ms": statistics.mean(group) if group else 0.0,
            "p95_latency_ms": _percentile(group, 0.95),
            "p99_latency_ms": _percentile(group, 0.99),
            "mean_first_wait_ms": statistics.mean(waits) if waits else 0.0,
            "p95_first_wait_ms": _percentile(waits, 0.95),
            "slo_miss_rate": len(canvas_slo_miss) / len(canvas_slo)
            if canvas_slo
            else 0.0,
        }

    return {
        "policy": result.policy,
        "completed_requests": len(completed),
        "makespan_s": makespan_s,
        "throughput_rps": len(completed) / makespan_s,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_first_wait_ms": statistics.mean(first_waits) if first_waits else 0.0,
        "p95_first_wait_ms": _percentile(first_waits, 0.95),
        "p99_first_wait_ms": _percentile(first_waits, 0.99),
        "slo_requests": len(slo_requests),
        "slo_misses": len(slo_misses),
        "slo_miss_rate": len(slo_misses) / len(slo_requests)
        if slo_requests
        else 0.0,
        "total_gpu_ms": total_gpu_ms,
        "num_forward_ops": len(result.operations),
        "avg_batch_size": (
            statistics.mean(len(operation.request_ids) for operation in result.operations)
            if result.operations
            else 0.0
        ),
        "avg_physical_canvas": (
            statistics.mean(operation.physical_canvas for operation in result.operations)
            if result.operations
            else 0.0
        ),
        "token_coupling_waste": physical_token_units - useful_token_units,
        "token_coupling_waste_ratio": (
            (physical_token_units - useful_token_units) / physical_token_units
            if physical_token_units
            else 0.0
        ),
        "attention_coupling_waste": physical_attention_units - useful_attention_units,
        "attention_coupling_waste_ratio": (
            (physical_attention_units - useful_attention_units) / physical_attention_units
            if physical_attention_units
            else 0.0
        ),
        "mode_counts": dict(Counter(operation.mode for operation in result.operations)),
        "per_canvas_latency": per_canvas,
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
    latency_table: str = "artifacts/elastic_canvas/forward_bench_sweep.json"
    output_prefix: str = "artifacts/elastic_canvas/canvas_scheduler"
    workload: str = "mix75"
    trace_lengths_path: str | None = None
    num_requests: int = 500
    arrival_rate_rps: float = 12.0
    arrival_process: str = "poisson"
    burst_size: int = 16
    burst_interval_ms: float = 5_000.0
    slo_policy: str = "none"
    slo_scale: float = 1.0
    refinement_steps: int = 115
    max_batch_size: int = 16
    canvas_classes: str = "32,64,128,256"
    policies: str = "arrival_dense,exact_bucket,exact_bucket_wait,exact_bucket_bounded,split_oldest,canvas_aware"
    planner_window: int = 32
    split_min_speedup: float = 1.03
    age_weight: float = 0.05
    min_bucket_size: int = 4
    target_bucket_size: int = 8
    max_bucket_wait_ms: float = 2_000.0
    deadline_safety_margin_ms: float = 250.0
    seed: int = 42


parser = dataclass_parser = None
try:
    import transformers

    parser = transformers.HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
except Exception:
    # Keep the script usable in minimal Python environments without transformers.
    import argparse

    arg_parser = argparse.ArgumentParser()
    for field_info in ScriptArguments.__dataclass_fields__.values():
        default = field_info.default
        arg_type = type(default) if default is not None else str
        arg_parser.add_argument(f"--{field_info.name}", default=default, type=arg_type)
    script_args = ScriptArguments(**vars(arg_parser.parse_args()))

rng = random.Random(script_args.seed)
canvas_classes = _parse_int_list(script_args.canvas_classes)
cost_model = ForwardCostModel(script_args.latency_table)
canvas_lengths = _workload_lengths(
    workload=script_args.workload,
    num_requests=script_args.num_requests,
    canvas_classes=canvas_classes,
    trace_lengths_path=script_args.trace_lengths_path,
    rng=rng,
)
requests = _arrivals(
    num_requests=script_args.num_requests,
    arrival_rate_rps=script_args.arrival_rate_rps,
    canvas_lengths=canvas_lengths,
    refinement_steps=script_args.refinement_steps,
    arrival_process=script_args.arrival_process,
    burst_size=script_args.burst_size,
    burst_interval_ms=script_args.burst_interval_ms,
    slo_policy=script_args.slo_policy,
    slo_scale=script_args.slo_scale,
    rng=rng,
)
policies = [part.strip() for part in script_args.policies.split(",") if part.strip()]

summaries = []
operation_rows = []
request_rows = []
for policy in policies:
    result = simulate_policy(
        policy=policy,
        base_requests=requests,
        cost_model=cost_model,
        max_batch_size=script_args.max_batch_size,
        planner_window=script_args.planner_window,
        split_min_speedup=script_args.split_min_speedup,
        age_weight=script_args.age_weight,
        min_bucket_size=script_args.min_bucket_size,
        target_bucket_size=script_args.target_bucket_size,
        max_bucket_wait_ms=script_args.max_bucket_wait_ms,
        deadline_safety_margin_ms=script_args.deadline_safety_margin_ms,
    )
    summary = _summarize(result, cost_model.prompt_tokens)
    summaries.append(summary)
    for operation_index, operation in enumerate(result.operations):
        operation_rows.append(
            {
                "policy": policy,
                "operation_index": operation_index,
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
    for request in result.requests:
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
                "deadline_ms": request.deadline_ms,
                "service_class": request.service_class,
                "slo_miss": (
                    request.completion_ms > request.deadline_ms
                    if request.completion_ms is not None
                    and request.deadline_ms is not None
                    else None
                ),
                "canvas": request.canvas,
            }
        )

baseline = next(
    summary for summary in summaries if summary["policy"] == policies[0]
)
for summary in summaries:
    summary["throughput_speedup_vs_" + policies[0]] = (
        summary["throughput_rps"] / baseline["throughput_rps"]
        if baseline["throughput_rps"]
        else 0.0
    )
    summary["p95_latency_ratio_vs_" + policies[0]] = (
        summary["p95_latency_ms"] / baseline["p95_latency_ms"]
        if baseline["p95_latency_ms"]
        else 0.0
    )
    summary["gpu_time_ratio_vs_" + policies[0]] = (
        summary["total_gpu_ms"] / baseline["total_gpu_ms"]
        if baseline["total_gpu_ms"]
        else 0.0
    )

output = {
    "latency_table": script_args.latency_table,
    "workload": script_args.workload,
    "trace_lengths_path": script_args.trace_lengths_path,
    "num_requests": script_args.num_requests,
    "arrival_rate_rps": script_args.arrival_rate_rps,
    "arrival_process": script_args.arrival_process,
    "burst_size": script_args.burst_size,
    "burst_interval_ms": script_args.burst_interval_ms,
    "slo_policy": script_args.slo_policy,
    "slo_scale": script_args.slo_scale,
    "refinement_steps": script_args.refinement_steps,
    "max_batch_size": script_args.max_batch_size,
    "min_bucket_size": script_args.min_bucket_size,
    "target_bucket_size": script_args.target_bucket_size,
    "max_bucket_wait_ms": script_args.max_bucket_wait_ms,
    "deadline_safety_margin_ms": script_args.deadline_safety_margin_ms,
    "planner_window": script_args.planner_window,
    "split_min_speedup": script_args.split_min_speedup,
    "age_weight": script_args.age_weight,
    "canvas_classes": canvas_classes,
    "canvas_distribution": dict(Counter(canvas_lengths)),
    "prompt_tokens_in_cost_model": cost_model.prompt_tokens,
    "policies": policies,
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
            "slo_miss",
            "gpu_ms",
            "tok_waste",
            "attn_waste",
            "ops",
            "avg_batch",
            "modes",
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
                f"{summary['slo_miss_rate']:.3f}",
                f"{summary['total_gpu_ms']:.1f}",
                f"{summary['token_coupling_waste_ratio']:.3f}",
                f"{summary['attention_coupling_waste_ratio']:.3f}",
                str(summary["num_forward_ops"]),
                f"{summary['avg_batch_size']:.2f}",
                json.dumps(summary["mode_counts"], ensure_ascii=True),
            ]
        )
    )
