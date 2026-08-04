"""
Run an oracle go/no-go study for AR+dLLM portfolio serving.

Run from repo root:
  python examples/fastdllm/llada/simulate_portfolio_serving.py \
    --workload mixed \
    --arrival_rates 0.5,1,2,4 \
    --num_requests 512 \
    --output_prefix artifacts/portfolio/portfolio_oracle

This is a fast simulator for deciding whether portfolio serving is worth a
real GPU prototype. Calibrate the AR and dLLM cost parameters with measured
latency tables before using the numbers as paper evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
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


def _round_canvas(output_tokens: int, canvas_classes: list[int]) -> int:
    for canvas in sorted(canvas_classes):
        if output_tokens <= canvas:
            return canvas
    return max(canvas_classes)


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    canvas_tokens: int
    slo_ms: float


@dataclass
class CompletedRequest:
    request: Request
    backend: str
    start_ms: float
    finish_ms: float
    service_ms: float

    @property
    def latency_ms(self) -> float:
        return self.finish_ms - self.request.arrival_ms

    @property
    def wait_ms(self) -> float:
        return self.start_ms - self.request.arrival_ms

    @property
    def slo_met(self) -> bool:
        return self.latency_ms <= self.request.slo_ms


@dataclass
class BackendCostModel:
    ar_prefill_ms_per_token: float
    ar_decode_ms_per_token: float
    ar_batch_alpha: float
    dllm_step_ms_per_128seq: float
    dllm_steps: int
    dllm_pack_alpha: float
    dllm_ref_seq_len: int

    def ar_single_ms(self, request: Request) -> float:
        return (
            self.ar_prefill_ms_per_token * request.prompt_tokens
            + self.ar_decode_ms_per_token * request.output_tokens
        )

    def dllm_single_ms(self, request: Request) -> float:
        seq_len = request.prompt_tokens + request.canvas_tokens
        seq_scale = (seq_len / self.dllm_ref_seq_len) ** 2
        return self.dllm_steps * self.dllm_step_ms_per_128seq * seq_scale

    def ar_batch_ms(self, batch: list[Request]) -> float:
        if not batch:
            return 0.0
        prefill_ms = self.ar_prefill_ms_per_token * sum(item.prompt_tokens for item in batch)
        decode_ms = self.ar_decode_ms_per_token * max(item.output_tokens for item in batch)
        return prefill_ms + decode_ms * (len(batch) ** self.ar_batch_alpha)

    def dllm_batch_ms(self, batch: list[Request]) -> float:
        if not batch:
            return 0.0
        seq_work = sum(
            ((item.prompt_tokens + item.canvas_tokens) / self.dllm_ref_seq_len) ** 2
            for item in batch
        )
        return self.dllm_steps * self.dllm_step_ms_per_128seq * (seq_work ** self.dllm_pack_alpha)


@dataclass
class BackendState:
    name: str
    num_workers: int
    max_batch_size: int
    service_fn: Callable[[list[Request]], float]
    worker_free_ms: list[float]
    queue: list[Request]

    @classmethod
    def make(
        cls,
        *,
        name: str,
        num_workers: int,
        max_batch_size: int,
        service_fn: Callable[[list[Request]], float],
    ) -> "BackendState":
        return cls(
            name=name,
            num_workers=num_workers,
            max_batch_size=max_batch_size,
            service_fn=service_fn,
            worker_free_ms=[0.0 for _ in range(num_workers)],
            queue=[],
        )

    def earliest_free_ms(self) -> float:
        if not self.worker_free_ms:
            return math.inf
        return min(self.worker_free_ms)


def _make_requests(
    *,
    workload: str,
    num_requests: int,
    arrival_rate_rps: float,
    canvas_classes: list[int],
    slo_ms: float,
    seed: int,
) -> list[Request]:
    rng = random.Random(seed)
    requests = []
    now_ms = 0.0
    for request_id in range(num_requests):
        if arrival_rate_rps > 0:
            now_ms += rng.expovariate(arrival_rate_rps) * 1000.0

        if workload == "short":
            output_tokens = rng.randint(8, 64)
        elif workload == "medium":
            output_tokens = rng.randint(65, 192)
        elif workload == "long":
            output_tokens = rng.randint(193, 512)
        elif workload == "bimodal":
            output_tokens = rng.choice([rng.randint(8, 48), rng.randint(224, 512)])
        elif workload == "mixed":
            mode = rng.random()
            if mode < 0.45:
                output_tokens = rng.randint(8, 64)
            elif mode < 0.80:
                output_tokens = rng.randint(65, 192)
            else:
                output_tokens = rng.randint(193, 512)
        elif workload == "shift":
            if request_id < num_requests // 2:
                output_tokens = rng.randint(8, 96)
            else:
                output_tokens = rng.randint(160, 512)
        else:
            raise ValueError(f"Unknown workload: {workload}")

        prompt_tokens = int(min(1024, max(16, rng.lognormvariate(math.log(128), 0.7))))
        canvas_tokens = _round_canvas(output_tokens, canvas_classes)
        requests.append(
            Request(
                request_id=request_id,
                arrival_ms=now_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                canvas_tokens=canvas_tokens,
                slo_ms=slo_ms,
            )
        )
    return requests


def _dispatch_ready(
    *,
    backend: BackendState,
    now_ms: float,
    completions: list[CompletedRequest],
) -> None:
    if backend.num_workers <= 0:
        return
    made_progress = True
    while made_progress:
        made_progress = False
        for worker_index, free_ms in enumerate(list(backend.worker_free_ms)):
            if free_ms > now_ms or not backend.queue:
                continue
            start_ms = max(free_ms, backend.queue[0].arrival_ms)
            if start_ms > now_ms:
                continue
            batch_size = min(backend.max_batch_size, len(backend.queue))
            batch = backend.queue[:batch_size]
            del backend.queue[:batch_size]
            service_ms = backend.service_fn(batch)
            finish_ms = start_ms + service_ms
            backend.worker_free_ms[worker_index] = finish_ms
            for request in batch:
                completions.append(
                    CompletedRequest(
                        request=request,
                        backend=backend.name,
                        start_ms=start_ms,
                        finish_ms=finish_ms,
                        service_ms=service_ms,
                    )
                )
            made_progress = True


def _drain_backend(backend: BackendState, completions: list[CompletedRequest]) -> None:
    while backend.queue:
        next_free = backend.earliest_free_ms()
        if next_free == math.inf:
            raise ValueError(f"Backend {backend.name} has queued work but no workers.")
        next_arrival = backend.queue[0].arrival_ms
        _dispatch_ready(
            backend=backend,
            now_ms=max(next_free, next_arrival),
            completions=completions,
        )


def _predict_completion_ms(
    *,
    backend: BackendState,
    request: Request,
) -> float:
    if backend.num_workers <= 0:
        return math.inf
    clone = BackendState(
        name=backend.name,
        num_workers=backend.num_workers,
        max_batch_size=backend.max_batch_size,
        service_fn=backend.service_fn,
        worker_free_ms=list(backend.worker_free_ms),
        queue=list(backend.queue) + [request],
    )
    predicted: list[CompletedRequest] = []
    _drain_backend(clone, predicted)
    for completion in predicted:
        if completion.request.request_id == request.request_id:
            return completion.finish_ms
    return math.inf


def _simulate(
    *,
    policy: str,
    requests: list[Request],
    cost_model: BackendCostModel,
    ar_workers: int,
    dllm_workers: int,
    ar_max_batch_size: int,
    dllm_max_batch_size: int,
    static_length_threshold: int,
) -> dict[str, float | int | str]:
    ar = BackendState.make(
        name="ar",
        num_workers=ar_workers,
        max_batch_size=ar_max_batch_size,
        service_fn=cost_model.ar_batch_ms,
    )
    dllm = BackendState.make(
        name="dllm",
        num_workers=dllm_workers,
        max_batch_size=dllm_max_batch_size,
        service_fn=cost_model.dllm_batch_ms,
    )
    completions: list[CompletedRequest] = []
    single_winners = {}

    for request in requests:
        _dispatch_ready(backend=ar, now_ms=request.arrival_ms, completions=completions)
        _dispatch_ready(backend=dllm, now_ms=request.arrival_ms, completions=completions)

        ar_single = cost_model.ar_single_ms(request)
        dllm_single = cost_model.dllm_single_ms(request)
        single_winners[request.request_id] = "ar" if ar_single <= dllm_single else "dllm"

        if policy == "ar_only":
            target = ar
        elif policy == "dllm_only":
            target = dllm
        elif policy == "static_length":
            target = ar if request.output_tokens <= static_length_threshold else dllm
        elif policy == "least_loaded":
            target = ar if ar.earliest_free_ms() <= dllm.earliest_free_ms() else dllm
        elif policy == "oracle_latency":
            ar_finish = _predict_completion_ms(backend=ar, request=request)
            dllm_finish = _predict_completion_ms(backend=dllm, request=request)
            target = ar if ar_finish <= dllm_finish else dllm
        else:
            raise ValueError(f"Unknown policy: {policy}")

        if target.num_workers <= 0:
            target = dllm if target.name == "ar" else ar
        target.queue.append(request)

    _drain_backend(ar, completions)
    _drain_backend(dllm, completions)
    completions_by_id = {item.request.request_id: item for item in completions}
    ordered = [completions_by_id[item.request_id] for item in requests]
    latencies = [item.latency_ms for item in ordered]
    waits = [item.wait_ms for item in ordered]
    makespan_ms = max(item.finish_ms for item in ordered) - min(item.request.arrival_ms for item in ordered)
    ar_count = sum(1 for item in ordered if item.backend == "ar")
    dllm_count = len(ordered) - ar_count
    slo_met = sum(1 for item in ordered if item.slo_met)
    routed_against_single = sum(
        1 for item in ordered if item.backend != single_winners[item.request.request_id]
    )
    return {
        "policy": policy,
        "ar_workers": ar_workers,
        "dllm_workers": dllm_workers,
        "throughput_rps": 1000.0 * len(ordered) / makespan_ms if makespan_ms > 0 else 0.0,
        "slo_goodput_rps": 1000.0 * slo_met / makespan_ms if makespan_ms > 0 else 0.0,
        "slo_attainment": slo_met / len(ordered) if ordered else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "p95_wait_ms": _percentile(waits, 0.95),
        "ar_fraction": ar_count / len(ordered) if ordered else 0.0,
        "dllm_fraction": dllm_count / len(ordered) if ordered else 0.0,
        "load_dependent_route_fraction": routed_against_single / len(ordered) if ordered else 0.0,
    }


def _run_capacity_oracle(
    *,
    requests: list[Request],
    cost_model: BackendCostModel,
    total_workers: int,
    ar_max_batch_size: int,
    dllm_max_batch_size: int,
    static_length_threshold: int,
) -> dict[str, float | int | str]:
    candidates = []
    for ar_workers in range(total_workers + 1):
        dllm_workers = total_workers - ar_workers
        if ar_workers == 0 and dllm_workers == 0:
            continue
        if ar_workers == 0:
            policy = "dllm_only"
        elif dllm_workers == 0:
            policy = "ar_only"
        else:
            policy = "oracle_latency"
        row = _simulate(
            policy=policy,
            requests=requests,
            cost_model=cost_model,
            ar_workers=ar_workers,
            dllm_workers=dllm_workers,
            ar_max_batch_size=ar_max_batch_size,
            dllm_max_batch_size=dllm_max_batch_size,
            static_length_threshold=static_length_threshold,
        )
        row["policy"] = "oracle_capacity_split"
        candidates.append(row)
    return max(candidates, key=lambda row: (float(row["slo_goodput_rps"]), -float(row["p95_latency_ms"])))


def _write_outputs(
    *,
    output_prefix: str,
    rows: list[dict[str, float | int | str]],
    config: dict[str, float | int | str],
) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    csv_path = prefix.with_name(prefix.name + "_summary.csv")
    summary_path.write_text(
        json.dumps({"config": config, "summaries": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved CSV: {csv_path}")


def _print_table(rows: list[dict[str, float | int | str]]) -> None:
    columns = [
        "arrival_rate_rps",
        "policy",
        "ar_workers",
        "dllm_workers",
        "throughput_rps",
        "slo_goodput_rps",
        "slo_attainment",
        "p95_latency_ms",
        "p99_latency_ms",
        "ar_fraction",
        "load_dependent_route_fraction",
    ]
    print("\t".join(columns))
    for row in rows:
        parts = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                parts.append(f"{value:.3f}")
            else:
                parts.append(str(value))
        print("\t".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="mixed", choices=["short", "medium", "long", "mixed", "bimodal", "shift"])
    parser.add_argument("--num_requests", type=int, default=512)
    parser.add_argument("--arrival_rates", default="0.5,1,2,4")
    parser.add_argument("--canvas_classes", default="32,64,128,256,512")
    parser.add_argument("--slo_ms", type=float, default=4000.0)
    parser.add_argument("--total_workers", type=int, default=2)
    parser.add_argument("--ar_workers", type=int, default=1)
    parser.add_argument("--dllm_workers", type=int, default=1)
    parser.add_argument("--ar_max_batch_size", type=int, default=16)
    parser.add_argument("--dllm_max_batch_size", type=int, default=16)
    parser.add_argument("--static_length_threshold", type=int, default=128)
    parser.add_argument("--ar_prefill_ms_per_token", type=float, default=0.02)
    parser.add_argument("--ar_decode_ms_per_token", type=float, default=7.0)
    parser.add_argument("--ar_batch_alpha", type=float, default=0.35)
    parser.add_argument("--dllm_step_ms_per_128seq", type=float, default=30.0)
    parser.add_argument("--dllm_steps", type=int, default=16)
    parser.add_argument("--dllm_pack_alpha", type=float, default=0.70)
    parser.add_argument("--dllm_ref_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", default="artifacts/portfolio/portfolio_oracle")
    args = parser.parse_args()

    arrival_rates = _parse_float_list(args.arrival_rates)
    canvas_classes = _parse_int_list(args.canvas_classes)
    cost_model = BackendCostModel(
        ar_prefill_ms_per_token=args.ar_prefill_ms_per_token,
        ar_decode_ms_per_token=args.ar_decode_ms_per_token,
        ar_batch_alpha=args.ar_batch_alpha,
        dllm_step_ms_per_128seq=args.dllm_step_ms_per_128seq,
        dllm_steps=args.dllm_steps,
        dllm_pack_alpha=args.dllm_pack_alpha,
        dllm_ref_seq_len=args.dllm_ref_seq_len,
    )

    rows = []
    policies = ["ar_only", "dllm_only", "static_length", "least_loaded", "oracle_latency"]
    for arrival_rate in arrival_rates:
        requests = _make_requests(
            workload=args.workload,
            num_requests=args.num_requests,
            arrival_rate_rps=arrival_rate,
            canvas_classes=canvas_classes,
            slo_ms=args.slo_ms,
            seed=args.seed,
        )
        for policy in policies:
            if policy == "ar_only":
                ar_workers, dllm_workers = args.total_workers, 0
            elif policy == "dllm_only":
                ar_workers, dllm_workers = 0, args.total_workers
            else:
                ar_workers, dllm_workers = args.ar_workers, args.dllm_workers
            row = _simulate(
                policy=policy,
                requests=requests,
                cost_model=cost_model,
                ar_workers=ar_workers,
                dllm_workers=dllm_workers,
                ar_max_batch_size=args.ar_max_batch_size,
                dllm_max_batch_size=args.dllm_max_batch_size,
                static_length_threshold=args.static_length_threshold,
            )
            row["arrival_rate_rps"] = arrival_rate
            rows.append(row)

        capacity_row = _run_capacity_oracle(
            requests=requests,
            cost_model=cost_model,
            total_workers=args.total_workers,
            ar_max_batch_size=args.ar_max_batch_size,
            dllm_max_batch_size=args.dllm_max_batch_size,
            static_length_threshold=args.static_length_threshold,
        )
        capacity_row["arrival_rate_rps"] = arrival_rate
        rows.append(capacity_row)

    config = {
        "workload": args.workload,
        "num_requests": args.num_requests,
        "arrival_rates": args.arrival_rates,
        "slo_ms": args.slo_ms,
        "total_workers": args.total_workers,
        "cost_model": "synthetic; calibrate with measured AR and dLLM latency before paper use",
    }
    _write_outputs(output_prefix=args.output_prefix, rows=rows, config=config)
    _print_table(rows)

    by_rate: dict[float, dict[str, dict[str, float | int | str]]] = {}
    for row in rows:
        by_rate.setdefault(float(row["arrival_rate_rps"]), {})[str(row["policy"])] = row
    print("\nGo/no-go signals:")
    for arrival_rate, policy_rows in sorted(by_rate.items()):
        static = policy_rows.get("static_length")
        oracle = policy_rows.get("oracle_latency")
        capacity = policy_rows.get("oracle_capacity_split")
        if not static or not oracle or not capacity:
            continue
        static_goodput = float(static["slo_goodput_rps"])
        oracle_goodput = float(oracle["slo_goodput_rps"])
        capacity_goodput = float(capacity["slo_goodput_rps"])
        oracle_gain = oracle_goodput / static_goodput if static_goodput > 0 else math.inf
        capacity_gain = capacity_goodput / oracle_goodput if oracle_goodput > 0 else math.inf
        load_flips = float(oracle["load_dependent_route_fraction"])
        print(
            f"  rps={arrival_rate:.3f}: oracle/static_goodput={oracle_gain:.3f}, "
            f"capacity/oracle_goodput={capacity_gain:.3f}, "
            f"load_dependent_routes={load_flips:.3f}"
        )


if __name__ == "__main__":
    main()
