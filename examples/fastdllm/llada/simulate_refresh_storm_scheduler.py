"""
Replay dual-cache refresh storms from measured Fast-dLLM phase profiles.

Run from repo root after profile_refresh_storm.py:
  python examples/fastdllm/llada/simulate_refresh_storm_scheduler.py \
    --phase_profile artifacts/refresh_storm/llada_dual_s128_summary.json \
    --num_requests 64 \
    --max_batch_size 8 \
    --memory_budget_mb 512 \
    --policies synchronized,staggered,memory_gated,static_refresh_cap,phase_aware_admission \
    --output_prefix artifacts/refresh_storm/dual_replay_s128
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


def _parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
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


class PhaseCostModel:
    def __init__(self, profile_path: str | Path):
        self.path = Path(profile_path)
        data = json.loads(self.path.read_text())
        self.profile = data
        self.latency_ms: dict[tuple[str, int], float] = {}
        self.memory_mb: dict[tuple[str, int], float] = {}
        self.observed_batches: dict[str, list[int]] = defaultdict(list)
        for row in data["phase_rows"]:
            phase = str(row["phase"])
            batch_size = int(row["batch_size"])
            self.latency_ms[(phase, batch_size)] = float(row["mean_latency_ms"])
            self.memory_mb[(phase, batch_size)] = float(row["p95_memory_peak_delta_mb"])
            self.observed_batches[phase].append(batch_size)
        if not self.latency_ms:
            raise ValueError(f"No phase rows found in {self.path}")

        phase_counts = Counter(str(record["phase"]) for record in data.get("rows", []))
        self.warmup_calls_per_request = int(
            data.get("warmup_calls_per_request") or phase_counts.get("warmup") or 4
        )
        self.refine_calls_per_request = int(
            data.get("refine_calls_per_request") or phase_counts.get("refine") or 41
        )
        # profile_refresh_storm stores aggregate phase_rows, so infer from phase rows if needed.
        for row in data["phase_rows"]:
            if str(row["phase"]) == "warmup":
                self.warmup_calls_per_request = int(row["num_calls"])
                break
        for row in data["phase_rows"]:
            if str(row["phase"]) == "refine":
                self.refine_calls_per_request = int(row["num_calls"])
                break

    def _nearest_batch(self, phase: str, batch_size: int) -> int:
        observed = sorted(set(self.observed_batches[phase]))
        if not observed:
            raise ValueError(f"No observed batch sizes for phase={phase}")
        return min(observed, key=lambda value: abs(value - batch_size))

    def predict_latency_ms(self, phase: str, batch_size: int) -> float:
        if batch_size <= 0:
            return 0.0
        key = (phase, batch_size)
        if key in self.latency_ms:
            return self.latency_ms[key]
        nearest = self._nearest_batch(phase, batch_size)
        return self.latency_ms[(phase, nearest)] * batch_size / nearest

    def predict_memory_mb(self, phase: str, batch_size: int) -> float:
        if batch_size <= 0:
            return 0.0
        key = (phase, batch_size)
        if key in self.memory_mb:
            return self.memory_mb[key]
        nearest = self._nearest_batch(phase, batch_size)
        return self.memory_mb[(phase, nearest)] * batch_size / nearest

    def max_batch_under_budget(
        self,
        *,
        phase: str,
        desired_batch_size: int,
        memory_budget_mb: float | None,
    ) -> int:
        if memory_budget_mb is None:
            return desired_batch_size
        for batch_size in range(desired_batch_size, 0, -1):
            if self.predict_memory_mb(phase, batch_size) <= memory_budget_mb:
                return batch_size
        return 0

    def refresh_cap_under_budget(self, memory_budget_mb: float | None) -> int:
        desired = max(self.observed_batches.get("warmup", [1]))
        return self.max_batch_under_budget(
            phase="warmup",
            desired_batch_size=desired,
            memory_budget_mb=memory_budget_mb,
        )


@dataclass
class Request:
    request_id: int
    arrival_ms: float
    phases: list[str]
    slo_deadline_ms: float | None = None
    rejected: bool = False
    admitted: bool = True
    admission_ms: float | None = None
    phase_index: int = 0
    phase_ready_ms: float = 0.0
    first_start_ms: float | None = None
    completion_ms: float | None = None
    refresh_wait_ms: float = 0.0
    refine_wait_ms: float = 0.0
    service_ms: float = 0.0

    @property
    def done(self) -> bool:
        return self.rejected or self.phase_index >= len(self.phases)

    @property
    def next_phase(self) -> str | None:
        if self.done:
            return None
        return self.phases[self.phase_index]

    @property
    def admission_wait_ms(self) -> float:
        if self.admission_ms is None:
            return 0.0
        return max(0.0, self.admission_ms - self.arrival_ms)


@dataclass
class Operation:
    policy: str
    operation_index: int
    phase: str
    start_ms: float
    end_ms: float
    latency_ms: float
    memory_mb: float
    batch_size: int
    request_ids: list[int]
    oom: bool


@dataclass
class SimulationResult:
    policy: str
    requests: list[Request]
    operations: list[Operation] = field(default_factory=list)
    gpu_idle_ms: float = 0.0
    oom_events: int = 0
    admission_rejections: int = 0
    rejected_requests: list[Request] = field(default_factory=list)


def _make_phase_sequence(warmup_calls: int, refine_calls: int) -> list[str]:
    phases = []
    base_refines = refine_calls // max(warmup_calls, 1)
    remainder = refine_calls % max(warmup_calls, 1)
    for block in range(warmup_calls):
        phases.append("warmup")
        count = base_refines + (1 if block < remainder else 0)
        phases.extend(["refine"] * count)
    return phases


def _make_open_arrivals(
    *,
    num_requests: int,
    phases: list[str],
    arrival_rate_rps: float,
    arrival_process: str,
    burst_size: int,
    burst_interval_ms: float,
    slo_ms: float,
    rng: random.Random,
) -> list[Request]:
    requests = []
    now_ms = 0.0
    for request_id in range(num_requests):
        if request_id > 0:
            if arrival_process == "poisson":
                now_ms += rng.expovariate(arrival_rate_rps) * 1000.0
            elif arrival_process == "bursty":
                if request_id % max(burst_size, 1) == 0:
                    now_ms += burst_interval_ms
            else:
                raise ValueError(f"Unknown arrival_process: {arrival_process}")
        request = Request(
            request_id=request_id,
            arrival_ms=now_ms,
            phases=list(phases),
            slo_deadline_ms=now_ms + slo_ms,
            admitted=False,
            admission_ms=None,
            phase_ready_ms=now_ms,
        )
        requests.append(request)
    return requests


def _make_requests(
    *,
    policy: str,
    num_requests: int,
    phases: list[str],
    stagger_ms: float,
) -> list[Request]:
    requests = []
    for request_id in range(num_requests):
        arrival_ms = request_id * stagger_ms if policy == "staggered" else 0.0
        requests.append(
            Request(
                request_id=request_id,
                arrival_ms=arrival_ms,
                phases=list(phases),
                admitted=False,
                admission_ms=None,
                phase_ready_ms=arrival_ms,
            )
        )
    return requests


def _uses_admission(policy: str) -> bool:
    return policy in (
        "static_refresh_cap",
        "phase_aware_admission",
        "static_active_cap",
        "current_memory_admission",
        "generic_slo_admission",
        "generic_slo_same_reject",
        "refresh_aware_admission",
        "slo_refresh_aware_admission",
    )


def _active_requests(requests: list[Request]) -> list[Request]:
    return [request for request in requests if request.admitted and not request.done]


def _pending_requests(requests: list[Request], now_ms: float) -> list[Request]:
    return [
        request
        for request in requests
        if not request.rejected
        and not request.admitted
        and request.arrival_ms <= now_ms
    ]


def _admit_request(request: Request, now_ms: float) -> None:
    request.admitted = True
    request.admission_ms = now_ms
    request.phase_ready_ms = now_ms


def _next_warmup_index(request: Request) -> int | None:
    for index in range(request.phase_index, len(request.phases)):
        if request.phases[index] == "warmup":
            return index
    return None


def _phase_aware_can_admit(
    *,
    candidate: Request,
    active: list[Request],
    refresh_cap: int,
) -> bool:
    counts: Counter[int] = Counter()
    for request in active:
        next_warmup = _next_warmup_index(request)
        if next_warmup is not None:
            counts[next_warmup] += 1
    candidate_next_warmup = _next_warmup_index(candidate)
    if candidate_next_warmup is None:
        return True
    return counts[candidate_next_warmup] + 1 <= refresh_cap


def _estimate_remaining_service_ms(
    request: Request,
    *,
    cost_model: PhaseCostModel,
    expected_batch_size: int,
) -> float:
    total = 0.0
    for phase in request.phases[request.phase_index :]:
        total += cost_model.predict_latency_ms(phase, expected_batch_size)
    return total


def _estimate_average_remaining_service_ms(
    request: Request,
    *,
    cost_model: PhaseCostModel,
    expected_batch_size: int,
) -> float:
    phases = request.phases[request.phase_index :]
    if not phases:
        return 0.0
    warmup_ms = cost_model.predict_latency_ms("warmup", expected_batch_size)
    refine_ms = cost_model.predict_latency_ms("refine", expected_batch_size)
    average_phase_ms = (warmup_ms + refine_ms) / 2.0
    return len(phases) * average_phase_ms


def _can_meet_slo(
    request: Request,
    *,
    now_ms: float,
    cost_model: PhaseCostModel,
    expected_batch_size: int,
) -> bool:
    deadline = request.slo_deadline_ms
    if deadline is None:
        return True
    estimated_completion = now_ms + _estimate_remaining_service_ms(
        request,
        cost_model=cost_model,
        expected_batch_size=expected_batch_size,
    )
    return estimated_completion <= float(deadline)


def _generic_can_meet_slo(
    request: Request,
    *,
    now_ms: float,
    cost_model: PhaseCostModel,
    expected_batch_size: int,
) -> bool:
    deadline = request.slo_deadline_ms
    if deadline is None:
        return True
    estimated_completion = now_ms + _estimate_average_remaining_service_ms(
        request,
        cost_model=cost_model,
        expected_batch_size=expected_batch_size,
    )
    return estimated_completion <= float(deadline)


def _admit_pending(
    *,
    policy: str,
    requests: list[Request],
    now_ms: float,
    max_batch_size: int,
    static_active_cap: int,
    refresh_cap: int,
) -> int:
    pending = sorted(_pending_requests(requests, now_ms), key=lambda r: (r.arrival_ms, r.request_id))
    admitted = 0

    if not _uses_admission(policy):
        for request in pending:
            _admit_request(request, max(now_ms, request.arrival_ms))
            admitted += 1
        return admitted

    active = _active_requests(requests)
    active_limit = static_active_cap if policy == "static_refresh_cap" else max_batch_size
    for request in pending:
        if len(active) >= active_limit:
            break
        if policy == "phase_aware_admission" and not _phase_aware_can_admit(
            candidate=request,
            active=active,
            refresh_cap=refresh_cap,
        ):
            continue
        _admit_request(request, now_ms)
        active.append(request)
        admitted += 1
    return admitted


def _admit_pending_open(
    *,
    policy: str,
    requests: list[Request],
    result: SimulationResult,
    now_ms: float,
    max_batch_size: int,
    static_active_cap: int,
    refresh_cap: int,
    cost_model: PhaseCostModel,
    target_rejections: int,
) -> int:
    pending = sorted(_pending_requests(requests, now_ms), key=lambda r: (r.arrival_ms, r.request_id))
    admitted = 0
    active = _active_requests(requests)

    if policy == "admit_all_memory_gated":
        for request in pending:
            _admit_request(request, now_ms)
            admitted += 1
        return admitted

    if policy == "static_active_cap":
        limit = static_active_cap
    elif policy == "current_memory_admission":
        limit = max_batch_size
    elif policy in (
        "refresh_aware_admission",
        "slo_refresh_aware_admission",
        "generic_slo_admission",
        "generic_slo_same_reject",
    ):
        limit = min(max_batch_size, max(1, refresh_cap))
    else:
        limit = static_active_cap

    for request in pending:
        if len(active) >= limit:
            break
        if policy == "generic_slo_same_reject" and len(result.rejected_requests) < target_rejections:
            request.rejected = True
            result.rejected_requests.append(request)
            result.admission_rejections += 1
            continue
        if policy in ("generic_slo_admission", "generic_slo_same_reject") and not _generic_can_meet_slo(
            request,
            now_ms=now_ms,
            cost_model=cost_model,
            expected_batch_size=max(1, min(limit, max_batch_size)),
        ):
            request.rejected = True
            result.rejected_requests.append(request)
            result.admission_rejections += 1
            continue
        if policy == "slo_refresh_aware_admission" and not _can_meet_slo(
            request,
            now_ms=now_ms,
            cost_model=cost_model,
            expected_batch_size=max(1, min(limit, max_batch_size)),
        ):
            request.rejected = True
            result.rejected_requests.append(request)
            result.admission_rejections += 1
            continue
        if policy == "current_memory_admission":
            # This baseline limits active population by instantaneous capacity only;
            # it does not reserve future refresh slots.
            pass
        _admit_request(request, now_ms)
        active.append(request)
        admitted += 1
    return admitted


def _ready_requests(requests: list[Request], now_ms: float) -> list[Request]:
    return [
        request
        for request in requests
        if request.admitted
        and not request.done
        and request.arrival_ms <= now_ms
        and request.phase_ready_ms <= now_ms
    ]


def _next_ready_time(requests: list[Request], now_ms: float) -> float | None:
    times = [
        max(request.arrival_ms, request.phase_ready_ms)
        for request in requests
        if request.admitted
        and not request.done
        and max(request.arrival_ms, request.phase_ready_ms) > now_ms
    ]
    times.extend(
        request.arrival_ms
        for request in requests
        if not request.rejected and not request.admitted and request.arrival_ms > now_ms
    )
    return min(times) if times else None


def _oldest(requests: list[Request]) -> list[Request]:
    return sorted(requests, key=lambda request: (request.phase_ready_ms, request.arrival_ms, request.request_id))


def _phase_buckets(ready: list[Request]) -> dict[str, list[Request]]:
    buckets: dict[str, list[Request]] = defaultdict(list)
    for request in ready:
        phase = request.next_phase
        if phase is not None:
            buckets[phase].append(request)
    return buckets


def _plan_oldest_phase(
    *,
    ready: list[Request],
    max_batch_size: int,
    cost_model: PhaseCostModel,
    memory_budget_mb: float | None,
    enforce_budget: bool,
) -> tuple[str, list[Request]]:
    oldest = _oldest(ready)[0]
    phase = oldest.next_phase
    assert phase is not None
    candidates = _oldest([request for request in ready if request.next_phase == phase])
    desired = min(max_batch_size, len(candidates))
    if enforce_budget:
        desired = cost_model.max_batch_under_budget(
            phase=phase,
            desired_batch_size=desired,
            memory_budget_mb=memory_budget_mb,
        )
    if desired <= 0:
        return phase, []
    return phase, candidates[:desired]


def _plan_memory_gated(
    *,
    ready: list[Request],
    now_ms: float,
    max_batch_size: int,
    cost_model: PhaseCostModel,
    memory_budget_mb: float | None,
    age_weight: float,
) -> tuple[str, list[Request]]:
    buckets = _phase_buckets(ready)
    best_score = -1.0
    best_phase = ""
    best_group: list[Request] = []
    for phase, bucket in buckets.items():
        ordered = _oldest(bucket)
        desired = min(max_batch_size, len(ordered))
        desired = cost_model.max_batch_under_budget(
            phase=phase,
            desired_batch_size=desired,
            memory_budget_mb=memory_budget_mb,
        )
        if desired <= 0:
            continue
        group = ordered[:desired]
        latency_ms = cost_model.predict_latency_ms(phase, len(group))
        oldest_age_ms = max(0.0, now_ms - ordered[0].phase_ready_ms)
        # Prefer efficient batches, but keep old refreshes from starving.
        score = len(group) / max(latency_ms, 1e-9) + age_weight * oldest_age_ms / 1000.0
        if score > best_score:
            best_score = score
            best_phase = phase
            best_group = group
    return best_phase, best_group


def simulate_policy(
    *,
    policy: str,
    cost_model: PhaseCostModel,
    num_requests: int,
    max_batch_size: int,
    memory_budget_mb: float | None,
    stagger_ms: float,
    age_weight: float,
    static_active_cap: int,
) -> SimulationResult:
    phases = _make_phase_sequence(
        warmup_calls=cost_model.warmup_calls_per_request,
        refine_calls=cost_model.refine_calls_per_request,
    )
    requests = _make_requests(
        policy=policy,
        num_requests=num_requests,
        phases=phases,
        stagger_ms=stagger_ms,
    )
    result = SimulationResult(policy=policy, requests=requests)
    now_ms = min((request.arrival_ms for request in requests), default=0.0)
    refresh_cap = max(1, cost_model.refresh_cap_under_budget(memory_budget_mb))
    if static_active_cap <= 0:
        static_active_cap = refresh_cap

    while any(not request.done for request in requests):
        _admit_pending(
            policy=policy,
            requests=requests,
            now_ms=now_ms,
            max_batch_size=max_batch_size,
            static_active_cap=static_active_cap,
            refresh_cap=refresh_cap,
        )
        ready = _ready_requests(requests, now_ms)
        if not ready:
            next_time = _next_ready_time(requests, now_ms)
            if next_time is None:
                break
            result.gpu_idle_ms += max(0.0, next_time - now_ms)
            now_ms = next_time
            _admit_pending(
                policy=policy,
                requests=requests,
                now_ms=now_ms,
                max_batch_size=max_batch_size,
                static_active_cap=static_active_cap,
                refresh_cap=refresh_cap,
            )
            ready = _ready_requests(requests, now_ms)
            if not ready:
                continue

        if policy in ("memory_gated", "phase_aware_admission"):
            phase, group = _plan_memory_gated(
                ready=ready,
                now_ms=now_ms,
                max_batch_size=max_batch_size,
                cost_model=cost_model,
                memory_budget_mb=memory_budget_mb,
                age_weight=age_weight,
            )
        else:
            phase, group = _plan_oldest_phase(
                ready=ready,
                max_batch_size=max_batch_size,
                cost_model=cost_model,
                memory_budget_mb=memory_budget_mb,
                enforce_budget=False,
            )

        if not group:
            result.admission_rejections += len(ready)
            break

        latency_ms = cost_model.predict_latency_ms(phase, len(group))
        memory_mb = cost_model.predict_memory_mb(phase, len(group))
        oom = memory_budget_mb is not None and memory_mb > memory_budget_mb
        if oom:
            result.oom_events += 1
        start_ms = now_ms
        end_ms = now_ms + latency_ms
        for request in group:
            if request.first_start_ms is None:
                request.first_start_ms = start_ms
            wait_ms = max(0.0, start_ms - request.phase_ready_ms)
            if phase == "warmup":
                request.refresh_wait_ms += wait_ms
            else:
                request.refine_wait_ms += wait_ms
            request.service_ms += latency_ms
            request.phase_index += 1
            request.phase_ready_ms = end_ms
            if request.done:
                request.completion_ms = end_ms

        result.operations.append(
            Operation(
                policy=policy,
                operation_index=len(result.operations),
                phase=phase,
                start_ms=start_ms,
                end_ms=end_ms,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                batch_size=len(group),
                request_ids=[request.request_id for request in group],
                oom=oom,
            )
        )
        now_ms = end_ms

    return result


def simulate_open_policy(
    *,
    policy: str,
    cost_model: PhaseCostModel,
    num_requests: int,
    max_batch_size: int,
    memory_budget_mb: float | None,
    arrival_rate_rps: float,
    arrival_process: str,
    burst_size: int,
    burst_interval_ms: float,
    slo_ms: float,
    age_weight: float,
    static_active_cap: int,
    target_rejections: int,
    seed: int,
) -> SimulationResult:
    phases = _make_phase_sequence(
        warmup_calls=cost_model.warmup_calls_per_request,
        refine_calls=cost_model.refine_calls_per_request,
    )
    rng = random.Random(seed)
    requests = _make_open_arrivals(
        num_requests=num_requests,
        phases=phases,
        arrival_rate_rps=arrival_rate_rps,
        arrival_process=arrival_process,
        burst_size=burst_size,
        burst_interval_ms=burst_interval_ms,
        slo_ms=slo_ms,
        rng=rng,
    )
    result = SimulationResult(policy=policy, requests=requests)
    now_ms = min((request.arrival_ms for request in requests), default=0.0)
    refresh_cap = max(1, cost_model.refresh_cap_under_budget(memory_budget_mb))
    if static_active_cap <= 0:
        static_active_cap = max_batch_size

    while any(not request.done for request in requests):
        _admit_pending_open(
            policy=policy,
            requests=requests,
            result=result,
            now_ms=now_ms,
            max_batch_size=max_batch_size,
            static_active_cap=static_active_cap,
            refresh_cap=refresh_cap,
            cost_model=cost_model,
            target_rejections=target_rejections,
        )
        ready = _ready_requests(requests, now_ms)
        if not ready:
            next_time = _next_ready_time(requests, now_ms)
            if next_time is None:
                break
            result.gpu_idle_ms += max(0.0, next_time - now_ms)
            now_ms = next_time
            continue

        phase, group = _plan_memory_gated(
            ready=ready,
            now_ms=now_ms,
            max_batch_size=max_batch_size,
            cost_model=cost_model,
            memory_budget_mb=memory_budget_mb,
            age_weight=age_weight,
        )
        if not group:
            break

        latency_ms = cost_model.predict_latency_ms(phase, len(group))
        memory_mb = cost_model.predict_memory_mb(phase, len(group))
        oom = memory_budget_mb is not None and memory_mb > memory_budget_mb
        if oom:
            result.oom_events += 1
        start_ms = now_ms
        end_ms = now_ms + latency_ms
        for request in group:
            if request.first_start_ms is None:
                request.first_start_ms = start_ms
            wait_ms = max(0.0, start_ms - request.phase_ready_ms)
            if phase == "warmup":
                request.refresh_wait_ms += wait_ms
            else:
                request.refine_wait_ms += wait_ms
            request.service_ms += latency_ms
            request.phase_index += 1
            request.phase_ready_ms = end_ms
            if request.done:
                request.completion_ms = end_ms

        result.operations.append(
            Operation(
                policy=policy,
                operation_index=len(result.operations),
                phase=phase,
                start_ms=start_ms,
                end_ms=end_ms,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                batch_size=len(group),
                request_ids=[request.request_id for request in group],
                oom=oom,
            )
        )
        now_ms = end_ms

    return result


def summarize(result: SimulationResult) -> dict[str, Any]:
    completed = [request for request in result.requests if request.completion_ms is not None]
    latencies = [
        request.completion_ms - request.arrival_ms
        for request in completed
        if request.completion_ms is not None
    ]
    slo_completed = [
        request
        for request in completed
        if request.slo_deadline_ms is not None
        and request.completion_ms is not None
        and request.completion_ms <= request.slo_deadline_ms
    ]
    slo_missed = [
        request
        for request in completed
        if request.slo_deadline_ms is not None
        and request.completion_ms is not None
        and request.completion_ms > request.slo_deadline_ms
    ]
    refresh_waits = [request.refresh_wait_ms for request in completed]
    refine_waits = [request.refine_wait_ms for request in completed]
    admission_waits = [request.admission_wait_ms for request in completed]
    services = [request.service_ms for request in completed]
    first_arrival = min((request.arrival_ms for request in result.requests), default=0.0)
    last_completion = max(
        (request.completion_ms or first_arrival for request in result.requests),
        default=first_arrival,
    )
    makespan_ms = max(last_completion - first_arrival, 1e-9)
    total_gpu_ms = sum(operation.latency_ms for operation in result.operations)
    phase_counts = Counter(operation.phase for operation in result.operations)
    logical_phase_counts = Counter()
    for request in result.requests:
        logical_phase_counts.update(request.phases)
    phase_batch_sizes: dict[str, list[int]] = defaultdict(list)
    for operation in result.operations:
        phase_batch_sizes[operation.phase].append(operation.batch_size)
    active_events = []
    for request in result.requests:
        if request.admission_ms is not None and request.completion_ms is not None:
            active_events.append((request.admission_ms, 1))
            active_events.append((request.completion_ms, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(active_events):
        active += delta
        max_active = max(max_active, active)

    return {
        "policy": result.policy,
        "completed_requests": len(completed),
        "makespan_ms": makespan_ms,
        "throughput_rps": len(completed) / (makespan_ms / 1000.0),
        "slo_goodput_rps": len(slo_completed) / (makespan_ms / 1000.0),
        "rejected_requests": len(result.rejected_requests),
        "rejected_rps": len(result.rejected_requests) / (makespan_ms / 1000.0),
        "slo_miss_rate": (
            len(slo_missed) / (len(slo_missed) + len(slo_completed))
            if (len(slo_missed) + len(slo_completed)) > 0
            else 0.0
        ),
        "gpu_busy_fraction": total_gpu_ms / makespan_ms,
        "gpu_idle_ms": result.gpu_idle_ms,
        "total_gpu_ms": total_gpu_ms,
        "num_physical_forwards": len(result.operations),
        "num_logical_warmup_calls": logical_phase_counts["warmup"],
        "num_logical_refine_calls": logical_phase_counts["refine"],
        "num_physical_warmup_forwards": phase_counts["warmup"],
        "num_physical_refine_forwards": phase_counts["refine"],
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_service_ms": statistics.mean(services) if services else 0.0,
        "p95_service_ms": _percentile(services, 0.95),
        "mean_admission_wait_ms": statistics.mean(admission_waits)
        if admission_waits
        else 0.0,
        "p95_admission_wait_ms": _percentile(admission_waits, 0.95),
        "mean_refresh_wait_ms": statistics.mean(refresh_waits) if refresh_waits else 0.0,
        "p95_refresh_wait_ms": _percentile(refresh_waits, 0.95),
        "p99_refresh_wait_ms": _percentile(refresh_waits, 0.99),
        "mean_refine_wait_ms": statistics.mean(refine_waits) if refine_waits else 0.0,
        "p95_refine_wait_ms": _percentile(refine_waits, 0.95),
        "peak_memory_mb": max((operation.memory_mb for operation in result.operations), default=0.0),
        "mean_memory_mb": statistics.mean([operation.memory_mb for operation in result.operations])
        if result.operations
        else 0.0,
        "oom_events": result.oom_events,
        "admission_rejections": result.admission_rejections,
        "max_active_requests": max_active,
        "mean_warmup_batch_size": statistics.mean(phase_batch_sizes["warmup"])
        if phase_batch_sizes["warmup"]
        else 0.0,
        "mean_refine_batch_size": statistics.mean(phase_batch_sizes["refine"])
        if phase_batch_sizes["refine"]
        else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_profile", required=True)
    parser.add_argument("--output_prefix", default="artifacts/refresh_storm/dual_replay")
    parser.add_argument("--mode", choices=["closed", "open"], default="closed")
    parser.add_argument(
        "--policies",
        default=(
            "synchronized,staggered,memory_gated,"
            "static_refresh_cap,phase_aware_admission"
        ),
    )
    parser.add_argument("--num_requests", type=int, default=64)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--memory_budget_mb", type=float, default=512.0)
    parser.add_argument("--arrival_rate_rps", type=float, default=1.5)
    parser.add_argument("--arrival_process", choices=["poisson", "bursty"], default="poisson")
    parser.add_argument("--burst_size", type=int, default=16)
    parser.add_argument("--burst_interval_ms", type=float, default=1_000.0)
    parser.add_argument("--slo_ms", type=float, default=45_000.0)
    parser.add_argument("--stagger_ms", type=float, default=50.0)
    parser.add_argument("--age_weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--static_active_cap",
        type=int,
        default=0,
        help="Active request cap for static_refresh_cap. 0 uses the measured refresh memory cap.",
    )
    parser.add_argument(
        "--target_rejections",
        type=int,
        default=0,
        help="Forced rejection count for generic_slo_same_reject.",
    )
    args = parser.parse_args()

    cost_model = PhaseCostModel(args.phase_profile)
    summaries = []
    operation_rows = []
    request_rows = []
    for policy in _parse_csv(args.policies):
        if args.mode == "open":
            result = simulate_open_policy(
                policy=policy,
                cost_model=cost_model,
                num_requests=args.num_requests,
                max_batch_size=args.max_batch_size,
                memory_budget_mb=args.memory_budget_mb,
                arrival_rate_rps=args.arrival_rate_rps,
                arrival_process=args.arrival_process,
                burst_size=args.burst_size,
                burst_interval_ms=args.burst_interval_ms,
                slo_ms=args.slo_ms,
                age_weight=args.age_weight,
                static_active_cap=args.static_active_cap,
                target_rejections=args.target_rejections,
                seed=args.seed,
            )
        else:
            result = simulate_policy(
                policy=policy,
                cost_model=cost_model,
                num_requests=args.num_requests,
                max_batch_size=args.max_batch_size,
                memory_budget_mb=args.memory_budget_mb,
                stagger_ms=args.stagger_ms,
                age_weight=args.age_weight,
                static_active_cap=args.static_active_cap,
            )
        summary = summarize(result)
        summaries.append(summary)
        for operation in result.operations:
            operation_rows.append(
                {
                    "policy": operation.policy,
                    "operation_index": operation.operation_index,
                    "phase": operation.phase,
                    "start_ms": operation.start_ms,
                    "end_ms": operation.end_ms,
                    "latency_ms": operation.latency_ms,
                    "memory_mb": operation.memory_mb,
                    "batch_size": operation.batch_size,
                    "oom": operation.oom,
                    "request_ids": ",".join(str(value) for value in operation.request_ids),
                }
            )
        for request in result.requests:
            request_rows.append(
                {
                    "policy": policy,
                    "request_id": request.request_id,
                    "arrival_ms": request.arrival_ms,
                    "slo_deadline_ms": request.slo_deadline_ms,
                    "rejected": request.rejected,
                    "admission_ms": request.admission_ms,
                    "admission_wait_ms": request.admission_wait_ms,
                    "first_start_ms": request.first_start_ms,
                    "completion_ms": request.completion_ms,
                    "latency_ms": (
                        request.completion_ms - request.arrival_ms
                        if request.completion_ms is not None
                        else None
                    ),
                    "service_ms": request.service_ms,
                    "refresh_wait_ms": request.refresh_wait_ms,
                    "refine_wait_ms": request.refine_wait_ms,
                }
            )

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
        summary["peak_memory_ratio_vs_" + baseline["policy"]] = (
            summary["peak_memory_mb"] / baseline["peak_memory_mb"]
            if baseline["peak_memory_mb"]
            else 0.0
        )

    output = {
        "phase_profile": str(cost_model.path),
        "mode": args.mode,
        "num_requests": args.num_requests,
        "max_batch_size": args.max_batch_size,
        "memory_budget_mb": args.memory_budget_mb,
        "static_active_cap": args.static_active_cap,
        "target_rejections": args.target_rejections,
        "arrival_rate_rps": args.arrival_rate_rps,
        "arrival_process": args.arrival_process,
        "burst_size": args.burst_size,
        "burst_interval_ms": args.burst_interval_ms,
        "slo_ms": args.slo_ms,
        "seed": args.seed,
        "stagger_ms": args.stagger_ms,
        "warmup_calls_per_request": cost_model.warmup_calls_per_request,
        "refine_calls_per_request": cost_model.refine_calls_per_request,
        "summaries": summaries,
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    operations_path = prefix.with_name(prefix.name + "_operations.csv")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    summary_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_csv(operations_path, operation_rows)
    _write_csv(requests_path, request_rows)

    print(f"Saved summary: {summary_path}")
    print(
        "policy\tthr\tgoodput\tp95_ms\tp99_ms\tpeak_mem\tOOM\treject\tSLOmiss\tadmit_wait95\trefresh_wait95\tgpu_busy\twarmup_batch\trefine_batch"
    )
    for summary in summaries:
        print(
            "{policy}\t{throughput_rps:.3f}\t{slo_goodput_rps:.3f}\t{p95_latency_ms:.1f}\t"
            "{p99_latency_ms:.1f}\t{peak_memory_mb:.1f}\t{oom_events}\t"
            "{rejected_requests}\t{slo_miss_rate:.3f}\t{p95_admission_wait_ms:.1f}\t"
            "{p95_refresh_wait_ms:.1f}\t{gpu_busy_fraction:.3f}\t"
            "{mean_warmup_batch_size:.2f}\t{mean_refine_batch_size:.2f}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
