"""
Re-analyze multi-tool readiness traces with incremental dispatch and AR probe cost.

Run from repo root after test_multitool_prefetch_signals.py:
  python examples/fastdllm/llada/analyze_multitool_incremental.py \
    --requests_csv artifacts/action_completeness/multitool_llada_s128_requests.csv \
    --calls_csv artifacts/action_completeness/multitool_llada_s128_calls.csv \
    --tool_latencies_ms 100,300,500,1000,2000 \
    --ar_probe_ms 350 \
    --output_prefix artifacts/action_completeness/multitool_llada_s128_incremental

The original latency model compares all-at-once readiness. This script models
incremental dispatch: each call is sent when it becomes stable. It also adds an
optional AR probe cost to approximate verified AR speculation instead of free
optimistic-prefix readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _finish_parallel(
    *,
    generation_ms: float,
    ready_ms: list[float | None],
    tool_latency_ms: float,
    probe_ms: float = 0.0,
) -> float | None:
    if any(value is None for value in ready_ms):
        return None
    tool_finish = max(float(value) + probe_ms + tool_latency_ms for value in ready_ms)
    return max(generation_ms, tool_finish)


def _finish_serial(
    *,
    generation_ms: float,
    ready_ms: list[float | None],
    tool_latency_ms: float,
    probe_ms: float = 0.0,
) -> float | None:
    if any(value is None for value in ready_ms):
        return None
    server_time = 0.0
    for arrival in sorted(float(value) + probe_ms for value in ready_ms):
        server_time = max(server_time, arrival) + tool_latency_ms
    return max(generation_ms, server_time)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _safe_subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize_values(
    rows: list[dict[str, Any]],
    key: str,
    *,
    prefix: str,
    output: dict[str, Any],
) -> None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return
    name_prefix = f"{prefix}_" if prefix else ""
    output[f"{name_prefix}mean_{key}"] = statistics.mean(values)
    output[f"{name_prefix}p50_{key}"] = _percentile(values, 0.50)
    output[f"{name_prefix}p90_{key}"] = _percentile(values, 0.90)
    output[f"{name_prefix}p95_{key}"] = _percentile(values, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_csv", required=True)
    parser.add_argument("--calls_csv", required=True)
    parser.add_argument("--tool_latencies_ms", default="100,300,500,1000,2000")
    parser.add_argument("--ar_probe_ms", type=float, default=350.0)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    tool_latencies = _parse_float_list(args.tool_latencies_ms)
    request_rows = _read_csv(Path(args.requests_csv))
    call_rows = _read_csv(Path(args.calls_csv))
    calls_by_request: dict[str, list[dict[str, str]]] = {}
    for row in call_rows:
        calls_by_request.setdefault(row["request_index"], []).append(row)

    latency_rows: list[dict[str, Any]] = []
    per_request_rows: list[dict[str, Any]] = []
    for request in request_rows:
        request_index = request["request_index"]
        generation_ms = float(request["generation_ms"])
        calls = sorted(
            calls_by_request.get(request_index, []),
            key=lambda row: int(row["call_index"]),
        )
        dllm_ready_ms = [
            (
                generation_ms * float(row["dllm_stable_fraction"])
                if _maybe_float(row.get("dllm_stable_fraction")) is not None
                else None
            )
            for row in calls
        ]
        ar_ready_ms = [
            (
                generation_ms * float(row["ar_stable_fraction"])
                if _maybe_float(row.get("ar_stable_fraction")) is not None
                else None
            )
            for row in calls
        ]
        per_request_rows.append(
            {
                "request_index": request_index,
                "generation_ms": generation_ms,
                "num_calls": len(calls),
                "dllm_first_call_ms": min(float(v) for v in dllm_ready_ms if v is not None),
                "dllm_last_call_ms": max(float(v) for v in dllm_ready_ms if v is not None),
                "ar_first_call_ms": min(float(v) for v in ar_ready_ms if v is not None),
                "ar_last_call_ms": max(float(v) for v in ar_ready_ms if v is not None),
                "dllm_call_ready_spread_ms": max(float(v) for v in dllm_ready_ms if v is not None)
                - min(float(v) for v in dllm_ready_ms if v is not None),
                "ar_call_ready_spread_ms": max(float(v) for v in ar_ready_ms if v is not None)
                - min(float(v) for v in ar_ready_ms if v is not None),
            }
        )
        for latency_ms in tool_latencies:
            no_spec_parallel = generation_ms + latency_ms
            no_spec_serial = generation_ms + latency_ms * len(calls)
            dllm_parallel = _finish_parallel(
                generation_ms=generation_ms,
                ready_ms=dllm_ready_ms,
                tool_latency_ms=latency_ms,
            )
            dllm_serial = _finish_serial(
                generation_ms=generation_ms,
                ready_ms=dllm_ready_ms,
                tool_latency_ms=latency_ms,
            )
            ar_parallel_free = _finish_parallel(
                generation_ms=generation_ms,
                ready_ms=ar_ready_ms,
                tool_latency_ms=latency_ms,
            )
            ar_serial_free = _finish_serial(
                generation_ms=generation_ms,
                ready_ms=ar_ready_ms,
                tool_latency_ms=latency_ms,
            )
            ar_parallel_probe = _finish_parallel(
                generation_ms=generation_ms,
                ready_ms=ar_ready_ms,
                tool_latency_ms=latency_ms,
                probe_ms=args.ar_probe_ms,
            )
            ar_serial_probe = _finish_serial(
                generation_ms=generation_ms,
                ready_ms=ar_ready_ms,
                tool_latency_ms=latency_ms,
                probe_ms=args.ar_probe_ms,
            )
            latency_rows.append(
                {
                    "request_index": request_index,
                    "tool_latency_ms": latency_ms,
                    "ar_probe_ms": args.ar_probe_ms,
                    "num_calls": len(calls),
                    "generation_ms": generation_ms,
                    "no_spec_parallel_ms": no_spec_parallel,
                    "no_spec_serial_ms": no_spec_serial,
                    "dllm_incremental_parallel_ms": dllm_parallel,
                    "dllm_incremental_serial_ms": dllm_serial,
                    "ar_optimistic_parallel_ms": ar_parallel_free,
                    "ar_optimistic_serial_ms": ar_serial_free,
                    "ar_verified_parallel_ms": ar_parallel_probe,
                    "ar_verified_serial_ms": ar_serial_probe,
                    "dllm_vs_ar_optimistic_parallel_speedup": _safe_ratio(
                        ar_parallel_free,
                        dllm_parallel,
                    ),
                    "dllm_vs_ar_optimistic_serial_speedup": _safe_ratio(
                        ar_serial_free,
                        dllm_serial,
                    ),
                    "dllm_vs_ar_verified_parallel_speedup": _safe_ratio(
                        ar_parallel_probe,
                        dllm_parallel,
                    ),
                    "dllm_vs_ar_verified_serial_speedup": _safe_ratio(
                        ar_serial_probe,
                        dllm_serial,
                    ),
                    "dllm_vs_no_spec_parallel_speedup": _safe_ratio(
                        no_spec_parallel,
                        dllm_parallel,
                    ),
                    "dllm_vs_no_spec_serial_speedup": _safe_ratio(
                        no_spec_serial,
                        dllm_serial,
                    ),
                    "dllm_saved_vs_ar_verified_parallel_ms": _safe_subtract(
                        ar_parallel_probe,
                        dllm_parallel,
                    ),
                    "dllm_saved_vs_ar_verified_serial_ms": _safe_subtract(
                        ar_serial_probe,
                        dllm_serial,
                    ),
                }
            )

    aggregate: dict[str, Any] = {
        "num_requests": len(request_rows),
        "num_calls": len(call_rows),
        "tool_latencies_ms": tool_latencies,
        "ar_probe_ms": args.ar_probe_ms,
    }
    aggregate["dllm_false_start_rate"] = (
        sum(
            1
            for row in call_rows
            if float(row.get("dllm_false_starts") or 0) > 0
        )
        / max(len(call_rows), 1)
    )
    aggregate["ar_false_start_rate"] = (
        sum(
            1
            for row in call_rows
            if float(row.get("ar_false_starts") or 0) > 0
        )
        / max(len(call_rows), 1)
    )
    for key in [
        "dllm_call_ready_spread_ms",
        "ar_call_ready_spread_ms",
        "dllm_first_call_ms",
        "dllm_last_call_ms",
        "ar_first_call_ms",
        "ar_last_call_ms",
    ]:
        _summarize_values(per_request_rows, key, prefix="", output=aggregate)
    for latency_ms in tool_latencies:
        rows = [
            row
            for row in latency_rows
            if float(row["tool_latency_ms"]) == float(latency_ms)
        ]
        for key in [
            "dllm_vs_ar_optimistic_parallel_speedup",
            "dllm_vs_ar_optimistic_serial_speedup",
            "dllm_vs_ar_verified_parallel_speedup",
            "dllm_vs_ar_verified_serial_speedup",
            "dllm_vs_no_spec_parallel_speedup",
            "dllm_vs_no_spec_serial_speedup",
            "dllm_saved_vs_ar_verified_parallel_ms",
            "dllm_saved_vs_ar_verified_serial_ms",
        ]:
            _summarize_values(
                rows,
                key,
                prefix=f"tool{latency_ms:g}",
                output=aggregate,
            )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    latency_path = prefix.with_name(prefix.name + "_latency_model.csv")
    request_path = prefix.with_name(prefix.name + "_requests.csv")
    summary_path.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "requests": per_request_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(latency_path, latency_rows)
    _write_csv(request_path, per_request_rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved latency model CSV: {latency_path}")
    print(f"Saved request CSV: {request_path}")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
