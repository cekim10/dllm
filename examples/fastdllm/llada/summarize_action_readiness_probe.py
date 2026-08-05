"""
Summarize AR auxiliary probe vs native dLLM action readiness.

Run from repo root:
  python examples/fastdllm/llada/summarize_action_readiness_probe.py \
    --summary_path artifacts/action_completeness/probe_vs_native_core_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    aggregate = data["aggregate"]

    print("Core metrics")
    for key in [
        "num_requests",
        "native_rows",
        "probe_ready_rate",
        "mean_probe_latency_ms",
        "p95_probe_latency_ms",
        "mean_probe_output_tokens",
        "mean_native_generation_ms",
        "mean_native_ready_ms",
        "mean_probe_over_native_generation",
        "probe_beats_native_ready_rate",
    ]:
        print(f"{key}\t{_fmt(aggregate.get(key))}")

    print("\nLatency saving")
    for key in sorted(aggregate):
        if not key.startswith("mean_native_saving_tool"):
            continue
        latency = key.removeprefix("mean_native_saving_tool").removesuffix("ms")
        probe_key = f"mean_probe_saving_tool{latency}ms"
        native_value = aggregate.get(key)
        probe_value = aggregate.get(probe_key)
        delta = (
            float(native_value) - float(probe_value)
            if isinstance(native_value, (int, float)) and isinstance(probe_value, (int, float))
            else None
        )
        print(
            f"tool={latency}ms\tnative={_fmt(native_value)}\t"
            f"probe={_fmt(probe_value)}\tnative_minus_probe={_fmt(delta)}"
        )

    probe_ready = aggregate.get("probe_ready_rate", 0.0) or 0.0
    native_delta = aggregate.get("mean_native_saving_tool300ms", 0.0) - aggregate.get(
        "mean_probe_saving_tool300ms",
        0.0,
    )
    extra_work = aggregate.get("mean_probe_over_native_generation", 1.0) or 1.0
    beats_rate = aggregate.get("probe_beats_native_ready_rate", 0.0) or 0.0

    print("\nInterpretation")
    if probe_ready < 0.70:
        print("Probe baseline is not reliable enough; compare only after prompt/model calibration.")
    elif native_delta >= 0.05 and extra_work >= 0.10:
        print(
            "Native readiness is favorable: it saves more latency while avoiding "
            "the auxiliary generation branch."
        )
    elif beats_rate >= 0.70:
        print(
            "Probe is often earlier than native readiness; the dLLM advantage must "
            "come from lower extra work/interference, not readiness time."
        )
    else:
        print(
            "Mixed result: native readiness and auxiliary probing need a fuller "
            "concurrency/interference evaluation."
        )


if __name__ == "__main__":
    main()
