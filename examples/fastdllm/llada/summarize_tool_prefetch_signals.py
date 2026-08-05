"""
Summarize go/no-go signals from test_tool_prefetch_signals.py.

Run from repo root:
  python examples/fastdllm/llada/summarize_tool_prefetch_signals.py \
    --summary_path artifacts/tool_prefetch/llada_prefix_s128_summary.json
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
    requests = data.get("requests", [])

    print("Core metrics")
    for key in [
        "num_requests",
        "num_final_ready",
        "final_ready_rate",
        "dllm_ready_rate",
        "ar_ready_rate",
        "dllm_beats_ar_rate",
        "mean_dllm_ready_fraction",
        "mean_ar_ready_fraction",
        "dllm_stable_rate",
        "ar_stable_rate",
        "mean_dllm_stable_fraction",
        "mean_ar_stable_fraction",
        "mean_dllm_false_starts",
        "mean_ar_false_starts",
        "mean_dllm_lead_fraction",
        "p50_dllm_lead_fraction",
    ]:
        print(f"{key}\t{_fmt(aggregate.get(key))}")

    print("\nLatency overlap")
    for key in sorted(aggregate):
        if key.startswith("mean_dllm_saving_tool"):
            latency = key.removeprefix("mean_dllm_saving_tool").removesuffix("ms")
            ar_key = f"mean_ar_saving_tool{latency}ms"
            dllm_value = aggregate.get(key)
            ar_value = aggregate.get(ar_key)
            gap = (
                float(dllm_value) - float(ar_value)
                if isinstance(dllm_value, (int, float)) and isinstance(ar_value, (int, float))
                else None
            )
            print(
                f"tool={latency}ms\tdllm={_fmt(dllm_value)}\t"
                f"ar={_fmt(ar_value)}\tdelta={_fmt(gap)}"
            )

    valid = [row for row in requests if row.get("final_ready")]
    early = [
        row for row in valid
        if row.get("dllm_ready_fraction") is not None
        and float(row["dllm_ready_fraction"]) <= 0.5
    ]
    stable_early = [
        row for row in valid
        if row.get("dllm_stable_fraction") is not None
        and float(row["dllm_stable_fraction"]) <= 0.7
    ]
    beats = [row for row in valid if row.get("dllm_beats_ar")]
    print("\nGo/no-go")
    if not valid:
        print("NO-GO: final tool-call accuracy is zero; prompt/extractor must be fixed first.")
        return
    early_rate = len(early) / len(valid)
    beats_rate = len(beats) / len(valid)
    saving_delta_300 = aggregate.get("mean_dllm_saving_tool300ms", 0.0) - aggregate.get(
        "mean_ar_saving_tool300ms",
        0.0,
    )
    stable_early_rate = len(stable_early) / len(valid)
    if (
        early_rate >= 0.70
        and stable_early_rate >= 0.60
        and beats_rate >= 0.50
        and saving_delta_300 >= 0.10
    ):
        print(
            "STRONG GO: dLLM intermediate states expose usable tool intent early "
            "and beat optimistic AR prefixes."
        )
    elif (
        early_rate >= 0.50
        and stable_early_rate >= 0.40
        and beats_rate >= 0.30
        and saving_delta_300 >= 0.05
    ):
        print(
            "CONDITIONAL GO: signal exists, but needs better extractor/runtime "
            "and AR baseline before committing."
        )
    else:
        print(
            "NO-GO: dLLM does not provide enough early, AR-distinct tool signal "
            "under this workload."
        )
    print(f"early_ready_rate_le_0p5\t{early_rate:.3f}")
    print(f"stable_ready_rate_le_0p7\t{stable_early_rate:.3f}")
    print(f"dllm_beats_ar_valid_rate\t{beats_rate:.3f}")
    print(f"saving_delta_300ms\t{saving_delta_300:.3f}")


if __name__ == "__main__":
    main()
