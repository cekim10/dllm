"""
Sweep AR probe-cost assumptions for multi-tool incremental dispatch.

Run from repo root after test_multitool_prefetch_signals.py:
  python3 examples/fastdllm/llada/sweep_multitool_probe_cost.py \
    --requests_csv artifacts/action_completeness/multitool_llada_s128_requests.csv \
    --calls_csv artifacts/action_completeness/multitool_llada_s128_calls.csv \
    --probe_costs_ms 0,50,100,200,350,500 \
    --output_prefix artifacts/action_completeness/multitool_llada_s128_probe_sweep

This is a lightweight wrapper around analyze_multitool_incremental.py. It
recomputes the same trace under several AR verified-probe cost assumptions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_csv", required=True)
    parser.add_argument("--calls_csv", required=True)
    parser.add_argument("--tool_latencies_ms", default="100,300,500,1000,2000")
    parser.add_argument("--probe_costs_ms", default="0,50,100,200,350,500")
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    script_path = Path(__file__).with_name("analyze_multitool_incremental.py")
    combined = []
    for probe_ms in _parse_float_list(args.probe_costs_ms):
        child_prefix = Path(f"{args.output_prefix}_probe{probe_ms:g}ms")
        cmd = [
            sys.executable,
            str(script_path),
            "--requests_csv",
            args.requests_csv,
            "--calls_csv",
            args.calls_csv,
            "--tool_latencies_ms",
            args.tool_latencies_ms,
            "--ar_probe_ms",
            str(probe_ms),
            "--output_prefix",
            str(child_prefix),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        summary_path = child_prefix.with_name(child_prefix.name + "_summary.json")
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregate = data["aggregate"]
        aggregate["summary_path"] = str(summary_path)
        combined.append(aggregate)

    output_path = Path(args.output_prefix).with_name(Path(args.output_prefix).name + "_combined.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved combined summary: {output_path}")


if __name__ == "__main__":
    main()
