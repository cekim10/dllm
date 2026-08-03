"""
Summarize canvas-coupling scheduler simulation outputs.

Run from repo root:
  python examples/fastdllm/llada/summarize_canvas_scheduler.py \
    artifacts/elastic_canvas/canvas_scheduler_*_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python examples/fastdllm/llada/summarize_canvas_scheduler.py "
        "artifacts/elastic_canvas/canvas_scheduler_*_summary.json"
    )

columns = [
    "file",
    "workload",
    "arrival",
    "process",
    "slo",
    "policy",
    "throughput",
    "thr_spd",
    "p95_ms",
    "p95_ratio",
    "p95_wait",
    "slo_miss",
    "gpu_ratio",
    "tok_waste",
    "attn_waste",
    "avg_batch",
]
print("\t".join(columns))

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    data = json.loads(path.read_text())
    summaries = data["summaries"]
    baseline_policy = summaries[0]["policy"]
    for summary in summaries:
        row = {
            "file": path.name,
            "workload": data["workload"],
            "arrival": data["arrival_rate_rps"],
            "process": data.get("arrival_process", "poisson"),
            "slo": data.get("slo_policy", "none"),
            "policy": summary["policy"],
            "throughput": summary["throughput_rps"],
            "thr_spd": summary.get(
                "throughput_speedup_vs_" + baseline_policy, 1.0
            ),
            "p95_ms": summary["p95_latency_ms"],
            "p95_ratio": summary.get("p95_latency_ratio_vs_" + baseline_policy, 1.0),
            "p95_wait": summary.get("p95_first_wait_ms", 0.0),
            "slo_miss": summary.get("slo_miss_rate", 0.0),
            "gpu_ratio": summary.get("gpu_time_ratio_vs_" + baseline_policy, 1.0),
            "tok_waste": summary["token_coupling_waste_ratio"],
            "attn_waste": summary["attention_coupling_waste_ratio"],
            "avg_batch": summary["avg_batch_size"],
        }
        print("\t".join(_fmt(row[column]) for column in columns))
