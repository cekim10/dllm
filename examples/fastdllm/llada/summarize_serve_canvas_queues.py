"""
Summarize real-forward canvas-queue serving prototype outputs.

Run from repo root:
  python examples/fastdllm/llada/summarize_serve_canvas_queues.py \
    artifacts/elastic_canvas/serve_canvas_queues_*_summary.json
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
        "Usage: python examples/fastdllm/llada/summarize_serve_canvas_queues.py "
        "artifacts/elastic_canvas/serve_canvas_queues_*_summary.json"
    )

columns = [
    "file",
    "workload",
    "arrival",
    "steps",
    "policy",
    "throughput",
    "thr_spd",
    "p95_ms",
    "p95_ratio",
    "wait95",
    "service95",
    "inter95",
    "gpu_busy",
    "gpu_ratio",
    "gpu_ms",
    "batch",
    "partial",
    "tok_waste",
    "attn_waste",
    "slo_miss",
]
print("\t".join(columns))

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "summaries" not in data:
        continue
    baseline = data["summaries"][0]["policy"]
    for summary in data["summaries"]:
        row = {
            "file": path.name,
            "workload": data["workload"],
            "arrival": data["arrival_rate_rps"],
            "steps": data["refinement_steps"],
            "policy": summary["policy"],
            "throughput": summary["throughput_rps"],
            "thr_spd": summary.get("throughput_speedup_vs_" + baseline, 1.0),
            "p95_ms": summary["p95_latency_ms"],
            "p95_ratio": summary.get("p95_latency_ratio_vs_" + baseline, 1.0),
            "wait95": summary["p95_first_wait_ms"],
            "service95": summary["p95_service_ms"],
            "inter95": summary["p95_inter_iteration_delay_ms"],
            "gpu_busy": summary["gpu_busy_fraction"],
            "gpu_ratio": summary.get("gpu_time_ratio_vs_" + baseline, 1.0),
            "gpu_ms": summary["total_gpu_ms"],
            "batch": summary["mean_batch_size"],
            "partial": summary["partial_dispatch_ratio"],
            "tok_waste": summary["token_coupling_waste_ratio"],
            "attn_waste": summary["attention_coupling_waste_ratio"],
            "slo_miss": summary["slo_miss_rate"],
        }
        print("\t".join(_fmt(row[column]) for column in columns))
