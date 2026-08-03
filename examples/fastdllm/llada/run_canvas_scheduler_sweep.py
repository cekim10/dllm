"""
Run a small canvas-coupling scheduler sweep and aggregate the summaries.

Run from repo root:
  python examples/fastdllm/llada/run_canvas_scheduler_sweep.py \
    --latency_table artifacts/elastic_canvas/forward_bench_sweep.json \
    --workloads rare,mix75,mix90 \
    --arrival_rates 0.1,0.5,2,8 \
    --slo_policy by_canvas \
    --slo_scale 10 \
    --output_dir artifacts/elastic_canvas/scheduler_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _parse_csv(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def _safe_name(value: str) -> str:
    return value.replace(".", "p").replace(":", "_").replace(",", "_")


parser = argparse.ArgumentParser()
parser.add_argument("--latency_table", default="artifacts/elastic_canvas/forward_bench_sweep.json")
parser.add_argument("--output_dir", default="artifacts/elastic_canvas/scheduler_sweep")
parser.add_argument("--workloads", default="rare,mix75,mix90,highvar")
parser.add_argument("--arrival_rates", default="0.1,0.5,2,8")
parser.add_argument("--num_requests", type=int, default=300)
parser.add_argument("--max_batch_size", type=int, default=16)
parser.add_argument("--refinement_steps", type=int, default=115)
parser.add_argument("--slo_policy", default="by_canvas")
parser.add_argument("--slo_scale", type=float, default=10.0)
parser.add_argument("--arrival_process", default="poisson")
parser.add_argument("--burst_size", type=int, default=16)
parser.add_argument("--burst_interval_ms", type=float, default=5000.0)
parser.add_argument(
    "--policies",
    default="arrival_dense,exact_bucket,exact_bucket_wait,exact_bucket_bounded,split_oldest,canvas_aware",
)
parser.add_argument("--min_bucket_size", type=int, default=4)
parser.add_argument("--target_bucket_size", type=int, default=8)
parser.add_argument("--max_bucket_wait_ms", type=float, default=2000.0)
parser.add_argument("--deadline_safety_margin_ms", type=float, default=250.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
simulator = Path(__file__).with_name("simulate_canvas_coupling_scheduler.py")

summary_paths = []
for workload in _parse_csv(args.workloads):
    for arrival_rate in _parse_csv(args.arrival_rates):
        prefix = output_dir / f"{_safe_name(workload)}_r{_safe_name(arrival_rate)}"
        cmd = [
            sys.executable,
            str(simulator),
            "--latency_table",
            args.latency_table,
            "--workload",
            workload,
            "--num_requests",
            str(args.num_requests),
            "--arrival_rate_rps",
            arrival_rate,
            "--arrival_process",
            args.arrival_process,
            "--burst_size",
            str(args.burst_size),
            "--burst_interval_ms",
            str(args.burst_interval_ms),
            "--max_batch_size",
            str(args.max_batch_size),
            "--refinement_steps",
            str(args.refinement_steps),
            "--slo_policy",
            args.slo_policy,
            "--slo_scale",
            str(args.slo_scale),
            "--policies",
            args.policies,
            "--min_bucket_size",
            str(args.min_bucket_size),
            "--target_bucket_size",
            str(args.target_bucket_size),
            "--max_bucket_wait_ms",
            str(args.max_bucket_wait_ms),
            "--deadline_safety_margin_ms",
            str(args.deadline_safety_margin_ms),
            "--seed",
            str(args.seed),
            "--output_prefix",
            str(prefix),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        summary_paths.append(prefix.with_name(prefix.name + "_summary.json"))

rows = []
for path in summary_paths:
    data = json.loads(path.read_text())
    baseline_policy = data["summaries"][0]["policy"]
    for summary in data["summaries"]:
        rows.append(
            {
                "file": path.name,
                "workload": data["workload"],
                "arrival_rate_rps": data["arrival_rate_rps"],
                "arrival_process": data["arrival_process"],
                "slo_policy": data["slo_policy"],
                "policy": summary["policy"],
                "throughput_rps": summary["throughput_rps"],
                "throughput_speedup": summary.get(
                    "throughput_speedup_vs_" + baseline_policy, 1.0
                ),
                "p95_latency_ms": summary["p95_latency_ms"],
                "p95_latency_ratio": summary.get(
                    "p95_latency_ratio_vs_" + baseline_policy, 1.0
                ),
                "p95_first_wait_ms": summary["p95_first_wait_ms"],
                "slo_miss_rate": summary["slo_miss_rate"],
                "gpu_time_ratio": summary.get(
                    "gpu_time_ratio_vs_" + baseline_policy, 1.0
                ),
                "token_coupling_waste_ratio": summary[
                    "token_coupling_waste_ratio"
                ],
                "attention_coupling_waste_ratio": summary[
                    "attention_coupling_waste_ratio"
                ],
                "avg_batch_size": summary["avg_batch_size"],
            }
        )

combined_json = output_dir / "combined_summary.json"
combined_csv = output_dir / "combined_summary.csv"
combined_json.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
with combined_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved combined JSON:", combined_json)
print("Saved combined CSV:", combined_csv)
