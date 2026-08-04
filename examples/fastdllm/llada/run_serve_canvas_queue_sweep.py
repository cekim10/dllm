"""
Run real-forward canvas-queue serving sweeps and aggregate summaries.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/run_serve_canvas_queue_sweep.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --workloads mix50,mix75,mix90,highvar,rare \
    --arrival_rates 0.5,2 \
    --refinement_steps_list 16 \
    --num_requests 64 \
    --output_dir artifacts/elastic_canvas/serve_sweep
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
parser.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
parser.add_argument("--output_dir", default="artifacts/elastic_canvas/serve_sweep")
parser.add_argument("--workloads", default="mix50,mix75,mix90,highvar,rare")
parser.add_argument("--arrival_rates", default="0.5,2")
parser.add_argument("--refinement_steps_list", default="8,16,32")
parser.add_argument("--num_requests", type=int, default=64)
parser.add_argument("--max_batch_size", type=int, default=16)
parser.add_argument(
    "--policies",
    default=(
        "arrival_dense,coarse_canvas_queue,exact_canvas_queue,"
        "exact_canvas_queue_wait,exact_canvas_queue_bounded"
    ),
)
parser.add_argument("--workload_prompt", default="Explain canvas-aware dLLM serving in one sentence.")
parser.add_argument("--slo_policy", default="by_canvas")
parser.add_argument("--slo_scale", type=float, default=10.0)
parser.add_argument("--target_bucket_size", type=int, default=8)
parser.add_argument("--max_bucket_wait_ms", type=float, default=20.0)
parser.add_argument("--coarse_canvas_groups", default="32,64;128,256")
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--rerun_existing",
    action="store_true",
    help="Rerun experiments even when the expected summary JSON already exists.",
)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
runner = Path(__file__).with_name("serve_canvas_queues.py")
summary_paths = []

for workload in _parse_csv(args.workloads):
    for arrival_rate in _parse_csv(args.arrival_rates):
        for steps in _parse_csv(args.refinement_steps_list):
            prefix = (
                output_dir
                / f"{_safe_name(workload)}_r{_safe_name(arrival_rate)}_s{_safe_name(steps)}"
            )
            cmd = [
                sys.executable,
                str(runner),
                "--model_name_or_path",
                args.model_name_or_path,
                "--prompt",
                args.workload_prompt,
                "--workload",
                workload,
                "--num_requests",
                str(args.num_requests),
                "--arrival_rate_rps",
                arrival_rate,
                "--refinement_steps",
                steps,
                "--max_batch_size",
                str(args.max_batch_size),
                "--policies",
                args.policies,
                "--slo_policy",
                args.slo_policy,
                "--slo_scale",
                str(args.slo_scale),
                "--target_bucket_size",
                str(args.target_bucket_size),
                "--max_bucket_wait_ms",
                str(args.max_bucket_wait_ms),
                "--coarse_canvas_groups",
                args.coarse_canvas_groups,
                "--warmup",
                str(args.warmup),
                "--seed",
                str(args.seed),
                "--output_prefix",
                str(prefix),
            ]
            summary_path = prefix.with_name(prefix.name + "_summary.json")
            if summary_path.exists() and not args.rerun_existing:
                print("Skipping existing:", summary_path, flush=True)
            else:
                print("Running:", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
            summary_paths.append(summary_path)

rows = []
for path in summary_paths:
    data = json.loads(path.read_text())
    baseline = data["summaries"][0]["policy"]
    for summary in data["summaries"]:
        rows.append(
            {
                "file": path.name,
                "workload": data["workload"],
                "arrival_rate_rps": data["arrival_rate_rps"],
                "refinement_steps": data["refinement_steps"],
                "policy": summary["policy"],
                "throughput_rps": summary["throughput_rps"],
                "throughput_speedup": summary.get(
                    "throughput_speedup_vs_" + baseline, 1.0
                ),
                "p95_latency_ms": summary["p95_latency_ms"],
                "p95_latency_ratio": summary.get(
                    "p95_latency_ratio_vs_" + baseline, 1.0
                ),
                "p95_first_wait_ms": summary["p95_first_wait_ms"],
                "p95_service_ms": summary["p95_service_ms"],
                "p95_inter_iteration_delay_ms": summary[
                    "p95_inter_iteration_delay_ms"
                ],
                "gpu_busy_fraction": summary["gpu_busy_fraction"],
                "gpu_time_ratio": summary.get(
                    "gpu_time_ratio_vs_" + baseline, 1.0
                ),
                "mean_batch_size": summary["mean_batch_size"],
                "partial_dispatch_ratio": summary["partial_dispatch_ratio"],
                "token_coupling_waste_ratio": summary[
                    "token_coupling_waste_ratio"
                ],
                "attention_coupling_waste_ratio": summary[
                    "attention_coupling_waste_ratio"
                ],
                "slo_miss_rate": summary["slo_miss_rate"],
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
