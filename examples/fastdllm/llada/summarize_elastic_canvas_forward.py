"""
Summarize elastic-canvas forward benchmark results.

Run from repo root:
  python examples/fastdllm/llada/summarize_elastic_canvas_forward.py \
    artifacts/elastic_canvas/forward_bench_sweep.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python examples/fastdllm/llada/summarize_elastic_canvas_forward.py "
        "artifacts/elastic_canvas/forward_bench_sweep.json"
    )

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

columns = [
    "batch",
    "pattern",
    "fixed_ms",
    "elastic_ms",
    "bucket_ms",
    "elastic_spd",
    "bucket_spd",
    "bucket_vs_elastic",
    "bucket_calls",
    "fixed_tokens",
    "dense_tokens",
    "bucket_tokens",
    "fixed_mem",
    "bucket_mem",
]
print("\t".join(columns))
for result in sorted(
    data["results"], key=lambda item: (int(item["batch_size"]), item["pattern"])
):
    dense_fixed = result["dense_fixed"]
    elastic_dense = result["elastic_dense"]
    bucketed = result["bucketed_shape_decoupled"]
    row = {
        "batch": result["batch_size"],
        "pattern": result["pattern"],
        "fixed_ms": dense_fixed["avg_ms"],
        "elastic_ms": elastic_dense["avg_ms"],
        "bucket_ms": bucketed["avg_ms"],
        "elastic_spd": result["elastic_dense_speedup_vs_fixed"],
        "bucket_spd": result["bucketed_speedup_vs_fixed"],
        "bucket_vs_elastic": result["bucketed_speedup_vs_elastic_dense"],
        "bucket_calls": bucketed["num_calls"],
        "fixed_tokens": result["fixed_physical_tokens"],
        "dense_tokens": result["elastic_dense_physical_tokens"],
        "bucket_tokens": result["bucketed_physical_tokens"],
        "fixed_mem": dense_fixed["peak_memory_mb"],
        "bucket_mem": bucketed["peak_memory_mb"],
    }
    print("\t".join(_fmt(row[column]) for column in columns))
