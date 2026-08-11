"""
Summarize NFE stage-ablation sweeps into stage-local cliff diagnostics.

Run from repo root:
  python examples/fastdllm/llada/summarize_nfe_stage_sweep.py \
    --summary_glob "artifacts/nfe_stage_ablation/multitool_3call_h128_l*_topk_summary.json" \
    --output_prefix artifacts/nfe_stage_ablation/multitool_3call_h128_topk_sweep

The input files should come from run_multitool_nfe_stage_ablation.py with
different --low_steps values and the same --high_steps baseline.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


LOW_STEPS_RE = re.compile(r"_l(\d+)_topk_summary\.json$")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _low_steps_from_path(path: Path, aggregate: dict[str, Any]) -> int:
    if "low_steps" in aggregate:
        return int(aggregate["low_steps"])
    match = LOW_STEPS_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer low_steps from {path}")
    return int(match.group(1))


def _paired_retention(requests_path: Path) -> dict[str, Any]:
    rows = _read_csv(requests_path)
    by_request: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_request[int(row["request_index"])][row["variant"]] = row
    base_success = [
        request_index
        for request_index, variants in by_request.items()
        if variants["all_high"]["final_all_ready"] == "True"
    ]
    result: dict[str, Any] = {"all_high_success_count": len(base_success)}
    for variant in ["low_stage_0", "low_stage_1", "low_stage_2", "all_low"]:
        kept = [
            request_index
            for request_index in base_success
            if by_request[request_index][variant]["final_all_ready"] == "True"
        ]
        lost = sorted(set(base_success) - set(kept))
        gained = [
            request_index
            for request_index, variants in by_request.items()
            if variants["all_high"]["final_all_ready"] != "True"
            and variants[variant]["final_all_ready"] == "True"
        ]
        result[f"{variant}_kept_count"] = len(kept)
        result[f"{variant}_retention_rate"] = (
            len(kept) / len(base_success) if base_success else None
        )
        result[f"{variant}_lost_request_indexes"] = ",".join(str(i) for i in lost)
        result[f"{variant}_gained_request_indexes"] = ",".join(
            str(i) for i in sorted(gained)
        )
    return result


def _request_generation_metrics(requests_path: Path) -> dict[str, Any]:
    rows = _read_csv(requests_path)
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    result: dict[str, Any] = {}
    for variant, variant_rows in by_variant.items():
        values = [float(row["total_generation_ms"]) for row in variant_rows]
        result[f"{variant}_mean_generation_ms_from_requests"] = (
            sum(values) / len(values) if values else None
        )
    baseline = result.get("all_high_mean_generation_ms_from_requests")
    if baseline:
        for variant in by_variant:
            mean_generation = result.get(f"{variant}_mean_generation_ms_from_requests")
            if mean_generation:
                result[f"{variant}_latency_speedup_vs_all_high_from_requests"] = (
                    baseline / mean_generation
                )
    return result


def _stage_transitions(stages_path: Path, high_steps: int, low_steps: int) -> dict[str, Any]:
    rows = _read_csv(stages_path)
    by_stage: dict[tuple[int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["request_index"]), int(row["stage_index"]))
        by_stage[key][int(row["steps"])] = row
    result: dict[str, Any] = {}
    for stage_index in sorted({stage for _, stage in by_stage}):
        counts = {
            "high_true_low_true": 0,
            "high_true_low_false": 0,
            "high_false_low_true": 0,
            "high_false_low_false": 0,
        }
        for (request_index, stage), values in by_stage.items():
            if stage != stage_index:
                continue
            high = values[high_steps]["ready"] == "True"
            low = values[low_steps]["ready"] == "True"
            if high and low:
                counts["high_true_low_true"] += 1
            elif high and not low:
                counts["high_true_low_false"] += 1
            elif not high and low:
                counts["high_false_low_true"] += 1
            else:
                counts["high_false_low_false"] += 1
        prefix = f"stage{stage_index}"
        result.update({f"{prefix}_{key}": value for key, value in counts.items()})
        high_successes = counts["high_true_low_true"] + counts["high_true_low_false"]
        result[f"{prefix}_low_retention_given_high_success"] = (
            counts["high_true_low_true"] / high_successes if high_successes else None
        )
    return result


def _first_step_at_or_above(rows: list[dict[str, Any]], key: str, threshold: float) -> int | None:
    for row in sorted(rows, key=lambda item: int(item["low_steps"])):
        value = row.get(key)
        if value is not None and float(value) >= threshold:
            return int(row["low_steps"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_glob", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--cliff_threshold", type=float, default=0.8)
    args = parser.parse_args()

    summary_paths = [Path(path) for path in sorted(glob.glob(args.summary_glob))]
    if not summary_paths:
        raise ValueError(f"No summary files matched {args.summary_glob!r}")

    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        data = _read_json(summary_path)
        aggregate = data["aggregate"]
        low_steps = _low_steps_from_path(summary_path, aggregate)
        high_steps = int(aggregate["high_steps"])
        prefix = summary_path.with_name(summary_path.name.removesuffix("_summary.json"))
        requests_path = prefix.with_name(prefix.name + "_requests.csv")
        stages_path = prefix.with_name(prefix.name + "_stages.csv")

        row: dict[str, Any] = {
            "file": str(summary_path),
            "high_steps": high_steps,
            "low_steps": low_steps,
            "all_high_final_all_ready_rate": aggregate[
                "all_high_final_all_ready_rate"
            ],
            "all_low_final_all_ready_rate": aggregate["all_low_final_all_ready_rate"],
            "all_low_mean_final_ready_count": aggregate["all_low_mean_final_ready_count"],
            "all_high_mean_generation_ms": aggregate.get("all_high_mean_generation_ms"),
            "all_low_mean_generation_ms": aggregate.get("all_low_mean_generation_ms"),
            "all_low_latency_speedup_vs_all_high": aggregate.get(
                "all_low_latency_speedup_vs_all_high"
            ),
        }
        for stage_index in range(3):
            row[f"low_stage_{stage_index}_final_all_ready_rate"] = aggregate[
                f"low_stage_{stage_index}_final_all_ready_rate"
            ]
            row[f"low_stage_{stage_index}_delta_vs_all_high"] = aggregate[
                f"low_stage_{stage_index}_final_all_ready_delta_vs_all_high"
            ]
            row[f"low_stage_{stage_index}_mean_final_ready_count"] = aggregate[
                f"low_stage_{stage_index}_mean_final_ready_count"
            ]
            row[f"stage{stage_index}_low_ready_rate"] = aggregate[
                f"stage{stage_index}_steps{low_steps}_ready_rate"
            ]
            row[f"stage{stage_index}_low_tool_correct_rate"] = aggregate[
                f"stage{stage_index}_steps{low_steps}_tool_correct_rate"
            ]
            row[f"stage{stage_index}_low_args_correct_rate"] = aggregate[
                f"stage{stage_index}_steps{low_steps}_args_correct_rate"
            ]
        row.update(_paired_retention(requests_path))
        request_generation = _request_generation_metrics(requests_path)
        row.update(request_generation)
        row["all_high_mean_generation_ms"] = (
            row["all_high_mean_generation_ms"]
            or request_generation.get("all_high_mean_generation_ms_from_requests")
        )
        row["all_low_mean_generation_ms"] = (
            row["all_low_mean_generation_ms"]
            or request_generation.get("all_low_mean_generation_ms_from_requests")
        )
        row["all_low_latency_speedup_vs_all_high"] = (
            row["all_low_latency_speedup_vs_all_high"]
            or request_generation.get("all_low_latency_speedup_vs_all_high_from_requests")
        )
        row.update(_stage_transitions(stages_path, high_steps, low_steps))
        rows.append(row)
    rows.sort(key=lambda row: int(row["low_steps"]))

    cliff: dict[str, Any] = {
        "cliff_threshold": args.cliff_threshold,
        "num_sweeps": len(rows),
    }
    for stage_index in range(3):
        cliff[f"stage{stage_index}_first_low_steps_ready_ge_threshold"] = (
            _first_step_at_or_above(
                rows,
                f"stage{stage_index}_low_ready_rate",
                args.cliff_threshold,
            )
        )
        cliff[f"stage{stage_index}_first_low_steps_retention_ge_threshold"] = (
            _first_step_at_or_above(
                rows,
                f"stage{stage_index}_low_retention_given_high_success",
                args.cliff_threshold,
            )
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_name(prefix.name + "_combined.csv")
    json_path = prefix.with_name(prefix.name + "_combined.json")
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "cliff": cliff,
                "rows": rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved combined CSV: {csv_path}")
    print(f"Saved combined JSON: {json_path}")
    print(json.dumps({"cliff": cliff}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
