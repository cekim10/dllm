"""
Utilities for block-usefulness profiling with Fast-dLLM LLaDA.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  python -u examples/fastdllm/llada/profile_block_usefulness.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _safe_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    x_norm = torch.linalg.vector_norm(x)
    y_norm = torch.linalg.vector_norm(y)
    if x_norm.item() == 0.0 or y_norm.item() == 0.0:
        return 1.0 if x_norm.item() == y_norm.item() else 0.0
    return float(F.cosine_similarity(x, y, dim=1).item())


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32)


@dataclass
class BlockUsefulnessProfiler:
    records: list[dict[str, Any]] = field(default_factory=list)
    _previous_outputs: dict[tuple[int, int, int], torch.Tensor] = field(
        default_factory=dict
    )

    def __call__(
        self,
        observer_context: dict[str, Any],
        hidden_before: torch.Tensor,
        hidden_after: torch.Tensor,
    ) -> None:
        layer_index = int(observer_context["layer_index"])
        block_index = int(observer_context["block_index"])
        step_index = int(observer_context["step_index"])
        model_call_index = int(observer_context["model_call_index"])
        block_ranges = observer_context["block_ranges"]
        mask_counts = observer_context.get("mask_counts", ())

        hidden_before_cpu = _to_float_tensor(hidden_before)
        hidden_after_cpu = _to_float_tensor(hidden_after)

        for request_index, (start, end) in enumerate(block_ranges):
            start = int(start)
            end = int(end)
            if end <= start:
                continue

            block_before = hidden_before_cpu[request_index, start:end]
            block_after = hidden_after_cpu[request_index, start:end]
            flat_before = block_before.reshape(-1)
            flat_after = block_after.reshape(-1)
            block_delta = flat_after - flat_before

            prev_key = (request_index, block_index, layer_index)
            previous_output = self._previous_outputs.get(prev_key)
            inter_step_delta_norm = None
            inter_step_cosine = None
            if previous_output is not None and previous_output.shape == flat_after.shape:
                inter_step_delta_norm = float(
                    torch.linalg.vector_norm(flat_after - previous_output).item()
                )
                inter_step_cosine = _safe_cosine_similarity(previous_output, flat_after)
            self._previous_outputs[prev_key] = flat_after.clone()

            self.records.append(
                {
                    "request_index": request_index,
                    "cache_mode": observer_context["cache_mode"],
                    "phase": observer_context["phase"],
                    "block_index": block_index,
                    "layer_index": layer_index,
                    "step_index": step_index,
                    "model_call_index": model_call_index,
                    "block_start": start,
                    "block_end": end,
                    "block_token_count": end - start,
                    "mask_count": int(mask_counts[request_index])
                    if request_index < len(mask_counts)
                    else None,
                    "layer_latency_ms": float(observer_context["layer_latency_ms"]),
                    "block_input_norm": float(
                        torch.linalg.vector_norm(flat_before).item()
                    ),
                    "block_output_norm": float(
                        torch.linalg.vector_norm(flat_after).item()
                    ),
                    "block_delta_norm": float(
                        torch.linalg.vector_norm(block_delta).item()
                    ),
                    "block_delta_cosine": _safe_cosine_similarity(
                        flat_before, flat_after
                    ),
                    "inter_step_output_delta_norm": inter_step_delta_norm,
                    "inter_step_output_cosine": inter_step_cosine,
                }
            )

    def build_summary(self) -> dict[str, Any]:
        grouped: dict[tuple[str, int, int], dict[str, float]] = {}
        counts: dict[tuple[str, int, int], int] = {}
        inter_step_norm_counts: dict[tuple[str, int, int], int] = {}
        inter_step_cos_counts: dict[tuple[str, int, int], int] = {}

        for record in self.records:
            key = (
                str(record["cache_mode"]),
                int(record["block_index"]),
                int(record["layer_index"]),
            )
            bucket = grouped.setdefault(
                key,
                {
                    "avg_latency_ms": 0.0,
                    "avg_block_delta_norm": 0.0,
                    "avg_block_delta_cosine": 0.0,
                    "avg_inter_step_output_delta_norm": 0.0,
                    "avg_inter_step_output_cosine": 0.0,
                },
            )
            counts[key] = counts.get(key, 0) + 1
            bucket["avg_latency_ms"] += float(record["layer_latency_ms"])
            bucket["avg_block_delta_norm"] += float(record["block_delta_norm"])
            bucket["avg_block_delta_cosine"] += float(record["block_delta_cosine"])
            if record["inter_step_output_delta_norm"] is not None:
                bucket["avg_inter_step_output_delta_norm"] += float(
                    record["inter_step_output_delta_norm"]
                )
                inter_step_norm_counts[key] = inter_step_norm_counts.get(key, 0) + 1
            if record["inter_step_output_cosine"] is not None:
                bucket["avg_inter_step_output_cosine"] += float(
                    record["inter_step_output_cosine"]
                )
                inter_step_cos_counts[key] = inter_step_cos_counts.get(key, 0) + 1

        summary_rows = []
        for key, values in grouped.items():
            count = counts[key]
            inter_step_norm_count = inter_step_norm_counts.get(key, 0)
            inter_step_cos_count = inter_step_cos_counts.get(key, 0)
            summary_rows.append(
                {
                    "cache_mode": key[0],
                    "block_index": key[1],
                    "layer_index": key[2],
                    "num_records": count,
                    "avg_latency_ms": values["avg_latency_ms"] / count,
                    "avg_block_delta_norm": values["avg_block_delta_norm"] / count,
                    "avg_block_delta_cosine": values["avg_block_delta_cosine"] / count,
                    "avg_inter_step_output_delta_norm": (
                        values["avg_inter_step_output_delta_norm"]
                        / inter_step_norm_count
                        if inter_step_norm_count > 0
                        else None
                    ),
                    "avg_inter_step_output_cosine": (
                        values["avg_inter_step_output_cosine"] / inter_step_cos_count
                        if inter_step_cos_count > 0
                        else None
                    ),
                }
            )
        summary_rows.sort(
            key=lambda row: (
                row["cache_mode"],
                row["block_index"],
                row["layer_index"],
            )
        )
        return {"num_records": len(self.records), "rows": summary_rows}

    def save(self, output_prefix: str | Path) -> dict[str, str]:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        records_path = prefix.with_suffix(".jsonl")
        with records_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        csv_path = prefix.with_suffix(".csv")
        if self.records:
            fieldnames = list(self.records[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)

        summary = self.build_summary()
        summary_path = prefix.with_name(prefix.name + "_summary.json")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

        return {
            "records_jsonl": str(records_path),
            "records_csv": str(csv_path),
            "summary_json": str(summary_path),
        }
