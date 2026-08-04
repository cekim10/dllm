"""
Profile refresh/reuse phase heterogeneity for cached Fast-dLLM LLaDA inference.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/profile_refresh_storm.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --batch_sizes 1,2,4,8 \
    --steps 128 \
    --max_new_tokens 128 \
    --block_size 32 \
    --use_cache prefix \
    --output_prefix artifacts/refresh_storm/llada_prefix
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import transformers

import dllm


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    if any(item <= 0 for item in values):
        raise ValueError(f"Batch sizes must be positive: {values}")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_messages(
    input_path: str | None,
    prompt: str,
    limit: int,
) -> list[list[dict[str, str]]]:
    if input_path is None:
        return [[{"role": "user", "content": prompt}]]

    path = Path(input_path)
    messages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                record = json.loads(line)
                if "messages" in record:
                    messages.append(record["messages"])
                else:
                    text = record.get("prompt") or record.get("text") or record.get("input")
                    if text is None:
                        raise ValueError(f"Unsupported JSONL record: {record}")
                    messages.append([{"role": "user", "content": text}])
            else:
                messages.append([{"role": "user", "content": line}])
            if len(messages) >= limit:
                break
    if not messages:
        raise ValueError(f"No prompts found in {path}")
    return messages


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass
class ModelCallProfiler:
    batch_size: int
    records: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, context: dict[str, Any]) -> None:
        record = _jsonable(context)
        record["experiment_batch_size"] = int(self.batch_size)
        record["record_index"] = len(self.records)
        record["memory_peak_delta_mb"] = (
            float(record.get("memory_peak_delta_bytes", 0)) / (1024.0 * 1024.0)
        )
        record["memory_peak_mb"] = (
            float(record.get("memory_peak_bytes", 0)) / (1024.0 * 1024.0)
        )
        self.records.append(record)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("experiment_batch_size"),
            record.get("cache_mode"),
            record.get("phase"),
            record.get("model_query_length"),
            record.get("has_past_key_values"),
        )
        groups[key].append(record)

    rows = []
    for key, bucket in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        latencies = [float(row["model_call_latency_ms"]) for row in bucket]
        memory_deltas = [float(row["memory_peak_delta_mb"]) for row in bucket]
        rows.append(
            {
                "batch_size": key[0],
                "cache_mode": key[1],
                "phase": key[2],
                "model_query_length": key[3],
                "has_past_key_values": key[4],
                "num_calls": len(bucket),
                "total_latency_ms": sum(latencies),
                "mean_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "mean_memory_peak_delta_mb": statistics.mean(memory_deltas),
                "p95_memory_peak_delta_mb": _percentile(memory_deltas, 0.95),
            }
        )

    phase_rows = []
    phase_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get("experiment_batch_size"),
            record.get("cache_mode"),
            record.get("phase"),
        )
        phase_groups[key].append(record)
    for key, bucket in sorted(phase_groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        latencies = [float(row["model_call_latency_ms"]) for row in bucket]
        memory_deltas = [float(row["memory_peak_delta_mb"]) for row in bucket]
        phase_rows.append(
            {
                "batch_size": key[0],
                "cache_mode": key[1],
                "phase": key[2],
                "num_calls": len(bucket),
                "total_latency_ms": sum(latencies),
                "mean_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "mean_memory_peak_delta_mb": statistics.mean(memory_deltas),
                "p95_memory_peak_delta_mb": _percentile(memory_deltas, 0.95),
            }
        )

    ratios = []
    by_batch: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phase_rows:
        by_batch[int(row["batch_size"])][str(row["phase"])] = row
    for batch_size, phases in sorted(by_batch.items()):
        warmup = phases.get("warmup")
        refine = phases.get("refine")
        if warmup is None or refine is None:
            continue
        ratios.append(
            {
                "batch_size": batch_size,
                "warmup_mean_latency_ms": warmup["mean_latency_ms"],
                "refine_mean_latency_ms": refine["mean_latency_ms"],
                "warmup_to_refine_latency_ratio": (
                    warmup["mean_latency_ms"] / refine["mean_latency_ms"]
                    if refine["mean_latency_ms"]
                    else 0.0
                ),
                "warmup_mean_memory_delta_mb": warmup["mean_memory_peak_delta_mb"],
                "refine_mean_memory_delta_mb": refine["mean_memory_peak_delta_mb"],
                "warmup_to_refine_memory_delta_ratio": (
                    warmup["mean_memory_peak_delta_mb"]
                    / refine["mean_memory_peak_delta_mb"]
                    if refine["mean_memory_peak_delta_mb"]
                    else 0.0
                ),
                "warmup_time_fraction": (
                    warmup["total_latency_ms"]
                    / (warmup["total_latency_ms"] + refine["total_latency_ms"])
                    if warmup["total_latency_ms"] + refine["total_latency_ms"] > 0
                    else 0.0
                ),
            }
        )

    return {
        "num_records": len(records),
        "rows": rows,
        "phase_rows": phase_rows,
        "warmup_refine_ratios": ratios,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str | None = None
    prompt: str = "Explain refresh storms in diffusion LLM serving in one sentence."
    limit: int = 16
    batch_sizes: str = "1,2,4,8"
    seed: int = 42
    output_prefix: str = "artifacts/refresh_storm/llada_prefix"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "prefix"
    threshold: float | None = 0.9
    factor: float | None = None
    begin_suppress_tokens: list[int] | None = None
    return_dict: bool = False
    warmup_runs: int = 1


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

messages = _load_messages(
    input_path=script_args.input_path,
    prompt=script_args.prompt,
    limit=script_args.limit,
)
batch_sizes = _parse_int_list(script_args.batch_sizes)

fastdllm_config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
    script_args.model_name_or_path
)
model = dllm.utils.get_model(model_args=script_args, config=fastdllm_config).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.fastdllm.llada.FastdLLMLLaDASampler(
    model=model, tokenizer=tokenizer
)

all_records = []
run_rows = []
wall_start = time.perf_counter()
for batch_size in batch_sizes:
    batch_messages = [messages[index % len(messages)] for index in range(batch_size)]
    inputs = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    for _ in range(max(0, sampler_config.warmup_runs)):
        _ = sampler.sample(inputs, config=sampler_config, return_dict=False)

    profiler = ModelCallProfiler(batch_size=batch_size)
    run_start = time.perf_counter()
    outputs = sampler.sample(
        inputs,
        config=sampler_config,
        return_dict=False,
        model_call_observer=profiler,
    )
    run_elapsed_s = time.perf_counter() - run_start
    records = profiler.records
    all_records.extend(records)
    run_rows.append(
        {
            "batch_size": batch_size,
            "num_model_calls": len(records),
            "wall_elapsed_s": run_elapsed_s,
            "output_shape": list(outputs.shape) if hasattr(outputs, "shape") else None,
        }
    )
    print(
        f"batch={batch_size} calls={len(records)} wall={run_elapsed_s:.2f}s",
        flush=True,
    )

summary = _summarize_records(all_records)
summary.update(
    {
        "model_name_or_path": script_args.model_name_or_path,
        "batch_sizes": batch_sizes,
        "steps": sampler_config.steps,
        "max_new_tokens": sampler_config.max_new_tokens,
        "block_size": sampler_config.block_size,
        "use_cache": sampler_config.use_cache,
        "threshold": sampler_config.threshold,
        "factor": sampler_config.factor,
        "prompt_count": len(messages),
        "wall_elapsed_s": time.perf_counter() - wall_start,
        "runs": run_rows,
    }
)

prefix = Path(script_args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
records_path = prefix.with_suffix(".jsonl")
records_csv_path = prefix.with_suffix(".csv")
summary_path = prefix.with_name(prefix.name + "_summary.json")
phase_csv_path = prefix.with_name(prefix.name + "_phase_summary.csv")
ratio_csv_path = prefix.with_name(prefix.name + "_ratios.csv")

_write_jsonl(records_path, all_records)
_write_csv(records_csv_path, all_records)
_write_csv(phase_csv_path, summary["phase_rows"])
_write_csv(ratio_csv_path, summary["warmup_refine_ratios"])
summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

print(f"Saved records: {records_path}")
print(f"Saved summary: {summary_path}")
print("Warmup/refine ratios:")
for row in summary["warmup_refine_ratios"]:
    print(
        "batch={batch_size} latency_ratio={warmup_to_refine_latency_ratio:.2f} "
        "memory_ratio={warmup_to_refine_memory_delta_ratio:.2f} "
        "warmup_time_fraction={warmup_time_fraction:.2f}".format(**row)
    )
