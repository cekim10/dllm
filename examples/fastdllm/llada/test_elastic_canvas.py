"""
Measure fixed-canvas waste and oracle elastic-canvas serving headroom.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_elastic_canvas.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm
from dllm.pipelines.fastdllm.llada.elastic_canvas import (
    ElasticCanvasConfig,
    summarize_elastic_canvas,
)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_messages(
    input_path: str | None, prompt: str, limit: int
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
                    text = (
                        record.get("prompt")
                        or record.get("text")
                        or record.get("input")
                    )
                    if text is None:
                        raise ValueError(f"Unsupported JSONL record: {record}")
                    messages.append([{"role": "user", "content": text}])
            else:
                messages.append([{"role": "user", "content": line}])
            if len(messages) >= limit:
                break
    return messages


def _normalize_inputs(inputs: Any) -> list[list[int]]:
    if isinstance(inputs, torch.Tensor):
        if inputs.dim() == 1:
            return [inputs.tolist()]
        return inputs.tolist()
    if inputs and isinstance(inputs[0], int):
        return [inputs]
    return inputs


def _generated_span_length(
    *,
    tokenizer,
    sequence: list[int],
    prompt_ids: list[int],
    decoded_text: str,
    max_new_tokens: int,
    length_source: str,
) -> int:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = getattr(tokenizer, "eot_token_id", None)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    stop_ids = {tok for tok in (eos_id, eot_id, mask_id) if tok is not None}

    start = len(prompt_ids)
    span = sequence[start : start + max_new_tokens]
    eos_length = len(span)
    for idx, token_id in enumerate(span):
        if token_id in stop_ids:
            eos_length = idx
            break

    decoded_length = len(
        tokenizer(decoded_text, add_special_tokens=False).input_ids
        if decoded_text
        else []
    )

    if length_source == "sequence_eos":
        return eos_length
    if length_source == "decoded_tokens":
        return min(decoded_length, max_new_tokens)
    if length_source == "min_eos_decoded":
        return min(eos_length, decoded_length, max_new_tokens)
    raise ValueError(f"Unknown length_source: {length_source}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_reallocation(
    *,
    device: torch.device,
    dtype: torch.dtype,
    lengths: list[int],
    d_model: int,
    initial_canvas: int,
    page_size: int,
    repeats: int,
) -> dict[str, float]:
    if repeats <= 0:
        return {"avg_ms": 0.0, "num_expansions": 0.0}

    elapsed = 0.0
    expansions = 0
    for target in lengths:
        current = initial_canvas
        state = torch.empty((1, current, d_model), device=device, dtype=dtype)
        while current < target:
            next_len = min(target, current + page_size)
            _sync_device(device)
            start = time.perf_counter()
            for _ in range(repeats):
                expanded = torch.empty((1, next_len, d_model), device=device, dtype=dtype)
                expanded[:, :current].copy_(state)
            _sync_device(device)
            elapsed += time.perf_counter() - start
            expansions += repeats
            state = expanded
            current = next_len
    return {
        "avg_ms": (elapsed / expansions) * 1000.0 if expansions else 0.0,
        "num_expansions": float(expansions),
    }


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str | None = None
    prompt: str = "What is 2 + 2? Answer briefly."
    limit: int = 16
    seed: int = 42
    output_prefix: str = "artifacts/elastic_canvas/elastic"
    length_source: str = "decoded_tokens"
    initial_canvas: int = 32
    page_size: int = 32
    batch_size: int = 8
    benchmark_reallocation: bool = False
    reallocation_repeats: int = 5

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    steps: int = 256
    max_new_tokens: int = 256
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "prefix"
    threshold: float | None = 0.9
    factor: float | None = None
    begin_suppress_tokens: list[int] | None = None


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

messages = _load_messages(
    input_path=script_args.input_path,
    prompt=script_args.prompt,
    limit=script_args.limit,
)
fastdllm_config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
    script_args.model_name_or_path
)
model = dllm.utils.get_model(model_args=script_args, config=fastdllm_config).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.fastdllm.llada.FastdLLMLLaDASampler(
    model=model, tokenizer=tokenizer
)

records = []
for request_index, message in enumerate(messages):
    inputs = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = _normalize_inputs(inputs)[0]

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    _sync_device(model.device)
    start_time = time.perf_counter()
    outputs = sampler.sample(inputs, config=sampler_config, return_dict=True)
    _sync_device(model.device)
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(model.device) / (1024 * 1024)
        if model.device.type == "cuda"
        else None
    )

    sequence = outputs.sequences[0].tolist()
    decoded = dllm.utils.sample_trim(tokenizer, [sequence], [prompt_ids])[0]
    useful_length = _generated_span_length(
        tokenizer=tokenizer,
        sequence=sequence,
        prompt_ids=prompt_ids,
        decoded_text=decoded,
        max_new_tokens=sampler_config.max_new_tokens,
        length_source=script_args.length_source,
    )
    records.append(
        {
            "request_index": request_index,
            "prompt_tokens": len(prompt_ids),
            "fixed_canvas": sampler_config.max_new_tokens,
            "useful_generated_tokens": useful_length,
            "fixed_padding_tokens": sampler_config.max_new_tokens - useful_length,
            "fixed_useful_ratio": useful_length / sampler_config.max_new_tokens,
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_mb": peak_memory_mb,
            "num_histories": len(outputs.histories) if outputs.histories else None,
            "decoded_chars": len(decoded),
            "decoded_text": decoded,
        }
    )
    print(
        json.dumps(
            {
                "request_index": request_index,
                "useful_generated_tokens": useful_length,
                "fixed_useful_ratio": records[-1]["fixed_useful_ratio"],
                "elapsed_seconds": elapsed_seconds,
                "peak_memory_mb": peak_memory_mb,
            },
            ensure_ascii=True,
        )
    )

useful_lengths = [int(record["useful_generated_tokens"]) for record in records]
summary = summarize_elastic_canvas(
    useful_lengths,
    ElasticCanvasConfig(
        fixed_canvas=sampler_config.max_new_tokens,
        initial_canvas=script_args.initial_canvas,
        page_size=script_args.page_size,
        batch_size=script_args.batch_size,
        steps=sampler_config.steps,
    ),
)
summary.update(
    {
        "model_name_or_path": script_args.model_name_or_path,
        "length_source": script_args.length_source,
        "total_elapsed_seconds": sum(float(r["elapsed_seconds"]) for r in records),
        "avg_elapsed_seconds": sum(float(r["elapsed_seconds"]) for r in records)
        / len(records),
        "avg_peak_memory_mb": (
            sum(float(r["peak_memory_mb"]) for r in records) / len(records)
            if records[0]["peak_memory_mb"] is not None
            else None
        ),
        "max_peak_memory_mb": (
            max(float(r["peak_memory_mb"]) for r in records)
            if records[0]["peak_memory_mb"] is not None
            else None
        ),
        "avg_useful_generated_tokens": sum(useful_lengths) / len(useful_lengths),
        "max_useful_generated_tokens": max(useful_lengths),
        "min_useful_generated_tokens": min(useful_lengths),
    }
)

if script_args.benchmark_reallocation:
    dtype = next(model.parameters()).dtype
    summary["naive_reallocation_benchmark"] = _benchmark_reallocation(
        device=model.device,
        dtype=dtype,
        lengths=summary["per_request_elastic_lengths"],
        d_model=model.config.d_model,
        initial_canvas=script_args.initial_canvas,
        page_size=script_args.page_size,
        repeats=script_args.reallocation_repeats,
    )

prefix = Path(script_args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
summary_path = prefix.with_name(prefix.name + "_summary.json")
records_path = prefix.with_name(prefix.name + "_records.jsonl")
csv_path = prefix.with_name(prefix.name + "_records.csv")
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=True, indent=2)
with records_path.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
_write_csv(csv_path, records)

print("Saved summary:", summary_path)
print("Saved records:", records_path)
print(
    json.dumps(
        {
            "num_requests": summary["num_requests"],
            "fixed_useful_ratio": summary["fixed_useful_ratio"],
            "oracle_token_volume_reduction": summary[
                "oracle_token_volume_reduction"
            ],
            "oracle_attention_volume_reduction": summary[
                "oracle_attention_volume_reduction"
            ],
            "elastic_dense_padding_tokens": summary["elastic_dense_padding_tokens"],
            "total_growth_events": summary["total_growth_events"],
        },
        ensure_ascii=True,
        indent=2,
    )
)
