"""
Benchmark dense versus shape-decoupled forward execution for elastic canvases.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/benchmark_elastic_canvas_forward.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --batch_sizes 1,2,4,8,16 \
    --patterns short,medium,long,mixed,mix50,mix75,mix90,lowvar,highvar,adversarial,trace \
    --trace_summary_path artifacts/elastic_canvas/llada256_summary.json \
    --output_path artifacts/elastic_canvas/forward_bench.json
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_inputs(inputs: Any) -> list[int]:
    if isinstance(inputs, torch.Tensor):
        if inputs.dim() == 1:
            return inputs.tolist()
        return inputs[0].tolist()
    if inputs and isinstance(inputs[0], int):
        return inputs
    return inputs[0]


def _parse_int_list(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths:
        raise ValueError("length list must not be empty")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"lengths must be positive: {lengths}")
    return lengths


def _parse_batch_sizes(value: str | None, batch_size: int) -> list[int]:
    if value is None:
        return [batch_size]
    return _parse_int_list(value)


def _repeat_to_batch(lengths: list[int], batch_size: int) -> list[int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    repeated = []
    while len(repeated) < batch_size:
        repeated.extend(lengths)
    return repeated[:batch_size]


def _pattern_lengths(
    *,
    pattern: str,
    batch_size: int,
    fixed_canvas: int,
    trace_summary_path: str | None,
) -> list[int]:
    if pattern.startswith("lengths:"):
        return _repeat_to_batch(_parse_int_list(pattern.split(":", 1)[1]), batch_size)
    if pattern == "short":
        return [32] * batch_size
    if pattern == "medium":
        return [128] * batch_size
    if pattern == "long":
        return [fixed_canvas] * batch_size
    if pattern == "mixed":
        return _repeat_to_batch([32, 64, 128, fixed_canvas], batch_size)
    if pattern == "mix50":
        num_short = batch_size // 2
        return [32] * num_short + [fixed_canvas] * (batch_size - num_short)
    if pattern == "mix75":
        num_short = int(round(batch_size * 0.75))
        num_short = min(max(num_short, 0), batch_size)
        return [32] * num_short + [fixed_canvas] * (batch_size - num_short)
    if pattern == "mix90":
        num_short = int(round(batch_size * 0.90))
        if batch_size > 1:
            num_short = min(num_short, batch_size - 1)
        return [32] * num_short + [fixed_canvas] * (batch_size - num_short)
    if pattern == "lowvar":
        center = min(max(128, 1), fixed_canvas)
        candidates = [
            max(1, min(fixed_canvas, center + offset))
            for offset in (-16, -8, 0, 8, 16)
        ]
        return _repeat_to_batch(candidates, batch_size)
    if pattern == "highvar":
        candidates = [32, 48, 64, 128, 192, fixed_canvas]
        candidates = [min(length, fixed_canvas) for length in candidates]
        return _repeat_to_batch(candidates, batch_size)
    if pattern == "adversarial":
        return [32] * max(0, batch_size - 1) + [fixed_canvas]
    if pattern == "trace":
        if trace_summary_path is None:
            raise ValueError("pattern 'trace' requires --trace_summary_path")
        with Path(trace_summary_path).open("r", encoding="utf-8") as f:
            summary = json.load(f)
        lengths = [int(x) for x in summary["per_request_elastic_lengths"]]
        return _repeat_to_batch(lengths, batch_size)
    raise ValueError(f"Unknown pattern: {pattern}")


def _build_batch(
    *,
    prompt_ids: list[int],
    canvas_lengths: list[int],
    physical_canvas: int,
    mask_token_id: int,
    eos_token_id: int,
    device: torch.device,
    mask_inactive_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(canvas_lengths)
    prompt_len = len(prompt_ids)
    total_len = prompt_len + physical_canvas
    input_ids = torch.full(
        (batch_size, total_len),
        eos_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, total_len),
        dtype=torch.long,
        device=device,
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    for row, canvas_len in enumerate(canvas_lengths):
        if canvas_len > physical_canvas:
            raise ValueError(
                f"canvas_len={canvas_len} exceeds physical_canvas={physical_canvas}"
            )
        input_ids[row, :prompt_len] = prompt_tensor
        input_ids[row, prompt_len : prompt_len + canvas_len] = mask_token_id
        valid_canvas = canvas_len if mask_inactive_tail else physical_canvas
        attention_mask[row, : prompt_len + valid_canvas] = 1
    return input_ids, attention_mask


def _run_forward_once(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    _ = model(input_ids=input_ids, attention_mask=attention_mask).logits


def _measure(
    *,
    model,
    calls: list[tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    device = model.device
    with torch.inference_mode():
        for _ in range(warmup):
            for input_ids, attention_mask in calls:
                _run_forward_once(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        _sync_device(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for input_ids, attention_mask in calls:
                _run_forward_once(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            _sync_device(device)
            timings.append(time.perf_counter() - start)

    timings_ms = [value * 1000.0 for value in timings]
    return {
        "avg_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "num_calls": len(calls),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        ),
    }


def _bucket_calls(
    *,
    prompt_ids: list[int],
    canvas_lengths: list[int],
    mask_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for length in canvas_lengths:
        buckets[length].append(length)

    calls = []
    for physical_canvas, bucket_lengths in sorted(buckets.items()):
        calls.append(
            _build_batch(
                prompt_ids=prompt_ids,
                canvas_lengths=bucket_lengths,
                physical_canvas=physical_canvas,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                device=device,
                mask_inactive_tail=False,
            )
        )
    return calls


def _token_units(prompt_len: int, canvas_lengths: list[int], physical_canvas: int) -> int:
    return len(canvas_lengths) * (prompt_len + physical_canvas)


def _attention_units(
    prompt_len: int, canvas_lengths: list[int], physical_canvas: int | None
) -> int:
    if physical_canvas is not None:
        seq_len = prompt_len + physical_canvas
        return len(canvas_lengths) * seq_len * seq_len
    return sum((prompt_len + length) ** 2 for length in canvas_lengths)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    prompt: str = "Explain elastic canvas serving in one sentence."
    fixed_canvas: int = 256
    batch_size: int = 8
    batch_sizes: str | None = None
    patterns: str = "short,medium,long,mixed,mix50,mix75,mix90,lowvar,highvar,adversarial"
    trace_summary_path: str | None = None
    warmup: int = 2
    repeats: int = 5
    seed: int = 42
    output_path: str = "artifacts/elastic_canvas/forward_bench.json"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


parser = transformers.HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
transformers.set_seed(script_args.seed)

config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
    script_args.model_name_or_path
)
model = dllm.utils.get_model(model_args=script_args, config=config).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)

inputs = tokenizer.apply_chat_template(
    [[{"role": "user", "content": script_args.prompt}]],
    add_generation_prompt=True,
    tokenize=True,
)
prompt_ids = _normalize_inputs(inputs)
mask_token_id = tokenizer.mask_token_id
eos_token_id = tokenizer.eos_token_id
if mask_token_id is None:
    raise ValueError("tokenizer.mask_token_id is required")
if eos_token_id is None:
    raise ValueError("tokenizer.eos_token_id is required")

results = []
for batch_size in _parse_batch_sizes(script_args.batch_sizes, script_args.batch_size):
    for pattern in [
        part.strip() for part in script_args.patterns.split(",") if part.strip()
    ]:
        canvas_lengths = _pattern_lengths(
            pattern=pattern,
            batch_size=batch_size,
            fixed_canvas=script_args.fixed_canvas,
            trace_summary_path=script_args.trace_summary_path,
        )
        max_canvas = max(canvas_lengths)

        dense_fixed_call = _build_batch(
            prompt_ids=prompt_ids,
            canvas_lengths=[script_args.fixed_canvas] * len(canvas_lengths),
            physical_canvas=script_args.fixed_canvas,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            device=model.device,
            mask_inactive_tail=False,
        )
        elastic_dense_call = _build_batch(
            prompt_ids=prompt_ids,
            canvas_lengths=canvas_lengths,
            physical_canvas=max_canvas,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            device=model.device,
            mask_inactive_tail=True,
        )
        bucketed_calls = _bucket_calls(
            prompt_ids=prompt_ids,
            canvas_lengths=canvas_lengths,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            device=model.device,
        )

        dense_fixed = _measure(
            model=model,
            calls=[dense_fixed_call],
            warmup=script_args.warmup,
            repeats=script_args.repeats,
        )
        elastic_dense = _measure(
            model=model,
            calls=[elastic_dense_call],
            warmup=script_args.warmup,
            repeats=script_args.repeats,
        )
        bucketed = _measure(
            model=model,
            calls=bucketed_calls,
            warmup=script_args.warmup,
            repeats=script_args.repeats,
        )

        dense_fixed_ms = float(dense_fixed["avg_ms"])
        elastic_dense_ms = float(elastic_dense["avg_ms"])
        bucketed_ms = float(bucketed["avg_ms"])
        result = {
            "pattern": pattern,
            "batch_size": len(canvas_lengths),
            "prompt_tokens": len(prompt_ids),
            "canvas_lengths": canvas_lengths,
            "fixed_canvas": script_args.fixed_canvas,
            "max_elastic_canvas": max_canvas,
            "num_bucketed_calls": len(bucketed_calls),
            "fixed_physical_tokens": _token_units(
                len(prompt_ids), canvas_lengths, script_args.fixed_canvas
            ),
            "elastic_dense_physical_tokens": _token_units(
                len(prompt_ids), canvas_lengths, max_canvas
            ),
            "bucketed_physical_tokens": sum(
                input_ids.numel() for input_ids, _ in bucketed_calls
            ),
            "fixed_attention_units": _attention_units(
                len(prompt_ids), canvas_lengths, script_args.fixed_canvas
            ),
            "elastic_dense_attention_units": _attention_units(
                len(prompt_ids), canvas_lengths, max_canvas
            ),
            "bucketed_attention_units": _attention_units(
                len(prompt_ids), canvas_lengths, None
            ),
            "dense_fixed": dense_fixed,
            "elastic_dense": elastic_dense,
            "bucketed_shape_decoupled": bucketed,
            "elastic_dense_speedup_vs_fixed": dense_fixed_ms / elastic_dense_ms
            if elastic_dense_ms > 0
            else 0.0,
            "bucketed_speedup_vs_fixed": dense_fixed_ms / bucketed_ms
            if bucketed_ms > 0
            else 0.0,
            "bucketed_speedup_vs_elastic_dense": elastic_dense_ms / bucketed_ms
            if bucketed_ms > 0
            else 0.0,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=True, indent=2))

output = {
    "model_name_or_path": script_args.model_name_or_path,
    "prompt": script_args.prompt,
    "warmup": script_args.warmup,
    "repeats": script_args.repeats,
    "batch_sizes": _parse_batch_sizes(
        script_args.batch_sizes, script_args.batch_size
    ),
    "results": results,
}
output_path = Path(script_args.output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=True, indent=2)
print("Saved:", output_path)
