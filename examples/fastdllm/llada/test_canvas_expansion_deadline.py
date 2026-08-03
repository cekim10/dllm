"""
Sweep the latest safe step for expanding a small LLaDA response canvas.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_canvas_expansion_deadline.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/elastic_canvas_prompts.jsonl \
    --limit 10 \
    --expansion_after_steps "0,1,2,4,8,12,16" \
    --output_prefix artifacts/elastic_canvas/expansion_deadline
"""

from __future__ import annotations

import csv
import difflib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.fastdllm.llada.sampler import _get_transfer_index


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
                    text = record.get("prompt") or record.get("text") or record.get("input")
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


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    if any(value < 0 for value in values):
        raise ValueError(f"Values must be non-negative: {values}")
    return values


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


def _token_edit_distance(a: list[int], b: list[int]) -> int:
    rows = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        prev = rows[0]
        rows[0] = i
        for j, token_b in enumerate(b, start=1):
            old = rows[j]
            cost = 0 if token_a == token_b else 1
            rows[j] = min(rows[j] + 1, rows[j - 1] + 1, prev + cost)
            prev = old
    return rows[-1]


def _apply_suppressions(
    logits: torch.Tensor,
    suppress_tokens: list[int] | None,
    begin_suppress_tokens: list[int] | None,
) -> None:
    if suppress_tokens:
        for token_id in suppress_tokens:
            logits[:, :, token_id] = -torch.inf
    if begin_suppress_tokens:
        for token_id in begin_suppress_tokens:
            logits[:, :, token_id] = -torch.inf


def _collect_boundary_signal(
    *,
    logits: torch.Tensor,
    x: torch.Tensor,
    tokenizer,
    prompt_len: int,
    visible_canvas: int,
    tail_width: int,
) -> dict[str, float]:
    if visible_canvas <= 0:
        return {
            "tail_mask_fraction": 0.0,
            "eos_margin_mean": 0.0,
            "eos_margin_max": 0.0,
            "boundary_confidence_mean": 0.0,
            "boundary_confidence_max": 0.0,
        }

    tail_len = min(tail_width, visible_canvas)
    start = prompt_len + visible_canvas - tail_len
    end = prompt_len + visible_canvas
    tail_tokens = x[:, start:end]
    tail_logits = logits[:, start:end, :].to(torch.float32)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = getattr(tokenizer, "eot_token_id", None)
    excluded = [token_id for token_id in (eos_id, eot_id) if token_id is not None]

    non_eos_logits = tail_logits.clone()
    for token_id in excluded:
        non_eos_logits[:, :, token_id] = -torch.inf

    if eos_id is None:
        eos_margin = torch.zeros(tail_logits.shape[:2], device=tail_logits.device)
    else:
        eos_margin = tail_logits[:, :, eos_id] - non_eos_logits.max(dim=-1).values

    probs = torch.softmax(non_eos_logits, dim=-1)
    boundary_confidence = probs.max(dim=-1).values
    mask_id = tokenizer.mask_token_id
    tail_mask_fraction = (tail_tokens == mask_id).to(torch.float32).mean()

    return {
        "tail_mask_fraction": float(tail_mask_fraction.detach().cpu().item()),
        "eos_margin_mean": float(eos_margin.mean().detach().cpu().item()),
        "eos_margin_max": float(eos_margin.max().detach().cpu().item()),
        "boundary_confidence_mean": float(
            boundary_confidence.mean().detach().cpu().item()
        ),
        "boundary_confidence_max": float(boundary_confidence.max().detach().cpu().item()),
    }


@torch.inference_mode()
def _sample_with_expansion_deadline(
    *,
    model,
    tokenizer,
    prompt_ids: list[int],
    initial_canvas: int,
    final_canvas: int,
    expansion_after_steps: int,
    config,
    scheduler,
    tail_width: int,
) -> tuple[list[int], dict[str, Any], list[dict[str, Any]]]:
    if initial_canvas <= 0 or final_canvas <= 0:
        raise ValueError("Canvas lengths must be positive.")
    if initial_canvas > final_canvas:
        raise ValueError("initial_canvas must be <= final_canvas.")

    device = model.device
    mask_id = tokenizer.mask_token_id
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    visible_canvas = final_canvas if expansion_after_steps == 0 else initial_canvas
    x = torch.cat(
        [
            prompt,
            torch.full((1, visible_canvas), mask_id, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)

    def expand_if_needed(reason: str, first_block_steps: int) -> tuple[bool, str]:
        nonlocal x, attention_mask, visible_canvas
        if visible_canvas >= final_canvas:
            return False, reason
        added = final_canvas - visible_canvas
        x = torch.cat(
            [
                x,
                torch.full((1, added), mask_id, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        attention_mask = torch.ones_like(x, dtype=torch.long, device=device)
        visible_canvas = final_canvas
        return True, f"{reason}@first_block_step_{first_block_steps}"

    num_blocks = math.ceil(final_canvas / config.block_size)
    steps_per_block = math.ceil(config.steps / max(num_blocks, 1))
    model_calls = 0
    response_token_work = 0
    total_token_work = 0
    response_attention_work = 0
    total_attention_work = 0
    first_block_steps = 0
    expanded_at_first_block_step = 0 if expansion_after_steps == 0 else None
    expansion_reason = "initial_full_canvas" if expansion_after_steps == 0 else None
    signals = []

    for block_index in range(num_blocks):
        block_start = len(prompt_ids) + block_index * config.block_size
        block_end_final = min(block_start + config.block_size, len(prompt_ids) + final_canvas)

        if block_start >= len(prompt_ids) + visible_canvas:
            expanded, reason = expand_if_needed("needed_for_next_block", first_block_steps)
            if expanded:
                expanded_at_first_block_step = first_block_steps
                expansion_reason = reason

        block_end = min(block_end_final, len(prompt_ids) + visible_canvas)
        block_width = block_end - block_start
        if block_width <= 0:
            continue

        block_mask_index = torch.zeros(
            (1, config.block_size), dtype=torch.bool, device=device
        )
        block_mask_index[:, :block_width] = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index,
            steps=steps_per_block,
            scheduler=scheduler,
            stochastic=config.stochastic_transfer,
        )
        effective_steps = max(int(num_transfer_tokens.size(1)), 1)
        step_index = 0

        while (x[:, block_start:block_end] == mask_id).any():
            if (
                block_index == 0
                and visible_canvas < final_canvas
                and first_block_steps >= expansion_after_steps
            ):
                expanded, reason = expand_if_needed(
                    "deadline_before_model_call", first_block_steps
                )
                if expanded:
                    expanded_at_first_block_step = first_block_steps
                    expansion_reason = reason
                    block_end = min(
                        block_end_final, len(prompt_ids) + visible_canvas
                    )

            if config.threshold is None and config.factor is None:
                if step_index >= effective_steps:
                    break
                quota = num_transfer_tokens[:, step_index]
            else:
                quota = None
                if step_index > steps_per_block + config.block_size + 4:
                    raise RuntimeError(
                        "Expansion deadline sampler exceeded refinement safety cap. "
                        "Check threshold/factor settings."
                    )

            mask_allowed = torch.zeros_like(x, dtype=torch.bool)
            mask_allowed[:, block_start:block_end] = (
                x[:, block_start:block_end] == mask_id
            )
            out = model(x, attention_mask=attention_mask)
            logits = out.logits
            model_calls += 1
            response_token_work += visible_canvas
            total_token_work += int(x.shape[1])
            response_attention_work += visible_canvas * visible_canvas
            total_attention_work += int(x.shape[1]) * int(x.shape[1])

            _apply_suppressions(
                logits,
                config.suppress_tokens,
                config.begin_suppress_tokens,
            )
            if config.right_shift_logits:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            if block_index == 0:
                signal = _collect_boundary_signal(
                    logits=logits,
                    x=x,
                    tokenizer=tokenizer,
                    prompt_len=len(prompt_ids),
                    visible_canvas=visible_canvas,
                    tail_width=tail_width,
                )
                signal.update(
                    {
                        "first_block_step": int(first_block_steps),
                        "visible_canvas": int(visible_canvas),
                        "expanded": bool(visible_canvas == final_canvas),
                        "model_call_index": int(model_calls - 1),
                    }
                )
                signals.append(signal)

            x0, transfer_idx = _get_transfer_index(
                logits=logits,
                temperature=config.temperature,
                remasking=config.remasking,
                mask_index=mask_allowed,
                x=x,
                num_transfer_tokens=quota,
                threshold=config.threshold,
                factor=config.factor,
            )
            x = torch.where(transfer_idx, x0, x)
            step_index += 1
            if block_index == 0:
                first_block_steps += 1

    sequence = x[0].detach().cpu().tolist()
    stats = {
        "initial_canvas": int(initial_canvas),
        "final_canvas": int(final_canvas),
        "expansion_after_steps": int(expansion_after_steps),
        "expanded_at_first_block_step": expanded_at_first_block_step,
        "expansion_reason": expansion_reason,
        "model_calls": int(model_calls),
        "steps_per_block": int(steps_per_block),
        "response_token_work": int(response_token_work),
        "total_token_work": int(total_token_work),
        "response_attention_work": int(response_attention_work),
        "total_attention_work": int(total_attention_work),
    }
    return sequence, stats, signals


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str | None = None
    prompt: str = "Explain why caching helps diffusion language model serving."
    limit: int = 10
    seed: int = 42
    output_prefix: str = "artifacts/elastic_canvas/expansion_deadline"
    initial_canvas: int = 32
    final_canvas: int = 256
    expansion_after_steps: str = "0,1,2,4,8,12,16"
    tail_width: int = 8
    length_source: str = "decoded_tokens"

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
    stochastic_transfer: bool = False
    threshold: float | None = 0.9
    factor: float | None = None
    use_cache: str | None = None
    begin_suppress_tokens: list[int] | None = None


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

if sampler_config.max_new_tokens != script_args.final_canvas:
    raise ValueError(
        "--max_new_tokens must match --final_canvas for this deadline test."
    )

deadlines = _parse_int_list(script_args.expansion_after_steps)
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
signal_records = []
for request_index, message in enumerate(messages):
    inputs = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = _normalize_inputs(inputs)[0]
    request_records = []

    for deadline in deadlines:
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        _sync_device(model.device)
        start_time = time.perf_counter()
        sequence, stats, signals = _sample_with_expansion_deadline(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            initial_canvas=script_args.initial_canvas,
            final_canvas=script_args.final_canvas,
            expansion_after_steps=deadline,
            config=sampler_config,
            scheduler=sampler.scheduler,
            tail_width=script_args.tail_width,
        )
        _sync_device(model.device)
        elapsed_seconds = time.perf_counter() - start_time
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(model.device) / (1024 * 1024)
            if model.device.type == "cuda"
            else None
        )

        decoded = dllm.utils.sample_trim(tokenizer, [sequence], [prompt_ids])[0]
        useful_length = _generated_span_length(
            tokenizer=tokenizer,
            sequence=sequence,
            prompt_ids=prompt_ids,
            decoded_text=decoded,
            max_new_tokens=script_args.final_canvas,
            length_source=script_args.length_source,
        )
        gen_tokens = sequence[
            len(prompt_ids) : len(prompt_ids) + script_args.final_canvas
        ]
        pre_expand_signals = [
            signal for signal in signals if not bool(signal["expanded"])
        ]
        last_pre_expand_signal = pre_expand_signals[-1] if pre_expand_signals else {}
        record = {
            "request_index": request_index,
            "variant": f"expand_after_{deadline}",
            "expansion_after_steps": deadline,
            "prompt_tokens": len(prompt_ids),
            "initial_canvas": stats["initial_canvas"],
            "final_canvas": stats["final_canvas"],
            "expanded_at_first_block_step": stats["expanded_at_first_block_step"],
            "expansion_reason": stats["expansion_reason"],
            "model_calls": stats["model_calls"],
            "steps_per_block": stats["steps_per_block"],
            "response_token_work": stats["response_token_work"],
            "total_token_work": stats["total_token_work"],
            "response_attention_work": stats["response_attention_work"],
            "total_attention_work": stats["total_attention_work"],
            "useful_generated_tokens": useful_length,
            "decoded_chars": len(decoded),
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_mb": peak_memory_mb,
            "pre_expand_tail_mask_fraction": last_pre_expand_signal.get(
                "tail_mask_fraction"
            ),
            "pre_expand_eos_margin_mean": last_pre_expand_signal.get(
                "eos_margin_mean"
            ),
            "pre_expand_eos_margin_max": last_pre_expand_signal.get("eos_margin_max"),
            "pre_expand_boundary_confidence_mean": last_pre_expand_signal.get(
                "boundary_confidence_mean"
            ),
            "pre_expand_boundary_confidence_max": last_pre_expand_signal.get(
                "boundary_confidence_max"
            ),
            "decoded_text": decoded,
            "generated_token_ids": gen_tokens,
        }
        request_records.append(record)
        for signal in signals:
            signal_record = dict(signal)
            signal_record.update(
                {
                    "request_index": request_index,
                    "variant": f"expand_after_{deadline}",
                    "expansion_after_steps": deadline,
                }
            )
            signal_records.append(signal_record)

    fixed_record = next(
        record for record in request_records if int(record["expansion_after_steps"]) == 0
    )
    fixed_tokens = fixed_record["generated_token_ids"]
    fixed_text = fixed_record["decoded_text"]
    fixed_response_token_work = max(int(fixed_record["response_token_work"]), 1)
    fixed_total_attention_work = max(int(fixed_record["total_attention_work"]), 1)
    for record in request_records:
        edit_distance = _token_edit_distance(fixed_tokens, record["generated_token_ids"])
        max_tokens = max(len(fixed_tokens), len(record["generated_token_ids"]), 1)
        text_similarity = difflib.SequenceMatcher(
            None, fixed_text, record["decoded_text"]
        ).ratio()
        record.update(
            {
                "text_similarity_vs_fixed": text_similarity,
                "token_edit_distance_vs_fixed": edit_distance,
                "token_edit_ratio_vs_fixed": edit_distance / max_tokens,
                "length_delta_vs_fixed": int(record["useful_generated_tokens"])
                - int(fixed_record["useful_generated_tokens"]),
                "response_token_work_ratio_vs_fixed": int(
                    record["response_token_work"]
                )
                / fixed_response_token_work,
                "total_attention_work_ratio_vs_fixed": int(
                    record["total_attention_work"]
                )
                / fixed_total_attention_work,
            }
        )
        records.append(record)

    print(
        json.dumps(
            {
                "request_index": request_index,
                "fixed_useful_tokens": fixed_record["useful_generated_tokens"],
                "variants": [
                    {
                        "deadline": r["expansion_after_steps"],
                        "useful_tokens": r["useful_generated_tokens"],
                        "similarity": round(r["text_similarity_vs_fixed"], 4),
                        "token_edit_ratio": round(r["token_edit_ratio_vs_fixed"], 4),
                        "work_ratio": round(
                            r["response_token_work_ratio_vs_fixed"], 4
                        ),
                    }
                    for r in request_records
                ],
            },
            ensure_ascii=True,
        )
    )


summary_by_variant = {}
for variant in sorted({record["variant"] for record in records}):
    group = [record for record in records if record["variant"] == variant]
    summary_by_variant[variant] = {
        "num_requests": len(group),
        "avg_text_similarity_vs_fixed": sum(
            float(record["text_similarity_vs_fixed"]) for record in group
        )
        / len(group),
        "avg_token_edit_ratio_vs_fixed": sum(
            float(record["token_edit_ratio_vs_fixed"]) for record in group
        )
        / len(group),
        "avg_length_delta_vs_fixed": sum(
            int(record["length_delta_vs_fixed"]) for record in group
        )
        / len(group),
        "avg_useful_generated_tokens": sum(
            int(record["useful_generated_tokens"]) for record in group
        )
        / len(group),
        "avg_response_token_work_ratio_vs_fixed": sum(
            float(record["response_token_work_ratio_vs_fixed"]) for record in group
        )
        / len(group),
        "avg_total_attention_work_ratio_vs_fixed": sum(
            float(record["total_attention_work_ratio_vs_fixed"]) for record in group
        )
        / len(group),
        "avg_elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in group)
        / len(group),
        "avg_peak_memory_mb": (
            sum(float(record["peak_memory_mb"]) for record in group) / len(group)
            if group[0]["peak_memory_mb"] is not None
            else None
        ),
        "avg_model_calls": sum(int(record["model_calls"]) for record in group)
        / len(group),
    }

summary = {
    "model_name_or_path": script_args.model_name_or_path,
    "num_requests": len(messages),
    "initial_canvas": script_args.initial_canvas,
    "final_canvas": script_args.final_canvas,
    "expansion_after_steps": deadlines,
    "block_size": sampler_config.block_size,
    "steps": sampler_config.steps,
    "threshold": sampler_config.threshold,
    "factor": sampler_config.factor,
    "tail_width": script_args.tail_width,
    "length_source": script_args.length_source,
    "note": (
        "deadline=0 is the fixed/full-canvas oracle. deadline=k starts with "
        "initial_canvas visible and expands to final_canvas before the first "
        "block's k-th model call."
    ),
    "variants": summary_by_variant,
}

prefix = Path(script_args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
summary_path = prefix.with_name(prefix.name + "_summary.json")
records_path = prefix.with_name(prefix.name + "_records.jsonl")
signals_path = prefix.with_name(prefix.name + "_signals.jsonl")
csv_path = prefix.with_name(prefix.name + "_records.csv")
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=True, indent=2)
with records_path.open("w", encoding="utf-8") as f:
    for record in records:
        serializable = dict(record)
        serializable.pop("generated_token_ids", None)
        f.write(json.dumps(serializable, ensure_ascii=True) + "\n")
with signals_path.open("w", encoding="utf-8") as f:
    for record in signal_records:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
_write_csv(
    csv_path,
    [{k: v for k, v in record.items() if k != "generated_token_ids"} for record in records],
)

print("Saved summary:", summary_path)
print("Saved records:", records_path)
print("Saved signals:", signals_path)
print(
    json.dumps(
        {
            "num_requests": summary["num_requests"],
            "variants": summary["variants"],
        },
        ensure_ascii=True,
        indent=2,
    )
)
