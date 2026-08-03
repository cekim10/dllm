"""
Test whether online response-canvas expansion preserves LLaDA generation quality.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_canvas_expansion_quality.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/elastic_canvas_prompts.jsonl \
    --limit 10 \
    --output_prefix artifacts/elastic_canvas/expansion_quality
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
from dllm.pipelines.fastdllm.llada.elastic_canvas import round_canvas_length
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


def _parse_schedules(text: str) -> list[list[int]]:
    schedules = []
    for raw_schedule in text.split(";"):
        raw_schedule = raw_schedule.strip()
        if not raw_schedule:
            continue
        schedule = [int(part.strip()) for part in raw_schedule.split(",") if part.strip()]
        if not schedule:
            continue
        if schedule != sorted(schedule):
            raise ValueError(f"Canvas schedule must be nondecreasing: {schedule}")
        schedules.append(schedule)
    if not schedules:
        raise ValueError("At least one expansion schedule is required.")
    return schedules


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


@torch.inference_mode()
def _sample_visible_canvas_schedule(
    *,
    model,
    tokenizer,
    prompt_ids: list[int],
    schedule: list[int],
    config,
    scheduler,
    steps_reference_canvas: int,
) -> tuple[list[int], dict[str, Any]]:
    if schedule[-1] < 1:
        raise ValueError(f"Invalid canvas schedule: {schedule}")

    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    x = torch.cat(
        [
            prompt,
            torch.full((1, schedule[0]), mask_id, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)

    reference_blocks = math.ceil(steps_reference_canvas / config.block_size)
    steps_per_block = math.ceil(config.steps / max(reference_blocks, 1))
    next_block_index = 0
    model_calls = 0
    expansion_events = 0
    visible_canvas = schedule[0]

    for schedule_index, target_canvas in enumerate(schedule):
        if target_canvas > visible_canvas:
            added = target_canvas - visible_canvas
            x = torch.cat(
                [
                    x,
                    torch.full((1, added), mask_id, dtype=torch.long, device=device),
                ],
                dim=1,
            )
            attention_mask = torch.ones_like(x, dtype=torch.long, device=device)
            visible_canvas = target_canvas
            expansion_events += 1
        elif schedule_index > 0 and target_canvas < visible_canvas:
            raise ValueError(f"Canvas schedule must be nondecreasing: {schedule}")

        target_blocks = math.ceil(target_canvas / config.block_size)
        while next_block_index < target_blocks:
            block_start = len(prompt_ids) + next_block_index * config.block_size
            block_end = min(
                block_start + config.block_size,
                len(prompt_ids) + target_canvas,
            )
            block_width = block_end - block_start
            if block_width <= 0:
                next_block_index += 1
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
                if config.threshold is None and config.factor is None:
                    if step_index >= effective_steps:
                        break
                    quota = num_transfer_tokens[:, step_index]
                else:
                    # Threshold/factor modes force at least one transfer per call.
                    quota = None
                    if step_index > steps_per_block + config.block_size + 4:
                        raise RuntimeError(
                            "Expansion sampler exceeded refinement safety cap. "
                            "Check threshold/factor settings."
                        )

                mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                mask_allowed[:, block_start:block_end] = (
                    x[:, block_start:block_end] == mask_id
                )
                out = model(x, attention_mask=attention_mask)
                logits = out.logits
                model_calls += 1

                _apply_suppressions(
                    logits,
                    config.suppress_tokens,
                    config.begin_suppress_tokens,
                )
                if config.right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

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

            next_block_index += 1

    sequence = x[0].detach().cpu().tolist()
    stats = {
        "final_canvas": int(schedule[-1]),
        "initial_canvas": int(schedule[0]),
        "expansion_events": int(expansion_events),
        "model_calls": int(model_calls),
        "steps_per_block": int(steps_per_block),
    }
    return sequence, stats


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
    output_prefix: str = "artifacts/elastic_canvas/expansion_quality"
    schedules: str = "32,64,128,256;64,128,256"
    fixed_canvas: int = 256
    oracle_initial_canvas: int = 32
    oracle_page_size: int = 32
    include_oracle_final_canvas: bool = True
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

if sampler_config.max_new_tokens != script_args.fixed_canvas:
    raise ValueError(
        "--max_new_tokens must match --fixed_canvas for this expansion-quality test."
    )

messages = _load_messages(
    input_path=script_args.input_path,
    prompt=script_args.prompt,
    limit=script_args.limit,
)
schedules = _parse_schedules(script_args.schedules)
for schedule in schedules:
    if schedule[-1] != script_args.fixed_canvas:
        raise ValueError(
            f"Expansion schedule must end at fixed_canvas={script_args.fixed_canvas}: "
            f"{schedule}"
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

    variants = [{"name": "fixed_from_start", "schedule": [script_args.fixed_canvas]}]
    variants.extend(
        {
            "name": "expand_" + "_".join(str(length) for length in schedule),
            "schedule": schedule,
        }
        for schedule in schedules
    )

    fixed_record = None
    request_records = []
    for variant in variants:
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        _sync_device(model.device)
        start_time = time.perf_counter()
        sequence, stats = _sample_visible_canvas_schedule(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            schedule=variant["schedule"],
            config=sampler_config,
            scheduler=sampler.scheduler,
            steps_reference_canvas=script_args.fixed_canvas,
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
            max_new_tokens=stats["final_canvas"],
            length_source=script_args.length_source,
        )
        gen_tokens = sequence[len(prompt_ids) : len(prompt_ids) + stats["final_canvas"]]
        record = {
            "request_index": request_index,
            "variant": variant["name"],
            "schedule": ",".join(str(length) for length in variant["schedule"]),
            "prompt_tokens": len(prompt_ids),
            "final_canvas": stats["final_canvas"],
            "initial_canvas": stats["initial_canvas"],
            "expansion_events": stats["expansion_events"],
            "model_calls": stats["model_calls"],
            "steps_per_block": stats["steps_per_block"],
            "useful_generated_tokens": useful_length,
            "decoded_chars": len(decoded),
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_mb": peak_memory_mb,
            "decoded_text": decoded,
            "generated_token_ids": gen_tokens,
        }
        if variant["name"] == "fixed_from_start":
            fixed_record = record
        request_records.append(record)

    assert fixed_record is not None
    if script_args.include_oracle_final_canvas:
        oracle_canvas = round_canvas_length(
            int(fixed_record["useful_generated_tokens"]),
            initial_canvas=script_args.oracle_initial_canvas,
            page_size=script_args.oracle_page_size,
            max_canvas=script_args.fixed_canvas,
        )
        if oracle_canvas < script_args.fixed_canvas:
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.device)
            _sync_device(model.device)
            start_time = time.perf_counter()
            sequence, stats = _sample_visible_canvas_schedule(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                schedule=[oracle_canvas],
                config=sampler_config,
                scheduler=sampler.scheduler,
                steps_reference_canvas=script_args.fixed_canvas,
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
                max_new_tokens=oracle_canvas,
                length_source=script_args.length_source,
            )
            request_records.append(
                {
                    "request_index": request_index,
                    "variant": f"oracle_canvas_{oracle_canvas}_from_start",
                    "schedule": str(oracle_canvas),
                    "prompt_tokens": len(prompt_ids),
                    "final_canvas": stats["final_canvas"],
                    "initial_canvas": stats["initial_canvas"],
                    "expansion_events": stats["expansion_events"],
                    "model_calls": stats["model_calls"],
                    "steps_per_block": stats["steps_per_block"],
                    "useful_generated_tokens": useful_length,
                    "decoded_chars": len(decoded),
                    "elapsed_seconds": elapsed_seconds,
                    "peak_memory_mb": peak_memory_mb,
                    "decoded_text": decoded,
                    "generated_token_ids": sequence[
                        len(prompt_ids) : len(prompt_ids) + stats["final_canvas"]
                    ],
                }
            )

    fixed_tokens = fixed_record["generated_token_ids"]
    fixed_text = fixed_record["decoded_text"]
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
                        "variant": r["variant"],
                        "useful_tokens": r["useful_generated_tokens"],
                        "similarity": round(r["text_similarity_vs_fixed"], 4),
                        "token_edit_ratio": round(r["token_edit_ratio_vs_fixed"], 4),
                        "elapsed_seconds": round(r["elapsed_seconds"], 3),
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
        "avg_elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in group)
        / len(group),
        "avg_peak_memory_mb": (
            sum(float(record["peak_memory_mb"]) for record in group) / len(group)
            if group[0]["peak_memory_mb"] is not None
            else None
        ),
        "avg_expansion_events": sum(
            int(record["expansion_events"]) for record in group
        )
        / len(group),
        "avg_model_calls": sum(int(record["model_calls"]) for record in group)
        / len(group),
    }

summary = {
    "model_name_or_path": script_args.model_name_or_path,
    "num_requests": len(messages),
    "fixed_canvas": script_args.fixed_canvas,
    "block_size": sampler_config.block_size,
    "steps": sampler_config.steps,
    "steps_reference_canvas": script_args.fixed_canvas,
    "threshold": sampler_config.threshold,
    "factor": sampler_config.factor,
    "length_source": script_args.length_source,
    "note": (
        "Expansion variants reveal only the currently allocated canvas. "
        "Previously generated blocks remain committed, matching the existing "
        "blockwise sampler's no-remasking behavior."
    ),
    "variants": summary_by_variant,
}

prefix = Path(script_args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
summary_path = prefix.with_name(prefix.name + "_summary.json")
records_path = prefix.with_name(prefix.name + "_records.jsonl")
csv_path = prefix.with_name(prefix.name + "_records.csv")
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=True, indent=2)
with records_path.open("w", encoding="utf-8") as f:
    for record in records:
        serializable = dict(record)
        serializable.pop("generated_token_ids", None)
        f.write(json.dumps(serializable, ensure_ascii=True) + "\n")
_write_csv(
    csv_path,
    [{k: v for k, v in record.items() if k != "generated_token_ids"} for record in records],
)

print("Saved summary:", summary_path)
print("Saved records:", records_path)
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
