"""
Test oracle action-priority remasking for structured external actions.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_action_priority_remasking.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/tool_prefetch_prompts_core.jsonl \
    --limit 20 \
    --bias_strengths 0.05,0.10,0.20 \
    --steps 128 \
    --max_new_tokens 128 \
    --block_size 32 \
    --use_cache prefix \
    --threshold 0.9 \
    --output_prefix artifacts/action_completeness/action_priority_core_s128

This is an upper-bound experiment. It first runs the baseline, marks output
positions whose final tokens correspond to the target structured action, then
reruns with an additive transfer-score bias on those positions. If this cannot
pull readiness earlier without hurting final action correctness, action-priority
remasking is not a viable mechanism.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm

from test_tool_prefetch_signals import (
    _decode_generated,
    _false_start_count,
    _first_ready_fraction,
    _format_prompt,
    _history_texts,
    _load_records,
    _normalize_inputs,
    _score_extraction,
    _stable_ready_fraction,
    _compact,
)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _target_compact(record: dict[str, Any]) -> str:
    parts = [str(record["tool"])]
    for key, value in record["args"].items():
        parts.append(str(key))
        parts.append(str(value))
    return _compact(" ".join(parts))


def _oracle_action_positions(
    *,
    tokenizer: Any,
    final_sequence: list[int],
    prompt_len: int,
    max_new_tokens: int,
    record: dict[str, Any],
    min_token_chars: int,
) -> list[int]:
    target = _target_compact(record)
    positions = []
    for offset, token_id in enumerate(final_sequence[prompt_len : prompt_len + max_new_tokens]):
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        compact = _compact(text)
        if len(compact) < min_token_chars:
            continue
        if compact in target:
            positions.append(prompt_len + offset)
    return positions


def _make_transfer_bias(
    *,
    shape: tuple[int, int],
    positions: list[int],
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    bias = torch.zeros(shape, dtype=torch.float32, device=device)
    for position in positions:
        if 0 <= position < shape[1]:
            bias[:, position] = float(strength)
    return bias


def _run_sample(
    *,
    sampler: Any,
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    sampler_config: Any,
    transfer_bias: torch.Tensor | None,
) -> tuple[Any, list[int], float]:
    inputs = tokenizer.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = _normalize_inputs(inputs)[0]
    _sync_device(model.device)
    start = time.perf_counter()
    outputs = sampler.sample(
        inputs,
        config=sampler_config,
        return_dict=True,
        transfer_bias=transfer_bias,
    )
    _sync_device(model.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if outputs.histories is None:
        raise RuntimeError("Sampler did not return histories.")
    return outputs, prompt_ids, elapsed_ms


def _analyze_outputs(
    *,
    tokenizer: Any,
    outputs: Any,
    prompt_len: int,
    max_new_tokens: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    target_tool = str(record["tool"])
    target_args = {str(key): str(value) for key, value in record["args"].items()}
    final_sequence = outputs.sequences[0].tolist()
    final_text = _decode_generated(
        tokenizer=tokenizer,
        sequence=final_sequence,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    texts = _history_texts(
        tokenizer=tokenizer,
        histories=outputs.histories,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    first_step, first_fraction, first_score = _first_ready_fraction(
        texts=texts,
        target_tool=target_tool,
        target_args=target_args,
    )
    stable_step, stable_fraction = _stable_ready_fraction(
        texts=texts,
        target_tool=target_tool,
        target_args=target_args,
    )
    final_score = _score_extraction(final_text, target_tool, target_args)
    false_starts = _false_start_count(
        texts=texts,
        target_tool=target_tool,
        target_args=target_args,
    )
    return {
        "final_text": final_text,
        "final_ready": final_score["ready"],
        "final_detected_tool": final_score["detected_tool"],
        "ready_step": first_step,
        "ready_fraction": first_fraction,
        "stable_step": stable_step,
        "stable_fraction": stable_fraction,
        "false_starts": false_starts,
        "ready_score": first_score,
        "num_history_steps": len(texts),
        "final_sequence": final_sequence,
    }


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
    input_path: str = "examples/fastdllm/llada/tool_prefetch_prompts_core.jsonl"
    limit: int = 20
    bias_strengths: str = "0.05,0.10,0.20"
    min_token_chars: int = 2
    seed: int = 42
    output_prefix: str = "artifacts/action_completeness/action_priority_core_s128"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path,
            "BASE_MODELS_DIR",
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


def main() -> None:
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)
    bias_strengths = _parse_float_list(script_args.bias_strengths)
    records = _load_records(script_args.input_path, script_args.limit)

    fastdllm_config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
        script_args.model_name_or_path
    )
    model = dllm.utils.get_model(model_args=script_args, config=fastdllm_config).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.pipelines.fastdllm.llada.FastdLLMLLaDASampler(
        model=model,
        tokenizer=tokenizer,
    )

    request_rows = []
    decoded_rows = []
    for request_index, record in enumerate(records):
        messages = _format_prompt(record)
        baseline_outputs, prompt_ids, baseline_ms = _run_sample(
            sampler=sampler,
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            sampler_config=sampler_config,
            transfer_bias=None,
        )
        baseline = _analyze_outputs(
            tokenizer=tokenizer,
            outputs=baseline_outputs,
            prompt_len=len(prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
            record=record,
        )
        positions = _oracle_action_positions(
            tokenizer=tokenizer,
            final_sequence=baseline["final_sequence"],
            prompt_len=len(prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
            record=record,
            min_token_chars=script_args.min_token_chars,
        )
        base_row = {
            "request_index": request_index,
            "policy": "baseline",
            "bias_strength": 0.0,
            "target_tool": record["tool"],
            "elapsed_ms": baseline_ms,
            "action_position_count": len(positions),
            "final_ready": baseline["final_ready"],
            "ready_fraction": baseline["ready_fraction"],
            "stable_fraction": baseline["stable_fraction"],
            "false_starts": baseline["false_starts"],
            "quality_preserved": True,
            "ready_delta_vs_baseline": 0.0,
        }
        request_rows.append(base_row)
        decoded_rows.append(
            {
                "request_index": request_index,
                "policy": "baseline",
                "target_tool": record["tool"],
                "prompt": record["prompt"],
                "final_text": baseline["final_text"],
                "action_positions": positions,
            }
        )
        for strength in bias_strengths:
            bias = _make_transfer_bias(
                shape=baseline_outputs.sequences.shape,
                positions=positions,
                strength=strength,
                device=model.device,
            )
            steered_outputs, _, steered_ms = _run_sample(
                sampler=sampler,
                tokenizer=tokenizer,
                model=model,
                messages=messages,
                sampler_config=sampler_config,
                transfer_bias=bias,
            )
            steered = _analyze_outputs(
                tokenizer=tokenizer,
                outputs=steered_outputs,
                prompt_len=len(prompt_ids),
                max_new_tokens=sampler_config.max_new_tokens,
                record=record,
            )
            baseline_fraction = baseline["stable_fraction"]
            steered_fraction = steered["stable_fraction"]
            ready_delta = (
                float(baseline_fraction) - float(steered_fraction)
                if baseline_fraction is not None and steered_fraction is not None
                else None
            )
            request_rows.append(
                {
                    "request_index": request_index,
                    "policy": "action_priority",
                    "bias_strength": strength,
                    "target_tool": record["tool"],
                    "elapsed_ms": steered_ms,
                    "action_position_count": len(positions),
                    "final_ready": steered["final_ready"],
                    "ready_fraction": steered["ready_fraction"],
                    "stable_fraction": steered["stable_fraction"],
                    "false_starts": steered["false_starts"],
                    "quality_preserved": bool(steered["final_ready"]),
                    "ready_delta_vs_baseline": ready_delta,
                }
            )
            decoded_rows.append(
                {
                    "request_index": request_index,
                    "policy": f"action_priority_{strength}",
                    "target_tool": record["tool"],
                    "prompt": record["prompt"],
                    "final_text": steered["final_text"],
                    "action_positions": positions,
                }
            )
        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "baseline_stable": baseline["stable_fraction"],
                    "action_positions": len(positions),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    aggregate: dict[str, Any] = {
        "num_requests": len(records),
        "bias_strengths": bias_strengths,
        "steps": sampler_config.steps,
        "max_new_tokens": sampler_config.max_new_tokens,
        "threshold": sampler_config.threshold,
        "use_cache": sampler_config.use_cache,
    }
    for strength in [0.0] + bias_strengths:
        rows = [
            row
            for row in request_rows
            if abs(float(row["bias_strength"]) - float(strength)) < 1e-9
        ]
        if not rows:
            continue
        prefix = "baseline" if strength == 0.0 else f"bias_{strength:g}"
        valid_ready = [
            float(row["stable_fraction"])
            for row in rows
            if row["stable_fraction"] is not None
        ]
        deltas = [
            float(row["ready_delta_vs_baseline"])
            for row in rows
            if row["ready_delta_vs_baseline"] is not None
        ]
        aggregate[f"{prefix}_final_ready_rate"] = (
            sum(1 for row in rows if row["final_ready"]) / len(rows)
        )
        aggregate[f"{prefix}_mean_stable_fraction"] = (
            statistics.mean(valid_ready) if valid_ready else None
        )
        aggregate[f"{prefix}_p50_stable_fraction"] = (
            _percentile(valid_ready, 0.50) if valid_ready else None
        )
        aggregate[f"{prefix}_mean_ready_delta"] = (
            statistics.mean(deltas) if deltas else None
        )
        aggregate[f"{prefix}_quality_preserved_rate"] = (
            sum(1 for row in rows if row["quality_preserved"]) / len(rows)
        )
        aggregate[f"{prefix}_mean_false_starts"] = statistics.mean(
            [float(row["false_starts"]) for row in rows]
        )

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    decoded_path = prefix.with_name(prefix.name + "_decoded.jsonl")
    summary_path.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "requests": request_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(requests_path, request_rows)
    with decoded_path.open("w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Saved summary: {summary_path}")
    print(f"Saved request CSV: {requests_path}")
    print(f"Saved decoded JSONL: {decoded_path}")
    print("Aggregate:")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
