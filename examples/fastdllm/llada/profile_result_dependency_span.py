"""
Profile the upper bound for selective remasking after an in-flight tool result.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/profile_result_dependency_span.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/inflight_binding_prompts.jsonl \
    --limit 10 \
    --tool_latencies_ms 100,300,500,1000,2000 \
    --steps 128 \
    --max_new_tokens 160 \
    --block_size 32 \
    --use_cache none \
    --threshold 0.9 \
    --output_prefix artifacts/action_completeness/result_dependency_core_s128

This script compares two full denoising runs for the same structured action:
one where the external result is unavailable and one where it is available.
It then estimates the oracle downstream span that must change after binding
the result. This is a go/no-go upper bound for replacing re-prefill/restart
with RESULT binding plus selective ANSWER remasking.
"""

from __future__ import annotations

import csv
import difflib
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

from test_inflight_result_binding import _format_prompt
from test_tool_prefetch_signals import (
    _contains_value,
    _decode_generated,
    _first_ready_fraction,
    _history_texts,
    _load_records,
    _normalize_inputs,
    _score_extraction,
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


def _section_after_label(text: str, label: str) -> tuple[int, int] | None:
    lowered = text.lower()
    marker = label.lower()
    start = lowered.find(marker)
    if start < 0:
        return None
    return start, start + len(marker)


def _extract_sections(text: str) -> dict[str, str]:
    result = _section_after_label(text, "RESULT:")
    answer = _section_after_label(text, "ANSWER:")
    if result is None:
        return {
            "prefix": text,
            "result": "",
            "answer": "",
            "has_result_label": False,
            "has_answer_label": answer is not None,
        }
    result_label_start, result_value_start = result
    if answer is None or answer[0] < result_value_start:
        return {
            "prefix": text[:result_label_start],
            "result": text[result_value_start:],
            "answer": "",
            "has_result_label": True,
            "has_answer_label": False,
        }
    answer_label_start, answer_value_start = answer
    return {
        "prefix": text[:result_label_start],
        "result": text[result_value_start:answer_label_start],
        "answer": text[answer_value_start:],
        "has_result_label": True,
        "has_answer_label": True,
    }


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _lcs_len(left: list[int], right: list[int]) -> int:
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def _diff_stats(tokenizer: Any, left_text: str, right_text: str) -> dict[str, Any]:
    left = _token_ids(tokenizer, left_text)
    right = _token_ids(tokenizer, right_text)
    lcs = _lcs_len(left, right)
    right_changed = max(0, len(right) - lcs)
    left_changed = max(0, len(left) - lcs)
    return {
        "left_tokens": len(left),
        "right_tokens": len(right),
        "lcs_tokens": lcs,
        "right_changed_tokens": right_changed,
        "left_changed_tokens": left_changed,
        "right_changed_fraction": right_changed / max(len(right), 1),
        "left_changed_fraction": left_changed / max(len(left), 1),
    }


def _run_sample(
    *,
    sampler: Any,
    tokenizer: Any,
    model: Any,
    record: dict[str, Any],
    sampler_config: Any,
    result_available: bool,
) -> tuple[Any, list[int], float]:
    inputs = tokenizer.apply_chat_template(
        [_format_prompt(record, result_available=result_available)],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = _normalize_inputs(inputs)[0]
    _sync_device(model.device)
    start = time.perf_counter()
    outputs = sampler.sample(inputs, config=sampler_config, return_dict=True)
    _sync_device(model.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return outputs, prompt_ids, elapsed_ms


def _analyze_final(
    *,
    tokenizer: Any,
    outputs: Any,
    prompt_len: int,
    max_new_tokens: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    final_text = _decode_generated(
        tokenizer=tokenizer,
        sequence=outputs.sequences[0].tolist(),
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )
    target_tool = str(record["tool"])
    target_args = {str(key): str(value) for key, value in record["args"].items()}
    action_score = _score_extraction(final_text, target_tool, target_args)
    answer_terms = [str(term) for term in record.get("answer_terms", [])]
    result_terms_present = all(_contains_value(final_text, term) for term in answer_terms)
    result_present = _contains_value(final_text, str(record["result"]))
    return {
        "final_text": final_text,
        "action_ready": action_score["ready"],
        "result_present": result_present,
        "result_terms_present": result_terms_present,
        "success": bool(action_score["ready"] and result_terms_present),
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
    input_path: str = "examples/fastdllm/llada/inflight_binding_prompts.jsonl"
    limit: int = 10
    tool_latencies_ms: str = "100,300,500,1000,2000"
    seed: int = 42
    output_prefix: str = "artifacts/action_completeness/result_dependency_core_s128"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path,
            "BASE_MODELS_DIR",
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    steps: int = 128
    max_new_tokens: int = 160
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "none"
    threshold: float | None = 0.9
    factor: float | None = None
    begin_suppress_tokens: list[int] | None = None


def main() -> None:
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)
    tool_latencies_ms = _parse_float_list(script_args.tool_latencies_ms)
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

    request_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []

    for request_index, record in enumerate(records):
        no_outputs, no_prompt_ids, no_ms = _run_sample(
            sampler=sampler,
            tokenizer=tokenizer,
            model=model,
            record=record,
            sampler_config=sampler_config,
            result_available=False,
        )
        result_outputs, result_prompt_ids, result_ms = _run_sample(
            sampler=sampler,
            tokenizer=tokenizer,
            model=model,
            record=record,
            sampler_config=sampler_config,
            result_available=True,
        )
        if no_outputs.histories is None:
            raise RuntimeError("Sampler did not return histories.")

        no_final = _analyze_final(
            tokenizer=tokenizer,
            outputs=no_outputs,
            prompt_len=len(no_prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
            record=record,
        )
        result_final = _analyze_final(
            tokenizer=tokenizer,
            outputs=result_outputs,
            prompt_len=len(result_prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
            record=record,
        )
        no_sections = _extract_sections(no_final["final_text"])
        result_sections = _extract_sections(result_final["final_text"])

        prefix_diff = _diff_stats(
            tokenizer,
            str(no_sections["prefix"]),
            str(result_sections["prefix"]),
        )
        result_diff = _diff_stats(
            tokenizer,
            str(no_sections["result"]),
            str(result_sections["result"]),
        )
        answer_diff = _diff_stats(
            tokenizer,
            str(no_sections["answer"]),
            str(result_sections["answer"]),
        )
        full_diff = _diff_stats(
            tokenizer,
            no_final["final_text"],
            result_final["final_text"],
        )

        result_tokens = result_diff["right_tokens"]
        answer_tokens = answer_diff["right_tokens"]
        changed_answer_tokens = answer_diff["right_changed_tokens"]
        generated_tokens = max(len(_token_ids(tokenizer, result_final["final_text"])), 1)
        result_answer_tokens = max(result_tokens + answer_tokens, 1)
        oracle_remask_tokens = result_tokens + changed_answer_tokens
        oracle_remask_fraction_of_output = oracle_remask_tokens / generated_tokens
        oracle_remask_fraction_of_result_answer = (
            oracle_remask_tokens / result_answer_tokens
        )

        target_tool = str(record["tool"])
        target_args = {str(key): str(value) for key, value in record["args"].items()}
        no_texts = _history_texts(
            tokenizer=tokenizer,
            histories=no_outputs.histories,
            prompt_len=len(no_prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
        )
        action_ready_step, action_ready_fraction, _ = _first_ready_fraction(
            texts=no_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        action_ready_ms = (
            float(no_ms) * float(action_ready_fraction)
            if action_ready_fraction is not None
            else None
        )

        request_row = {
            "request_index": request_index,
            "target_tool": record["tool"],
            "no_result_elapsed_ms": no_ms,
            "with_result_elapsed_ms": result_ms,
            "no_result_action_ready": no_final["action_ready"],
            "with_result_success": result_final["success"],
            "action_ready_step": action_ready_step,
            "action_ready_fraction": action_ready_fraction,
            "estimated_action_ready_ms": action_ready_ms,
            "has_no_result_label": no_sections["has_result_label"],
            "has_no_answer_label": no_sections["has_answer_label"],
            "has_with_result_label": result_sections["has_result_label"],
            "has_with_answer_label": result_sections["has_answer_label"],
            "prefix_right_changed_fraction": prefix_diff["right_changed_fraction"],
            "result_right_changed_fraction": result_diff["right_changed_fraction"],
            "answer_right_changed_fraction": answer_diff["right_changed_fraction"],
            "full_right_changed_fraction": full_diff["right_changed_fraction"],
            "result_tokens": result_tokens,
            "answer_tokens": answer_tokens,
            "changed_answer_tokens": changed_answer_tokens,
            "oracle_remask_tokens": oracle_remask_tokens,
            "generated_tokens": generated_tokens,
            "oracle_remask_fraction_of_output": oracle_remask_fraction_of_output,
            "oracle_remask_fraction_of_result_answer": (
                oracle_remask_fraction_of_result_answer
            ),
        }
        request_rows.append(request_row)
        decoded_rows.append(
            {
                "request_index": request_index,
                "target_tool": record["tool"],
                "prompt": record["prompt"],
                "result": record["result"],
                "no_result_text": no_final["final_text"],
                "with_result_text": result_final["final_text"],
                "no_result_sections": no_sections,
                "with_result_sections": result_sections,
            }
        )

        for tool_latency_ms in tool_latencies_ms:
            if action_ready_ms is None:
                continue
            sequential_restart_ms = (
                float(action_ready_ms) + float(tool_latency_ms) + float(result_ms)
            )
            tool_ready_ms = float(action_ready_ms) + float(tool_latency_ms)
            remask_cost_ms = float(result_ms) * float(oracle_remask_fraction_of_output)
            selective_remask_upper_ms = max(float(no_ms), tool_ready_ms) + remask_cost_ms
            latency_rows.append(
                {
                    "request_index": request_index,
                    "target_tool": record["tool"],
                    "tool_latency_ms": tool_latency_ms,
                    "action_ready_ms": action_ready_ms,
                    "tool_ready_ms": tool_ready_ms,
                    "no_result_elapsed_ms": no_ms,
                    "with_result_elapsed_ms": result_ms,
                    "oracle_remask_fraction_of_output": (
                        oracle_remask_fraction_of_output
                    ),
                    "oracle_remask_fraction_of_result_answer": (
                        oracle_remask_fraction_of_result_answer
                    ),
                    "remask_cost_ms": remask_cost_ms,
                    "sequential_restart_ms": sequential_restart_ms,
                    "selective_remask_upper_bound_ms": selective_remask_upper_ms,
                    "saved_ms": sequential_restart_ms - selective_remask_upper_ms,
                    "speedup": (
                        sequential_restart_ms / selective_remask_upper_ms
                        if selective_remask_upper_ms > 0
                        else None
                    ),
                    "with_result_success": result_final["success"],
                }
            )

        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "tool": record["tool"],
                    "with_result_success": result_final["success"],
                    "answer_changed_fraction": answer_diff["right_changed_fraction"],
                    "oracle_remask_fraction_of_output": (
                        oracle_remask_fraction_of_output
                    ),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    aggregate: dict[str, Any] = {
        "num_requests": len(request_rows),
        "steps": sampler_config.steps,
        "max_new_tokens": sampler_config.max_new_tokens,
        "use_cache": sampler_config.use_cache,
        "threshold": sampler_config.threshold,
        "tool_latencies_ms": tool_latencies_ms,
        "with_result_success_rate": (
            sum(row["with_result_success"] for row in request_rows)
            / max(len(request_rows), 1)
        ),
    }
    numeric_keys = [
        "no_result_elapsed_ms",
        "with_result_elapsed_ms",
        "action_ready_fraction",
        "answer_right_changed_fraction",
        "full_right_changed_fraction",
        "oracle_remask_fraction_of_output",
        "oracle_remask_fraction_of_result_answer",
    ]
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in request_rows
            if row.get(key) is not None
        ]
        if values:
            aggregate[f"mean_{key}"] = statistics.mean(values)
            aggregate[f"p95_{key}"] = _percentile(values, 0.95)

    for tool_latency_ms in tool_latencies_ms:
        rows = [
            row
            for row in latency_rows
            if float(row["tool_latency_ms"]) == float(tool_latency_ms)
        ]
        if not rows:
            continue
        aggregate[f"tool{tool_latency_ms:g}_mean_sequential_restart_ms"] = (
            statistics.mean([float(row["sequential_restart_ms"]) for row in rows])
        )
        aggregate[f"tool{tool_latency_ms:g}_mean_selective_remask_upper_ms"] = (
            statistics.mean(
                [float(row["selective_remask_upper_bound_ms"]) for row in rows]
            )
        )
        aggregate[f"tool{tool_latency_ms:g}_mean_saved_ms"] = statistics.mean(
            [float(row["saved_ms"]) for row in rows]
        )
        aggregate[f"tool{tool_latency_ms:g}_mean_speedup"] = statistics.mean(
            [float(row["speedup"]) for row in rows if row["speedup"] is not None]
        )

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    latency_path = prefix.with_name(prefix.name + "_latency_model.csv")
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
    _write_csv(latency_path, latency_rows)
    with decoded_path.open("w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Saved summary: {summary_path}")
    print(f"Saved request CSV: {requests_path}")
    print(f"Saved latency model CSV: {latency_path}")
    print(f"Saved decoded JSONL: {decoded_path}")
    print("Aggregate:")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
