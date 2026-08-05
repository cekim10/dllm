"""
Test in-flight result binding into a dLLM denoising canvas.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_inflight_result_binding.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/inflight_binding_prompts.jsonl \
    --limit 10 \
    --bind_fractions 0.25,0.50,0.75 \
    --steps 128 \
    --max_new_tokens 160 \
    --block_size 32 \
    --use_cache none \
    --threshold 0.9 \
    --output_prefix artifacts/action_completeness/inflight_binding_core_s128

This is an upper-bound mechanism test. The runtime knows the intended RESULT
span layout and clamps external result tokens into that span while denoising
continues. It also tests a reserved-slot policy where the runtime owns the
RESULT span from the beginning, keeps it masked until the external result
arrives, then binds the result while the ANSWER span continues denoising.
Start with use_cache=none so later model calls always see the bound tokens;
cached modes require cache invalidation or refresh after binding.
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


def _format_args(args: dict[str, Any]) -> str:
    return "; ".join(f"{key}={value}" for key, value in args.items())


def _target_output(record: dict[str, Any]) -> tuple[str, str, str]:
    args_text = _format_args(record["args"])
    result = str(record["result"])
    prefix = (
        f"TOOL: {record['tool']}\n"
        f"ARGS: {args_text}\n"
        "RESULT: "
    )
    suffix = "\nANSWER: "
    return prefix + result + suffix, prefix, result


def _format_prompt(
    record: dict[str, Any],
    *,
    result_available: bool,
) -> list[dict[str, str]]:
    available_tools = [
        "- weather(location, date)",
        "- calendar_api(calendar, start_date, end_date, keyword)",
        "- crm_api(company, role, city)",
    ]
    arg_names = ", ".join(str(key) for key in record["args"])
    if not any(str(record["tool"]) in line for line in available_tools):
        available_tools.append(f"- {record['tool']}({arg_names})")
    if result_available:
        result_line = f"External result: {record['result']}\nUse the external result exactly.\n"
    else:
        result_line = (
            "The external result is not available at generation start. Do not invent it.\n"
        )
    content = (
        "You are a structured action runtime test.\n"
        "Return exactly this format:\n"
        "TOOL: <tool_name>\n"
        "ARGS: key=value; key=value\n"
        "RESULT: <external result if it is available, otherwise pending>\n"
        "ANSWER: <one sentence using the RESULT>\n\n"
        f"{result_line}"
        f"Available tools:\n" + "\n".join(available_tools) + "\n\n"
        f"User request: {record['prompt']}\n"
        f"Required tool: {record['tool']}({arg_names})"
    )
    return [{"role": "user", "content": content}]


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _result_binding_plan(
    *,
    tokenizer: Any,
    prompt_len: int,
    record: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    _, prefix, result = _target_output(record)
    suffix = "\nANSWER: "
    prefix_tokens = _encode_no_special(tokenizer, prefix)
    result_tokens = _encode_no_special(tokenizer, result)
    suffix_tokens = _encode_no_special(tokenizer, suffix)
    start = prompt_len + len(prefix_tokens)
    positions = [
        start + offset
        for offset in range(len(result_tokens))
        if start + offset < prompt_len + max_new_tokens
    ]
    token_ids = result_tokens[: len(positions)]
    prefix_positions = [
        prompt_len + offset
        for offset in range(len(prefix_tokens))
        if prompt_len + offset < prompt_len + max_new_tokens
    ]
    prefix_token_ids = prefix_tokens[: len(prefix_positions)]
    suffix_start = start + len(result_tokens)
    suffix_positions = [
        suffix_start + offset
        for offset in range(len(suffix_tokens))
        if suffix_start + offset < prompt_len + max_new_tokens
    ]
    suffix_token_ids = suffix_tokens[: len(suffix_positions)]
    return {
        "result": result,
        "positions": positions,
        "token_ids": token_ids,
        "prefix_positions": prefix_positions,
        "prefix_token_ids": prefix_token_ids,
        "suffix_positions": suffix_positions,
        "suffix_token_ids": suffix_token_ids,
        "prefix_token_count": len(prefix_tokens),
        "result_token_count": len(result_tokens),
        "suffix_token_count": len(suffix_tokens),
    }


def _make_binding_hook(
    *,
    positions: list[int],
    token_ids: list[int],
    bind_update_index: int,
) -> Any:
    position_tensor = None
    token_tensor = None

    def hook(x: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        nonlocal position_tensor, token_tensor
        if int(context["canvas_update_index"]) < bind_update_index:
            return x
        if not positions:
            return x
        if position_tensor is None or position_tensor.device != x.device:
            position_tensor = torch.tensor(positions, dtype=torch.long, device=x.device)
            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=x.device)
        valid = position_tensor < x.shape[1]
        if valid.any():
            x = x.clone()
            x[:, position_tensor[valid]] = token_tensor[valid]
        return x

    return hook


def _make_reserved_slot_hook(
    *,
    mask_id: int,
    prefix_positions: list[int],
    prefix_token_ids: list[int],
    slot_positions: list[int],
    slot_token_ids: list[int],
    suffix_positions: list[int],
    suffix_token_ids: list[int],
    bind_update_index: int,
) -> Any:
    prefix_position_tensor = None
    prefix_token_tensor = None
    slot_position_tensor = None
    slot_token_tensor = None
    suffix_position_tensor = None
    suffix_token_tensor = None

    def _ensure_tensors(x: torch.Tensor) -> None:
        nonlocal prefix_position_tensor, prefix_token_tensor
        nonlocal slot_position_tensor, slot_token_tensor
        nonlocal suffix_position_tensor, suffix_token_tensor
        if prefix_position_tensor is not None and prefix_position_tensor.device == x.device:
            return
        prefix_position_tensor = torch.tensor(
            prefix_positions, dtype=torch.long, device=x.device
        )
        prefix_token_tensor = torch.tensor(
            prefix_token_ids, dtype=torch.long, device=x.device
        )
        slot_position_tensor = torch.tensor(
            slot_positions, dtype=torch.long, device=x.device
        )
        slot_token_tensor = torch.tensor(
            slot_token_ids, dtype=torch.long, device=x.device
        )
        suffix_position_tensor = torch.tensor(
            suffix_positions, dtype=torch.long, device=x.device
        )
        suffix_token_tensor = torch.tensor(
            suffix_token_ids, dtype=torch.long, device=x.device
        )

    def _clamp(
        x: torch.Tensor,
        positions: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        if positions.numel() == 0:
            return
        valid = positions < x.shape[1]
        if valid.any():
            x[:, positions[valid]] = token_ids[valid]

    def hook(x: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        _ensure_tensors(x)
        x = x.clone()
        assert prefix_position_tensor is not None
        assert prefix_token_tensor is not None
        assert slot_position_tensor is not None
        assert slot_token_tensor is not None
        assert suffix_position_tensor is not None
        assert suffix_token_tensor is not None
        _clamp(x, prefix_position_tensor, prefix_token_tensor)
        _clamp(x, suffix_position_tensor, suffix_token_tensor)
        if slot_position_tensor.numel() > 0:
            valid = slot_position_tensor < x.shape[1]
            if valid.any():
                if int(context["canvas_update_index"]) < bind_update_index:
                    x[:, slot_position_tensor[valid]] = int(mask_id)
                else:
                    x[:, slot_position_tensor[valid]] = slot_token_tensor[valid]
        return x

    return hook


def _run_sample(
    *,
    sampler: Any,
    tokenizer: Any,
    model: Any,
    record: dict[str, Any],
    sampler_config: Any,
    canvas_update_hook: Any | None,
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
    outputs = sampler.sample(
        inputs,
        config=sampler_config,
        return_dict=True,
        canvas_update_hook=canvas_update_hook,
    )
    _sync_device(model.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return outputs, prompt_ids, elapsed_ms


def _analyze(
    *,
    tokenizer: Any,
    outputs: Any,
    prompt_len: int,
    max_new_tokens: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    final_sequence = outputs.sequences[0].tolist()
    final_text = _decode_generated(
        tokenizer=tokenizer,
        sequence=final_sequence,
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
    bind_fractions: str = "0.25,0.50,0.75"
    tool_latencies_ms: str = "100,300,500"
    seed: int = 42
    output_prefix: str = "artifacts/action_completeness/inflight_binding_core_s128"

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
    bind_fractions = _parse_float_list(script_args.bind_fractions)
    tool_latencies_ms = _parse_float_list(script_args.tool_latencies_ms)
    records = _load_records(script_args.input_path, script_args.limit)

    fastdllm_config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
        script_args.model_name_or_path
    )
    model = dllm.utils.get_model(model_args=script_args, config=fastdllm_config).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer does not define mask_token_id.")
    sampler = dllm.pipelines.fastdllm.llada.FastdLLMLLaDASampler(
        model=model,
        tokenizer=tokenizer,
    )

    request_rows = []
    decoded_rows = []
    latency_rows = []
    for request_index, record in enumerate(records):
        baseline_outputs, prompt_ids, baseline_ms = _run_sample(
            sampler=sampler,
            tokenizer=tokenizer,
            model=model,
            record=record,
            sampler_config=sampler_config,
            canvas_update_hook=None,
            result_available=False,
        )
        if baseline_outputs.histories is None:
            raise RuntimeError("Sampler did not return histories.")
        baseline = _analyze(
            tokenizer=tokenizer,
            outputs=baseline_outputs,
            prompt_len=len(prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
            record=record,
        )
        target_tool = str(record["tool"])
        target_args = {str(key): str(value) for key, value in record["args"].items()}
        baseline_texts = _history_texts(
            tokenizer=tokenizer,
            histories=baseline_outputs.histories,
            prompt_len=len(prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
        )
        action_ready_step, action_ready_fraction, _ = _first_ready_fraction(
            texts=baseline_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        action_ready_ms = (
            float(baseline_ms) * float(action_ready_fraction)
            if action_ready_fraction is not None
            else None
        )
        plan = _result_binding_plan(
            tokenizer=tokenizer,
            prompt_len=len(prompt_ids),
            record=record,
            max_new_tokens=sampler_config.max_new_tokens,
        )
        request_rows.append(
            {
                "request_index": request_index,
                "policy": "no_bind",
                "bind_fraction": None,
                "bind_update_index": None,
                "target_tool": record["tool"],
                "elapsed_ms": baseline_ms,
                "action_ready_step": action_ready_step,
                "action_ready_fraction": action_ready_fraction,
                "estimated_action_ready_ms": action_ready_ms,
                "result_token_count": plan["result_token_count"],
                "bound_token_count": len(plan["positions"]),
                "action_ready": baseline["action_ready"],
                "result_present": baseline["result_present"],
                "result_terms_present": baseline["result_terms_present"],
                "success": baseline["success"],
            }
        )
        decoded_rows.append(
            {
                "request_index": request_index,
                "policy": "no_bind",
                "prompt": record["prompt"],
                "target_tool": record["tool"],
                "result": record["result"],
                "final_text": baseline["final_text"],
            }
        )

        restart_outputs, restart_prompt_ids, restart_ms = _run_sample(
            sampler=sampler,
            tokenizer=tokenizer,
            model=model,
            record=record,
            sampler_config=sampler_config,
            canvas_update_hook=None,
            result_available=True,
        )
        restart = _analyze(
            tokenizer=tokenizer,
            outputs=restart_outputs,
            prompt_len=len(restart_prompt_ids),
            max_new_tokens=sampler_config.max_new_tokens,
            record=record,
        )
        request_rows.append(
            {
                "request_index": request_index,
                "policy": "restart_with_result",
                "bind_fraction": None,
                "bind_update_index": None,
                "target_tool": record["tool"],
                "elapsed_ms": restart_ms,
                "action_ready_step": None,
                "action_ready_fraction": None,
                "estimated_action_ready_ms": None,
                "result_token_count": plan["result_token_count"],
                "bound_token_count": 0,
                "action_ready": restart["action_ready"],
                "result_present": restart["result_present"],
                "result_terms_present": restart["result_terms_present"],
                "success": restart["success"],
            }
        )
        decoded_rows.append(
            {
                "request_index": request_index,
                "policy": "restart_with_result",
                "prompt": record["prompt"],
                "target_tool": record["tool"],
                "result": record["result"],
                "final_text": restart["final_text"],
            }
        )

        for fraction in bind_fractions:
            bind_update_index = max(0, int(math.floor(fraction * sampler_config.steps)))
            hook = _make_binding_hook(
                positions=plan["positions"],
                token_ids=plan["token_ids"],
                bind_update_index=bind_update_index,
            )
            bound_outputs, _, bound_ms = _run_sample(
                sampler=sampler,
                tokenizer=tokenizer,
                model=model,
                record=record,
                sampler_config=sampler_config,
                canvas_update_hook=hook,
                result_available=False,
            )
            bound = _analyze(
                tokenizer=tokenizer,
                outputs=bound_outputs,
                prompt_len=len(prompt_ids),
                max_new_tokens=sampler_config.max_new_tokens,
                record=record,
            )
            request_rows.append(
                {
                    "request_index": request_index,
                    "policy": "bind",
                    "bind_fraction": fraction,
                    "bind_update_index": bind_update_index,
                    "target_tool": record["tool"],
                    "elapsed_ms": bound_ms,
                    "action_ready_step": action_ready_step,
                    "action_ready_fraction": action_ready_fraction,
                    "estimated_action_ready_ms": action_ready_ms,
                    "result_token_count": plan["result_token_count"],
                    "bound_token_count": len(plan["positions"]),
                    "action_ready": bound["action_ready"],
                    "result_present": bound["result_present"],
                    "result_terms_present": bound["result_terms_present"],
                    "success": bound["success"],
                }
            )
            decoded_rows.append(
                {
                    "request_index": request_index,
                    "policy": f"bind_{fraction}",
                    "prompt": record["prompt"],
                    "target_tool": record["tool"],
                    "result": record["result"],
                    "final_text": bound["final_text"],
                }
            )
            for tool_latency_ms in tool_latencies_ms:
                if action_ready_ms is None:
                    continue
                sequential_ms = float(action_ready_ms) + float(tool_latency_ms) + float(
                    restart_ms
                )
                tool_result_ready_ms = float(action_ready_ms) + float(tool_latency_ms)
                estimated_bind_ms = float(baseline_ms) * float(fraction)
                optimistic_bind_ms = max(
                    float(bound_ms),
                    tool_result_ready_ms,
                )
                latency_rows.append(
                    {
                        "policy": "bind",
                        "request_index": request_index,
                        "bind_fraction": fraction,
                        "tool_latency_ms": tool_latency_ms,
                        "action_ready_ms": action_ready_ms,
                        "tool_result_ready_ms": tool_result_ready_ms,
                        "estimated_bind_ms": estimated_bind_ms,
                        "bind_not_before_tool_ready": (
                            estimated_bind_ms >= tool_result_ready_ms
                        ),
                        "no_bind_elapsed_ms": baseline_ms,
                        "restart_with_result_ms": restart_ms,
                        "bind_elapsed_ms": bound_ms,
                        "sequential_restart_path_ms": sequential_ms,
                        "inflight_binding_upper_bound_ms": optimistic_bind_ms,
                        "saved_ms": sequential_ms - optimistic_bind_ms,
                        "speedup": (
                            sequential_ms / optimistic_bind_ms
                            if optimistic_bind_ms > 0
                            else None
                        ),
                        "bind_success": bound["success"],
                        "restart_success": restart["success"],
                    }
                )

            reserved_hook = _make_reserved_slot_hook(
                mask_id=int(tokenizer.mask_token_id),
                prefix_positions=plan["prefix_positions"],
                prefix_token_ids=plan["prefix_token_ids"],
                slot_positions=plan["positions"],
                slot_token_ids=plan["token_ids"],
                suffix_positions=plan["suffix_positions"],
                suffix_token_ids=plan["suffix_token_ids"],
                bind_update_index=bind_update_index,
            )
            reserved_outputs, _, reserved_ms = _run_sample(
                sampler=sampler,
                tokenizer=tokenizer,
                model=model,
                record=record,
                sampler_config=sampler_config,
                canvas_update_hook=reserved_hook,
                result_available=False,
            )
            reserved = _analyze(
                tokenizer=tokenizer,
                outputs=reserved_outputs,
                prompt_len=len(prompt_ids),
                max_new_tokens=sampler_config.max_new_tokens,
                record=record,
            )
            request_rows.append(
                {
                    "request_index": request_index,
                    "policy": "reserved_slot",
                    "bind_fraction": fraction,
                    "bind_update_index": bind_update_index,
                    "target_tool": record["tool"],
                    "elapsed_ms": reserved_ms,
                    "action_ready_step": action_ready_step,
                    "action_ready_fraction": action_ready_fraction,
                    "estimated_action_ready_ms": action_ready_ms,
                    "result_token_count": plan["result_token_count"],
                    "bound_token_count": len(plan["positions"]),
                    "reserved_prefix_token_count": len(plan["prefix_positions"]),
                    "reserved_suffix_token_count": len(plan["suffix_positions"]),
                    "action_ready": reserved["action_ready"],
                    "result_present": reserved["result_present"],
                    "result_terms_present": reserved["result_terms_present"],
                    "success": reserved["success"],
                }
            )
            decoded_rows.append(
                {
                    "request_index": request_index,
                    "policy": f"reserved_slot_{fraction}",
                    "prompt": record["prompt"],
                    "target_tool": record["tool"],
                    "result": record["result"],
                    "final_text": reserved["final_text"],
                }
            )
            for tool_latency_ms in tool_latencies_ms:
                if action_ready_ms is None:
                    continue
                sequential_ms = float(action_ready_ms) + float(tool_latency_ms) + float(
                    restart_ms
                )
                tool_result_ready_ms = float(action_ready_ms) + float(tool_latency_ms)
                estimated_bind_ms = float(baseline_ms) * float(fraction)
                optimistic_reserved_ms = max(
                    float(reserved_ms),
                    tool_result_ready_ms,
                )
                latency_rows.append(
                    {
                        "policy": "reserved_slot",
                        "request_index": request_index,
                        "bind_fraction": fraction,
                        "tool_latency_ms": tool_latency_ms,
                        "action_ready_ms": action_ready_ms,
                        "tool_result_ready_ms": tool_result_ready_ms,
                        "estimated_bind_ms": estimated_bind_ms,
                        "bind_not_before_tool_ready": (
                            estimated_bind_ms >= tool_result_ready_ms
                        ),
                        "no_bind_elapsed_ms": baseline_ms,
                        "restart_with_result_ms": restart_ms,
                        "bind_elapsed_ms": reserved_ms,
                        "sequential_restart_path_ms": sequential_ms,
                        "inflight_binding_upper_bound_ms": optimistic_reserved_ms,
                        "saved_ms": sequential_ms - optimistic_reserved_ms,
                        "speedup": (
                            sequential_ms / optimistic_reserved_ms
                            if optimistic_reserved_ms > 0
                            else None
                        ),
                        "bind_success": reserved["success"],
                        "restart_success": restart["success"],
                    }
                )
        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "no_bind_success": baseline["success"],
                    "restart_success": restart["success"],
                    "action_ready_fraction": action_ready_fraction,
                    "bound_tokens": len(plan["positions"]),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    aggregate: dict[str, Any] = {
        "num_requests": len(records),
        "bind_fractions": bind_fractions,
        "steps": sampler_config.steps,
        "max_new_tokens": sampler_config.max_new_tokens,
        "use_cache": sampler_config.use_cache,
        "threshold": sampler_config.threshold,
        "tool_latencies_ms": tool_latencies_ms,
    }
    for key, rows in _group_rows(request_rows).items():
        aggregate[f"{key}_success_rate"] = sum(row["success"] for row in rows) / len(rows)
        aggregate[f"{key}_action_ready_rate"] = (
            sum(row["action_ready"] for row in rows) / len(rows)
        )
        aggregate[f"{key}_result_terms_rate"] = (
            sum(row["result_terms_present"] for row in rows) / len(rows)
        )
        aggregate[f"{key}_mean_elapsed_ms"] = statistics.mean(
            [float(row["elapsed_ms"]) for row in rows]
        )
        aggregate[f"{key}_p95_elapsed_ms"] = _percentile(
            [float(row["elapsed_ms"]) for row in rows],
            0.95,
        )
    for tool_latency_ms in tool_latencies_ms:
        for fraction in bind_fractions:
            for policy in ("bind", "reserved_slot"):
                rows = [
                    row
                    for row in latency_rows
                    if row["policy"] == policy
                    and float(row["tool_latency_ms"]) == float(tool_latency_ms)
                    and float(row["bind_fraction"]) == float(fraction)
                ]
                if not rows:
                    continue
                key = f"latency_{policy}_tool{tool_latency_ms:g}_bind{fraction:g}"
                aggregate[f"{key}_mean_sequential_restart_ms"] = statistics.mean(
                    [float(row["sequential_restart_path_ms"]) for row in rows]
                )
                aggregate[f"{key}_mean_inflight_upper_bound_ms"] = statistics.mean(
                    [float(row["inflight_binding_upper_bound_ms"]) for row in rows]
                )
                aggregate[f"{key}_mean_saved_ms"] = statistics.mean(
                    [float(row["saved_ms"]) for row in rows]
                )
                aggregate[f"{key}_mean_speedup"] = statistics.mean(
                    [
                        float(row["speedup"])
                        for row in rows
                        if row["speedup"] is not None
                    ]
                )

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    decoded_path = prefix.with_name(prefix.name + "_decoded.jsonl")
    latency_path = prefix.with_name(prefix.name + "_latency_model.csv")
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


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["policy"] == "no_bind":
            key = "no_bind"
        elif row["policy"] == "restart_with_result":
            key = "restart_with_result"
        elif row["policy"] == "reserved_slot":
            key = f"reserved_slot_{float(row['bind_fraction']):g}"
        else:
            key = f"bind_{float(row['bind_fraction']):g}"
        grouped.setdefault(key, []).append(row)
    return grouped


if __name__ == "__main__":
    main()
