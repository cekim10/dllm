"""
Measure whether dLLM intermediate states expose multiple independent tool calls
in parallel rather than in serialization order.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_multitool_prefetch_signals.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/multitool_prefetch_prompts.jsonl \
    --limit 10 \
    --tool_latencies_ms 100,300,500,1000,2000 \
    --prompt_format action_list \
    --bias_strengths 0,0.2 \
    --steps 128 \
    --max_new_tokens 192 \
    --block_size 48 \
    --use_cache prefix \
    --threshold 0.9 \
    --output_prefix artifacts/action_completeness/multitool_llada_s128

This is a go/no-go characterization for multi-tool speculation. The key
question is whether the last independent call becomes ready much earlier in
dLLM refinement than in an optimistic AR final-prefix baseline.
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
    _ar_prefix_texts,
    _compact,
    _decode_generated,
    _history_texts,
    _normalize_inputs,
    _parse_float_list,
    _percentile,
    _score_extraction,
)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_multitool_records(path: str, limit: int) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            calls = record.get("calls")
            if "prompt" not in record or not isinstance(calls, list) or not calls:
                raise ValueError(f"Unsupported multi-tool record: {record}")
            for call in calls:
                if "tool" not in call or "args" not in call:
                    raise ValueError(f"Unsupported call: {call}")
            records.append(record)
            if len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _format_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    return _format_prompt_for_style(record, prompt_format="action_list")


def _format_prompt_for_style(
    record: dict[str, Any],
    *,
    prompt_format: str,
) -> list[dict[str, str]]:
    tool_lines = [
        "- flight_search(origin, destination, date)",
        "- weather(location, date)",
        "- calendar_api(calendar, start_date, end_date, keyword)",
        "- crm_api(company, role, city)",
    ]
    expected = []
    for index, call in enumerate(record["calls"], start=1):
        args = "; ".join(f"{key}={value}" for key, value in call["args"].items())
        expected.append(f"ACTION {index}: TOOL: {call['tool']} ARGS: {args}")
    header = (
        "You are a multi-tool router. The user request requires multiple independent "
        "read-only actions. Return every action; do not merge actions.\n"
        f"Available tools:\n" + "\n".join(tool_lines) + "\n\n"
    )
    footer = (
        "Expected actions, unordered in the user's wording but all required:\n"
        + "\n".join(expected)
        + f"\n\nUser request: {record['prompt']}"
    )
    if prompt_format == "action_list":
        format_lines = []
        for index in range(1, len(record["calls"]) + 1):
            format_lines.extend(
                [
                    f"ACTION {index}:",
                    "TOOL: <tool_name>",
                    "ARGS: key=value; key=value",
                ]
            )
        content = (
            header
            + "Return only this format:\n"
            + "\n".join(format_lines)
            + "\n\n"
            + footer
        )
    elif prompt_format == "json_array":
        array_items = ",".join(
            ['{"tool":"<tool_name>","args":{"key":"value"}}']
            * len(record["calls"])
        )
        content = (
            header
            + "Return only compact JSON with this shape:\n"
            + '{"actions":['
            + array_items
            + "]}\n\n"
            + footer
        )
    elif prompt_format == "named_object":
        content = (
            header
            + "Return only compact JSON with independent named fields, not an ordered list:\n"
            '{"weather_action":{"tool":"weather","args":{"location":"...","date":"..."}},'
            '"flight_action":{"tool":"flight_search","args":{"origin":"...","destination":"...","date":"..."}},'
            '"calendar_or_crm_action":{"tool":"calendar_api_or_crm_api","args":{"key":"value"}}}\n\n'
            + footer
        )
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")
    return [{"role": "user", "content": content}]


def _call_score(text: str, call: dict[str, Any]) -> dict[str, Any]:
    return _score_extraction(
        text,
        str(call["tool"]),
        {str(key): str(value) for key, value in call["args"].items()},
    )


def _call_ready_fractions(
    *,
    texts: list[str],
    calls: list[dict[str, Any]],
) -> tuple[list[int | None], list[float | None], list[int | None], list[float | None], list[int]]:
    denominator = max(len(texts) - 1, 1)
    first_steps: list[int | None] = []
    first_fractions: list[float | None] = []
    stable_steps: list[int | None] = []
    stable_fractions: list[float | None] = []
    false_starts: list[int] = []
    for call in calls:
        ready = [bool(_call_score(text, call)["ready"]) for text in texts]
        first_step = next((index for index, value in enumerate(ready) if value), None)
        stable_step = None
        for index, value in enumerate(ready):
            if value and all(ready[index:]):
                stable_step = index
                break
        starts = 0
        for current, following in zip(ready, ready[1:]):
            if current and not following:
                starts += 1
        first_steps.append(first_step)
        first_fractions.append(
            first_step / denominator if first_step is not None else None
        )
        stable_steps.append(stable_step)
        stable_fractions.append(
            stable_step / denominator if stable_step is not None else None
        )
        false_starts.append(starts)
    return first_steps, first_fractions, stable_steps, stable_fractions, false_starts


def _all_ready_step(
    *,
    texts: list[str],
    calls: list[dict[str, Any]],
) -> tuple[int | None, float | None]:
    denominator = max(len(texts) - 1, 1)
    for index, text in enumerate(texts):
        if all(bool(_call_score(text, call)["ready"]) for call in calls):
            return index, index / denominator
    return None, None


def _all_stable_step(
    *,
    texts: list[str],
    calls: list[dict[str, Any]],
) -> tuple[int | None, float | None]:
    denominator = max(len(texts) - 1, 1)
    ready = [
        all(bool(_call_score(text, call)["ready"]) for call in calls)
        for text in texts
    ]
    for index, value in enumerate(ready):
        if value and all(ready[index:]):
            return index, index / denominator
    return None, None


def _spread(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if len(present) < 2:
        return None
    return max(present) - min(present)


def _finish_time_parallel(
    *,
    generation_ms: float,
    ready_fractions: list[float | None],
    tool_latency_ms: float,
) -> float | None:
    if any(value is None for value in ready_fractions):
        return None
    tool_finish = max(float(value) * generation_ms + tool_latency_ms for value in ready_fractions)
    return max(generation_ms, tool_finish)


def _finish_time_serial(
    *,
    generation_ms: float,
    ready_fractions: list[float | None],
    tool_latency_ms: float,
) -> float | None:
    if any(value is None for value in ready_fractions):
        return None
    arrivals = sorted(float(value) * generation_ms for value in ready_fractions)
    server_time = 0.0
    for arrival in arrivals:
        server_time = max(server_time, arrival) + tool_latency_ms
    return max(generation_ms, server_time)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _target_compact_for_calls(calls: list[dict[str, Any]]) -> str:
    parts = []
    for call in calls:
        parts.append(str(call["tool"]))
        for key, value in call["args"].items():
            parts.append(str(key))
            parts.append(str(value))
    return _compact(" ".join(parts))


def _oracle_action_positions(
    *,
    tokenizer: Any,
    final_sequence: list[int],
    prompt_len: int,
    max_new_tokens: int,
    calls: list[dict[str, Any]],
    min_token_chars: int,
) -> list[int]:
    target = _target_compact_for_calls(calls)
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
) -> torch.Tensor | None:
    if strength <= 0:
        return None
    bias = torch.zeros(shape, dtype=torch.float32, device=device)
    for position in positions:
        if 0 <= position < shape[1]:
            bias[:, position] = float(strength)
    return bias


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str = "examples/fastdllm/llada/multitool_prefetch_prompts.jsonl"
    limit: int = 10
    seed: int = 42
    tool_latencies_ms: str = "100,300,500,1000,2000"
    prompt_format: str = "action_list"
    bias_strengths: str = "0"
    min_token_chars: int = 2
    output_prefix: str = "artifacts/action_completeness/multitool_llada_s128"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path,
            "BASE_MODELS_DIR",
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    steps: int = 128
    max_new_tokens: int = 192
    block_size: int = 48
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
    tool_latencies = _parse_float_list(script_args.tool_latencies_ms)
    bias_strengths = _parse_float_list(script_args.bias_strengths)

    records = _load_multitool_records(script_args.input_path, script_args.limit)
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
    call_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []

    for request_index, record in enumerate(records):
        calls = list(record["calls"])
        messages = _format_prompt_for_style(
            record,
            prompt_format=script_args.prompt_format,
        )
        inputs = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_ids = _normalize_inputs(inputs)[0]

        baseline_outputs = None
        baseline_positions: list[int] = []
        for bias_strength in bias_strengths:
            transfer_bias = None
            if bias_strength > 0:
                if baseline_outputs is None:
                    raise RuntimeError("Run bias_strength=0 before steered strengths.")
                transfer_bias = _make_transfer_bias(
                    shape=baseline_outputs.sequences.shape,
                    positions=baseline_positions,
                    strength=bias_strength,
                    device=model.device,
                )

            _sync_device(model.device)
            start = time.perf_counter()
            outputs = sampler.sample(
                inputs,
                config=sampler_config,
                return_dict=True,
                transfer_bias=transfer_bias,
            )
            _sync_device(model.device)
            generation_ms = (time.perf_counter() - start) * 1000.0

            if outputs.histories is None:
                raise RuntimeError("Sampler did not return histories.")
            final_sequence = outputs.sequences[0].tolist()
            final_text = _decode_generated(
                tokenizer=tokenizer,
                sequence=final_sequence,
                prompt_len=len(prompt_ids),
                max_new_tokens=sampler_config.max_new_tokens,
            )
            dllm_texts = _history_texts(
                tokenizer=tokenizer,
                histories=outputs.histories,
                prompt_len=len(prompt_ids),
                max_new_tokens=sampler_config.max_new_tokens,
            )
            ar_texts = _ar_prefix_texts(final_text, len(dllm_texts))
            if bias_strength == 0:
                baseline_outputs = outputs
                baseline_positions = _oracle_action_positions(
                    tokenizer=tokenizer,
                    final_sequence=final_sequence,
                    prompt_len=len(prompt_ids),
                    max_new_tokens=sampler_config.max_new_tokens,
                    calls=calls,
                    min_token_chars=script_args.min_token_chars,
                )

            dllm_first_steps, dllm_first_fracs, dllm_stable_steps, dllm_stable_fracs, dllm_false_starts = _call_ready_fractions(
                texts=dllm_texts,
                calls=calls,
            )
            ar_first_steps, ar_first_fracs, ar_stable_steps, ar_stable_fracs, ar_false_starts = _call_ready_fractions(
                texts=ar_texts,
                calls=calls,
            )
            dllm_all_step, dllm_all_fraction = _all_ready_step(texts=dllm_texts, calls=calls)
            ar_all_step, ar_all_fraction = _all_ready_step(texts=ar_texts, calls=calls)
            dllm_all_stable_step, dllm_all_stable_fraction = _all_stable_step(
                texts=dllm_texts,
                calls=calls,
            )
            ar_all_stable_step, ar_all_stable_fraction = _all_stable_step(
                texts=ar_texts,
                calls=calls,
            )

            final_ready = [bool(_call_score(final_text, call)["ready"]) for call in calls]
            request_row: dict[str, Any] = {
                "request_index": request_index,
                "prompt": record["prompt"],
                "prompt_format": script_args.prompt_format,
                "bias_strength": bias_strength,
                "oracle_action_position_count": len(baseline_positions),
                "num_calls": len(calls),
                "generation_ms": generation_ms,
                "num_history_steps": len(dllm_texts),
                "final_all_ready": all(final_ready),
                "final_ready_count": sum(final_ready),
                "dllm_all_ready_step": dllm_all_step,
                "dllm_all_ready_fraction": dllm_all_fraction,
                "dllm_all_stable_step": dllm_all_stable_step,
                "dllm_all_stable_fraction": dllm_all_stable_fraction,
                "dllm_call_ready_spread": _spread(dllm_first_fracs),
                "dllm_call_stable_spread": _spread(dllm_stable_fracs),
                "dllm_mean_call_ready_fraction": statistics.mean(
                    [float(value) for value in dllm_first_fracs if value is not None]
                ) if any(value is not None for value in dllm_first_fracs) else None,
                "ar_all_ready_step": ar_all_step,
                "ar_all_ready_fraction": ar_all_fraction,
                "ar_all_stable_step": ar_all_stable_step,
                "ar_all_stable_fraction": ar_all_stable_fraction,
                "ar_call_ready_spread": _spread(ar_first_fracs),
                "ar_call_stable_spread": _spread(ar_stable_fracs),
                "ar_mean_call_ready_fraction": statistics.mean(
                    [float(value) for value in ar_first_fracs if value is not None]
                ) if any(value is not None for value in ar_first_fracs) else None,
                "dllm_beats_ar_all_ready": (
                    dllm_all_fraction is not None
                    and (ar_all_fraction is None or dllm_all_fraction < ar_all_fraction)
                ),
                "dllm_all_ready_lead_fraction": (
                    ar_all_fraction - dllm_all_fraction
                    if ar_all_fraction is not None and dllm_all_fraction is not None
                    else None
                ),
            }
            request_rows.append(request_row)

            for call_index, call in enumerate(calls):
                call_rows.append(
                    {
                        "request_index": request_index,
                        "prompt_format": script_args.prompt_format,
                        "bias_strength": bias_strength,
                        "call_index": call_index,
                        "tool": call["tool"],
                        "args": json.dumps(call["args"], sort_keys=True),
                        "final_ready": final_ready[call_index],
                        "dllm_ready_step": dllm_first_steps[call_index],
                        "dllm_ready_fraction": dllm_first_fracs[call_index],
                        "dllm_stable_step": dllm_stable_steps[call_index],
                        "dllm_stable_fraction": dllm_stable_fracs[call_index],
                        "dllm_false_starts": dllm_false_starts[call_index],
                        "ar_ready_step": ar_first_steps[call_index],
                        "ar_ready_fraction": ar_first_fracs[call_index],
                        "ar_stable_step": ar_stable_steps[call_index],
                        "ar_stable_fraction": ar_stable_fracs[call_index],
                        "ar_false_starts": ar_false_starts[call_index],
                    }
                )

            for latency_ms in tool_latencies:
                no_spec_parallel = generation_ms + latency_ms
                no_spec_serial = generation_ms + latency_ms * len(calls)
                dllm_parallel = _finish_time_parallel(
                    generation_ms=generation_ms,
                    ready_fractions=dllm_first_fracs,
                    tool_latency_ms=latency_ms,
                )
                ar_parallel = _finish_time_parallel(
                    generation_ms=generation_ms,
                    ready_fractions=ar_first_fracs,
                    tool_latency_ms=latency_ms,
                )
                dllm_serial = _finish_time_serial(
                    generation_ms=generation_ms,
                    ready_fractions=dllm_first_fracs,
                    tool_latency_ms=latency_ms,
                )
                ar_serial = _finish_time_serial(
                    generation_ms=generation_ms,
                    ready_fractions=ar_first_fracs,
                    tool_latency_ms=latency_ms,
                )
                latency_rows.append(
                    {
                        "request_index": request_index,
                        "prompt_format": script_args.prompt_format,
                        "bias_strength": bias_strength,
                        "tool_latency_ms": latency_ms,
                        "num_calls": len(calls),
                        "generation_ms": generation_ms,
                        "no_spec_parallel_ms": no_spec_parallel,
                        "no_spec_serial_ms": no_spec_serial,
                        "dllm_parallel_ms": dllm_parallel,
                        "ar_parallel_ms": ar_parallel,
                        "dllm_serial_ms": dllm_serial,
                        "ar_serial_ms": ar_serial,
                        "dllm_vs_ar_parallel_speedup": _safe_ratio(
                            ar_parallel,
                            dllm_parallel,
                        ),
                        "dllm_vs_ar_serial_speedup": _safe_ratio(
                            ar_serial,
                            dllm_serial,
                        ),
                        "dllm_vs_no_spec_parallel_speedup": _safe_ratio(
                            no_spec_parallel,
                            dllm_parallel,
                        ),
                        "dllm_vs_no_spec_serial_speedup": _safe_ratio(
                            no_spec_serial,
                            dllm_serial,
                        ),
                    }
                )

            decoded_rows.append(
                {
                    "request_index": request_index,
                    "prompt_format": script_args.prompt_format,
                    "bias_strength": bias_strength,
                    "prompt": record["prompt"],
                    "calls": calls,
                    "final_text": final_text,
                    "dllm_all_ready_text": (
                        dllm_texts[dllm_all_step] if dllm_all_step is not None else None
                    ),
                    "ar_all_ready_text": (
                        ar_texts[ar_all_step] if ar_all_step is not None else None
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "request_index": request_index,
                        "bias_strength": bias_strength,
                        "final_ready_count": sum(final_ready),
                        "dllm_all_ready_fraction": dllm_all_fraction,
                        "ar_all_ready_fraction": ar_all_fraction,
                        "dllm_ready_spread": _spread(dllm_first_fracs),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    aggregate: dict[str, Any] = {
        "num_requests": len(request_rows),
        "num_calls": sum(int(row["num_calls"]) for row in request_rows),
        "steps": sampler_config.steps,
        "max_new_tokens": sampler_config.max_new_tokens,
        "block_size": sampler_config.block_size,
        "use_cache": sampler_config.use_cache,
        "threshold": sampler_config.threshold,
        "tool_latencies_ms": tool_latencies,
        "prompt_format": script_args.prompt_format,
        "bias_strengths": bias_strengths,
    }
    for bias_strength in bias_strengths:
        tag = f"bias{bias_strength:g}"
        rows_for_bias = [
            row
            for row in request_rows
            if abs(float(row["bias_strength"]) - float(bias_strength)) < 1e-9
        ]
        aggregate[f"{tag}_final_all_ready_rate"] = (
            sum(bool(row["final_all_ready"]) for row in rows_for_bias)
            / max(len(rows_for_bias), 1)
        )
        for key in [
            "generation_ms",
            "dllm_all_ready_fraction",
            "dllm_all_stable_fraction",
            "dllm_call_ready_spread",
            "dllm_call_stable_spread",
            "dllm_mean_call_ready_fraction",
            "ar_all_ready_fraction",
            "ar_all_stable_fraction",
            "ar_call_ready_spread",
            "ar_call_stable_spread",
            "ar_mean_call_ready_fraction",
            "dllm_all_ready_lead_fraction",
        ]:
            values = [
                float(row[key])
                for row in rows_for_bias
                if row.get(key) is not None
            ]
            if values:
                aggregate[f"{tag}_mean_{key}"] = statistics.mean(values)
                aggregate[f"{tag}_p95_{key}"] = _percentile(values, 0.95)
        aggregate[f"{tag}_dllm_beats_ar_all_ready_rate"] = (
            sum(bool(row["dllm_beats_ar_all_ready"]) for row in rows_for_bias)
            / max(len(rows_for_bias), 1)
        )
        for latency_ms in tool_latencies:
            rows = [
                row
                for row in latency_rows
                if abs(float(row["bias_strength"]) - float(bias_strength)) < 1e-9
                and float(row["tool_latency_ms"]) == latency_ms
            ]
            for key in [
                "dllm_vs_ar_parallel_speedup",
                "dllm_vs_ar_serial_speedup",
                "dllm_vs_no_spec_parallel_speedup",
                "dllm_vs_no_spec_serial_speedup",
            ]:
                values = [
                    float(row[key])
                    for row in rows
                    if row.get(key) not in (None, "")
                ]
                if values:
                    aggregate[f"{tag}_tool{latency_ms:g}_mean_{key}"] = statistics.mean(values)

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    calls_path = prefix.with_name(prefix.name + "_calls.csv")
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
    _write_csv(calls_path, call_rows)
    _write_csv(latency_path, latency_rows)
    with decoded_path.open("w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Saved summary: {summary_path}")
    print(f"Saved request CSV: {requests_path}")
    print(f"Saved calls CSV: {calls_path}")
    print(f"Saved latency model CSV: {latency_path}")
    print(f"Saved decoded JSONL: {decoded_path}")
    print("Aggregate:")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
