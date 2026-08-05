"""
Measure whether dLLM intermediate states expose tool/RAG intent earlier than AR prefixes.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_tool_prefetch_signals.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/tool_prefetch_prompts.jsonl \
    --limit 20 \
    --steps 128 \
    --max_new_tokens 128 \
    --block_size 32 \
    --use_cache prefix \
    --threshold 0.9 \
    --output_prefix artifacts/tool_prefetch/llada_prefix_s128
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm


TOOL_NAMES = [
    "flight_search",
    "weather",
    "web_search",
    "vector_search",
    "sql_query",
    "calendar_api",
    "crm_api",
]
TOOL_ALIASES = {
    "flight_search": ["flight_search", "flight search", "flights", "flight"],
    "weather": ["weather", "forecast"],
    "web_search": ["web_search", "web search", "search the web", "search online"],
    "vector_search": ["vector_search", "vector search", "retrieve", "knowledge base"],
    "sql_query": ["sql_query", "sql query", "query"],
    "calendar_api": ["calendar_api", "calendar api", "calendar"],
    "crm_api": ["crm_api", "crm api", "crm", "contacts"],
}
DATE_PATTERN = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def _normalize_inputs(inputs: Any) -> list[list[int]]:
    if isinstance(inputs, torch.Tensor):
        if inputs.dim() == 1:
            return [inputs.tolist()]
        return inputs.tolist()
    if inputs and isinstance(inputs[0], int):
        return [inputs]
    return inputs


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = DATE_PATTERN.sub(
        lambda match: f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}",
        lowered,
    )
    return re.sub(r"\s+", " ", lowered).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 2 and token not in {"the", "and", "for", "from", "with"}
    }


def _contains_value(text: str, value: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_value = _normalize_text(value)
    if normalized_value in normalized_text:
        return True
    return _compact(normalized_value) in _compact(normalized_text)


def _query_overlap(text: str, value: str) -> float:
    target = _token_set(value)
    if not target:
        return 0.0
    observed = _token_set(text)
    return len(target & observed) / len(target)


def _detect_tool(text: str, target_tool: str | None = None) -> str | None:
    normalized = _normalize_text(text)
    if target_tool is not None and target_tool in normalized:
        return target_tool
    for tool in TOOL_NAMES:
        if tool in normalized:
            return tool
    hits = []
    for tool, aliases in TOOL_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            hits.append(tool)
    if len(hits) == 1:
        return hits[0]
    return None


def _score_extraction(text: str, target_tool: str, target_args: dict[str, str]) -> dict[str, Any]:
    detected_tool = _detect_tool(text, target_tool=target_tool)
    tool_correct = detected_tool == target_tool
    arg_scores = {}
    for key, value in target_args.items():
        if key in {"query", "filter", "where", "select"}:
            score = _query_overlap(text, str(value))
            arg_scores[key] = score >= 0.67
        elif " " in str(value).strip():
            score = _query_overlap(text, str(value))
            arg_scores[key] = score >= 0.80
        else:
            arg_scores[key] = _contains_value(text, str(value))
    args_correct = all(arg_scores.values()) if arg_scores else True
    return {
        "detected_tool": detected_tool,
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "ready": tool_correct and args_correct,
        "arg_scores": arg_scores,
    }


def _format_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    tools = "\n".join(
        [
            "- flight_search(origin, destination, date)",
            "- weather(location, date)",
            "- web_search(query)",
            "- vector_search(query)",
            "- sql_query(table, select, where, order_by, limit)",
            "- calendar_api(calendar, start_date, end_date, keyword)",
            "- crm_api(company, role, city)",
        ]
    )
    content = (
        "You are a tool router. Select exactly one read-only tool for the user request.\n"
        f"Available tools:\n{tools}\n\n"
        "Return only this format:\n"
        "TOOL: <tool_name>\n"
        "ARGS: key=value; key=value\n\n"
        f"User request: {record['prompt']}"
    )
    return [{"role": "user", "content": content}]


def _load_records(path: str, limit: int) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record or "tool" not in record or "args" not in record:
                raise ValueError(f"Unsupported record: {record}")
            records.append(record)
            if len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _decode_generated(
    *,
    tokenizer: Any,
    sequence: list[int],
    prompt_len: int,
    max_new_tokens: int,
) -> str:
    span = sequence[prompt_len : prompt_len + max_new_tokens]
    return tokenizer.decode(span, skip_special_tokens=True)


def _history_texts(
    *,
    tokenizer: Any,
    histories: list[torch.Tensor],
    prompt_len: int,
    max_new_tokens: int,
) -> list[str]:
    texts = []
    for item in histories:
        tensor = item
        if tensor.dim() == 2:
            tensor = tensor[0]
        texts.append(
            _decode_generated(
                tokenizer=tokenizer,
                sequence=tensor.tolist(),
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
            )
        )
    return texts


def _first_ready_fraction(
    *,
    texts: list[str],
    target_tool: str,
    target_args: dict[str, str],
) -> tuple[int | None, float | None, dict[str, Any] | None]:
    denominator = max(len(texts) - 1, 1)
    for index, text in enumerate(texts):
        score = _score_extraction(text, target_tool, target_args)
        if score["ready"]:
            return index, index / denominator, score
    return None, None, None


def _stable_ready_fraction(
    *,
    texts: list[str],
    target_tool: str,
    target_args: dict[str, str],
) -> tuple[int | None, float | None]:
    ready = [
        bool(_score_extraction(text, target_tool, target_args)["ready"])
        for text in texts
    ]
    denominator = max(len(texts) - 1, 1)
    for index, is_ready in enumerate(ready):
        if is_ready and all(ready[index:]):
            return index, index / denominator
    return None, None


def _false_start_count(
    *,
    texts: list[str],
    target_tool: str,
    target_args: dict[str, str],
) -> int:
    ready = [
        bool(_score_extraction(text, target_tool, target_args)["ready"])
        for text in texts
    ]
    false_starts = 0
    for current, following in zip(ready, ready[1:]):
        if current and not following:
            false_starts += 1
    return false_starts


def _ar_prefix_texts(final_text: str, num_steps: int) -> list[str]:
    if num_steps <= 1:
        return [final_text]
    texts = []
    for index in range(num_steps):
        fraction = index / (num_steps - 1)
        end = math.ceil(len(final_text) * fraction)
        texts.append(final_text[:end])
    return texts


def _overlap_saving(
    *,
    ready_fraction: float | None,
    generation_ms: float,
    tool_latency_ms: float,
) -> float:
    sequential_ms = generation_ms + tool_latency_ms
    if ready_fraction is None:
        return 0.0
    ready_ms = generation_ms * ready_fraction
    speculative_ms = max(generation_ms, ready_ms + tool_latency_ms)
    return max(0.0, (sequential_ms - speculative_ms) / sequential_ms)


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
    input_path: str = "examples/fastdllm/llada/tool_prefetch_prompts.jsonl"
    limit: int = 20
    seed: int = 42
    tool_latencies_ms: str = "50,100,300,500"
    output_prefix: str = "artifacts/tool_prefetch/llada_prefix_s128"

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
    tool_latencies = _parse_float_list(script_args.tool_latencies_ms)

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
    step_rows = []
    for request_index, record in enumerate(records):
        messages = _format_prompt(record)
        inputs = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_ids = _normalize_inputs(inputs)[0]

        _sync_device(model.device)
        start = time.perf_counter()
        outputs = sampler.sample(inputs, config=sampler_config, return_dict=True)
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

        target_tool = str(record["tool"])
        target_args = {str(key): str(value) for key, value in record["args"].items()}
        final_score = _score_extraction(final_text, target_tool, target_args)
        dllm_step, dllm_fraction, dllm_score = _first_ready_fraction(
            texts=dllm_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        ar_step, ar_fraction, ar_score = _first_ready_fraction(
            texts=ar_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        dllm_stable_step, dllm_stable_fraction = _stable_ready_fraction(
            texts=dllm_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        ar_stable_step, ar_stable_fraction = _stable_ready_fraction(
            texts=ar_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        dllm_false_starts = _false_start_count(
            texts=dllm_texts,
            target_tool=target_tool,
            target_args=target_args,
        )
        ar_false_starts = _false_start_count(
            texts=ar_texts,
            target_tool=target_tool,
            target_args=target_args,
        )

        row: dict[str, Any] = {
            "request_index": request_index,
            "prompt": record["prompt"],
            "target_tool": target_tool,
            "target_args": json.dumps(target_args, sort_keys=True),
            "generation_ms": generation_ms,
            "num_history_steps": len(dllm_texts),
            "final_ready": final_score["ready"],
            "final_detected_tool": final_score["detected_tool"],
            "dllm_ready_step": dllm_step,
            "dllm_ready_fraction": dllm_fraction,
            "dllm_stable_step": dllm_stable_step,
            "dllm_stable_fraction": dllm_stable_fraction,
            "dllm_false_starts": dllm_false_starts,
            "ar_ready_step": ar_step,
            "ar_ready_fraction": ar_fraction,
            "ar_stable_step": ar_stable_step,
            "ar_stable_fraction": ar_stable_fraction,
            "ar_false_starts": ar_false_starts,
            "dllm_beats_ar": (
                dllm_fraction is not None
                and (ar_fraction is None or dllm_fraction < ar_fraction)
            ),
            "dllm_lead_fraction": (
                (ar_fraction - dllm_fraction)
                if ar_fraction is not None and dllm_fraction is not None
                else None
            ),
        }
        for latency_ms in tool_latencies:
            row[f"dllm_saving_tool{int(latency_ms)}ms"] = _overlap_saving(
                ready_fraction=dllm_fraction,
                generation_ms=generation_ms,
                tool_latency_ms=latency_ms,
            )
            row[f"ar_saving_tool{int(latency_ms)}ms"] = _overlap_saving(
                ready_fraction=ar_fraction,
                generation_ms=generation_ms,
                tool_latency_ms=latency_ms,
            )
        request_rows.append(row)
        decoded_rows.append(
            {
                "request_index": request_index,
                "prompt": record["prompt"],
                "target_tool": target_tool,
                "target_args": target_args,
                "final_text": final_text,
                "dllm_ready_text": dllm_texts[dllm_step] if dllm_step is not None else None,
                "ar_ready_text": ar_texts[ar_step] if ar_step is not None else None,
                "dllm_ready_score": dllm_score,
                "ar_ready_score": ar_score,
            }
        )
        for step_index, text in enumerate(dllm_texts):
            score = _score_extraction(text, target_tool, target_args)
            step_rows.append(
                {
                    "request_index": request_index,
                    "step": step_index,
                    "fraction": step_index / max(len(dllm_texts) - 1, 1),
                    "source": "dllm",
                    "detected_tool": score["detected_tool"],
                    "ready": score["ready"],
                    "text": text[:500],
                }
            )
        for step_index, text in enumerate(ar_texts):
            score = _score_extraction(text, target_tool, target_args)
            step_rows.append(
                {
                    "request_index": request_index,
                    "step": step_index,
                    "fraction": step_index / max(len(ar_texts) - 1, 1),
                    "source": "ar_prefix",
                    "detected_tool": score["detected_tool"],
                    "ready": score["ready"],
                    "text": text[:500],
                }
            )
        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "final_ready": final_score["ready"],
                    "dllm_ready_fraction": dllm_fraction,
                    "dllm_stable_fraction": dllm_stable_fraction,
                    "dllm_false_starts": dllm_false_starts,
                    "ar_ready_fraction": ar_fraction,
                    "ar_stable_fraction": ar_stable_fraction,
                    "dllm_beats_ar": row["dllm_beats_ar"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    valid_rows = [row for row in request_rows if row["final_ready"]]
    aggregate: dict[str, Any] = {
        "num_requests": len(request_rows),
        "num_final_ready": len(valid_rows),
        "final_ready_rate": len(valid_rows) / len(request_rows) if request_rows else 0.0,
        "steps": sampler_config.steps,
        "max_new_tokens": sampler_config.max_new_tokens,
        "use_cache": sampler_config.use_cache,
        "threshold": sampler_config.threshold,
    }
    if valid_rows:
        dllm_ready = [row for row in valid_rows if row["dllm_ready_fraction"] is not None]
        ar_ready = [row for row in valid_rows if row["ar_ready_fraction"] is not None]
        aggregate["dllm_ready_rate"] = len(dllm_ready) / len(valid_rows)
        aggregate["ar_ready_rate"] = len(ar_ready) / len(valid_rows)
        aggregate["dllm_beats_ar_rate"] = (
            sum(1 for row in valid_rows if row["dllm_beats_ar"]) / len(valid_rows)
        )
        aggregate["mean_dllm_ready_fraction"] = statistics.mean(
            [float(row["dllm_ready_fraction"]) for row in dllm_ready]
        ) if dllm_ready else None
        aggregate["mean_ar_ready_fraction"] = statistics.mean(
            [float(row["ar_ready_fraction"]) for row in ar_ready]
        ) if ar_ready else None
        dllm_stable = [
            row for row in valid_rows if row["dllm_stable_fraction"] is not None
        ]
        ar_stable = [
            row for row in valid_rows if row["ar_stable_fraction"] is not None
        ]
        aggregate["dllm_stable_rate"] = len(dllm_stable) / len(valid_rows)
        aggregate["ar_stable_rate"] = len(ar_stable) / len(valid_rows)
        aggregate["mean_dllm_stable_fraction"] = statistics.mean(
            [float(row["dllm_stable_fraction"]) for row in dllm_stable]
        ) if dllm_stable else None
        aggregate["mean_ar_stable_fraction"] = statistics.mean(
            [float(row["ar_stable_fraction"]) for row in ar_stable]
        ) if ar_stable else None
        aggregate["mean_dllm_false_starts"] = statistics.mean(
            [float(row["dllm_false_starts"]) for row in valid_rows]
        )
        aggregate["mean_ar_false_starts"] = statistics.mean(
            [float(row["ar_false_starts"]) for row in valid_rows]
        )
        leads = [
            float(row["dllm_lead_fraction"])
            for row in valid_rows
            if row["dllm_lead_fraction"] is not None
        ]
        aggregate["mean_dllm_lead_fraction"] = statistics.mean(leads) if leads else None
        aggregate["p50_dllm_lead_fraction"] = _percentile(leads, 0.50) if leads else None
        for latency_ms in tool_latencies:
            dllm_key = f"dllm_saving_tool{int(latency_ms)}ms"
            ar_key = f"ar_saving_tool{int(latency_ms)}ms"
            aggregate[f"mean_{dllm_key}"] = statistics.mean(
                [float(row[dllm_key]) for row in valid_rows]
            )
            aggregate[f"mean_{ar_key}"] = statistics.mean(
                [float(row[ar_key]) for row in valid_rows]
            )

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    steps_path = prefix.with_name(prefix.name + "_steps.csv")
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
    _write_csv(steps_path, step_rows)
    with decoded_path.open("w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Saved summary: {summary_path}")
    print(f"Saved request CSV: {requests_path}")
    print(f"Saved step CSV: {steps_path}")
    print(f"Saved decoded JSONL: {decoded_path}")
    print("Aggregate:")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
