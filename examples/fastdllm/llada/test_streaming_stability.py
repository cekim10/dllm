"""
Profile token stability for monotonic streaming on Fast-dLLM LLaDA.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/test_streaming_stability.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/elastic_canvas_prompts.jsonl \
    --limit 16 \
    --steps 128 \
    --max_new_tokens 128 \
    --block_size 32 \
    --use_cache prefix \
    --threshold 0.9 \
    --output_prefix artifacts/streaming_stability/llada_prefix_s128
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm


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


def _normalize_inputs(inputs: Any) -> list[list[int]]:
    if isinstance(inputs, torch.Tensor):
        if inputs.dim() == 1:
            return [inputs.tolist()]
        return inputs.tolist()
    if inputs and isinstance(inputs[0], int):
        return [inputs]
    return inputs


def _effective_generated_length(
    *,
    sequence: list[int],
    prompt_len: int,
    max_new_tokens: int,
    stop_ids: set[int],
) -> int:
    span = sequence[prompt_len : prompt_len + max_new_tokens]
    for index, token_id in enumerate(span):
        if token_id in stop_ids:
            return index
    return len(span)


def _first_final_stable_step(history_tokens: list[int], final_token: int) -> int:
    for step in range(len(history_tokens)):
        if history_tokens[step] != final_token:
            continue
        if all(token == final_token for token in history_tokens[step:]):
            return step
    return len(history_tokens) - 1


def _first_k_stable_commit_step(history_tokens: list[int], stable_window: int) -> tuple[int, int]:
    if stable_window <= 1:
        return 0, history_tokens[0]
    for step in range(stable_window - 1, len(history_tokens)):
        window = history_tokens[step - stable_window + 1 : step + 1]
        if all(token == window[0] for token in window):
            return step, window[0]
    return len(history_tokens) - 1, history_tokens[-1]


def _analyze_history(
    *,
    histories: list[torch.Tensor],
    final_sequence: list[int],
    prompt_len: int,
    max_new_tokens: int,
    stop_ids: set[int],
    stable_windows: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history_rows = []
    final_len = _effective_generated_length(
        sequence=final_sequence,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
    )
    if final_len <= 0:
        return [], {
            "final_generated_tokens": 0,
            "num_history_steps": len(histories),
        }

    history_lists = []
    for item in histories:
        tensor = item
        if tensor.dim() == 2:
            tensor = tensor[0]
        history_lists.append(tensor.tolist())

    num_steps = len(history_lists)
    stable_steps = []
    flip_counts = []
    for pos in range(final_len):
        absolute_pos = prompt_len + pos
        tokens = [history[absolute_pos] for history in history_lists]
        final_token = final_sequence[absolute_pos]
        stable_step = _first_final_stable_step(tokens, final_token)
        flips = sum(1 for left, right in zip(tokens, tokens[1:]) if left != right)
        stable_steps.append(stable_step)
        flip_counts.append(flips)
        row = {
            "position": pos,
            "absolute_position": absolute_pos,
            "final_token": final_token,
            "final_stable_step": stable_step,
            "final_stable_fraction": stable_step / max(num_steps - 1, 1),
            "flip_count": flips,
        }
        for stable_window in stable_windows:
            commit_step, committed_token = _first_k_stable_commit_step(tokens, stable_window)
            row[f"commit_step_w{stable_window}"] = commit_step
            row[f"commit_fraction_w{stable_window}"] = commit_step / max(num_steps - 1, 1)
            row[f"commit_correct_w{stable_window}"] = committed_token == final_token
        history_rows.append(row)

    request_summary: dict[str, Any] = {
        "final_generated_tokens": final_len,
        "num_history_steps": num_steps,
        "mean_final_stable_step": statistics.mean(stable_steps),
        "p50_final_stable_fraction": _percentile(
            [step / max(num_steps - 1, 1) for step in stable_steps],
            0.50,
        ),
        "p90_final_stable_fraction": _percentile(
            [step / max(num_steps - 1, 1) for step in stable_steps],
            0.90,
        ),
        "p95_final_stable_fraction": _percentile(
            [step / max(num_steps - 1, 1) for step in stable_steps],
            0.95,
        ),
        "mean_flip_count": statistics.mean(flip_counts),
        "p95_flip_count": _percentile([float(value) for value in flip_counts], 0.95),
    }
    for stable_window in stable_windows:
        commit_fractions = [
            float(row[f"commit_fraction_w{stable_window}"]) for row in history_rows
        ]
        correct = [
            bool(row[f"commit_correct_w{stable_window}"]) for row in history_rows
        ]
        request_summary[f"mean_commit_fraction_w{stable_window}"] = statistics.mean(
            commit_fractions
        )
        request_summary[f"p50_commit_fraction_w{stable_window}"] = _percentile(
            commit_fractions, 0.50
        )
        request_summary[f"p90_commit_fraction_w{stable_window}"] = _percentile(
            commit_fractions, 0.90
        )
        request_summary[f"correction_rate_w{stable_window}"] = (
            1.0 - sum(correct) / len(correct) if correct else 0.0
        )
    return history_rows, request_summary


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
    prompt: str = "Explain why diffusion language model streaming is different from autoregressive streaming."
    limit: int = 16
    stable_windows: str = "1,2,3,4"
    seed: int = 42
    output_prefix: str = "artifacts/streaming_stability/llada_prefix_s128"

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


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

stable_windows = [
    int(part.strip())
    for part in script_args.stable_windows.split(",")
    if part.strip()
]
if not stable_windows:
    raise ValueError("--stable_windows must contain at least one integer")

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

stop_ids = {
    token_id
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "eot_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    )
    if token_id is not None
}

request_rows = []
position_rows = []
decoded_rows = []
for request_index, message in enumerate(messages):
    inputs = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = _normalize_inputs(inputs)[0]

    _sync_device(model.device)
    start = time.perf_counter()
    outputs = sampler.sample(inputs, config=sampler_config, return_dict=True)
    _sync_device(model.device)
    elapsed_seconds = time.perf_counter() - start

    final_sequence = outputs.sequences[0].tolist()
    decoded = dllm.utils.sample_trim(tokenizer, [final_sequence], [prompt_ids])[0]
    if outputs.histories is None:
        raise RuntimeError("Sampler did not return histories.")
    rows, summary = _analyze_history(
        histories=outputs.histories,
        final_sequence=final_sequence,
        prompt_len=len(prompt_ids),
        max_new_tokens=sampler_config.max_new_tokens,
        stop_ids=stop_ids,
        stable_windows=stable_windows,
    )
    summary.update(
        {
            "request_index": request_index,
            "prompt_tokens": len(prompt_ids),
            "elapsed_seconds": elapsed_seconds,
            "decoded_chars": len(decoded),
            "decoded_text": decoded,
        }
    )
    request_rows.append(summary)
    decoded_rows.append(
        {
            "request_index": request_index,
            "prompt": message[-1]["content"] if message else "",
            "decoded_text": decoded,
        }
    )
    for row in rows:
        row["request_index"] = request_index
        position_rows.append(row)

    compact = {
        "request_index": request_index,
        "tokens": summary["final_generated_tokens"],
        "histories": summary["num_history_steps"],
        "p50_stable": summary.get("p50_final_stable_fraction", 0.0),
        "p90_stable": summary.get("p90_final_stable_fraction", 0.0),
    }
    for stable_window in stable_windows:
        compact[f"corr_w{stable_window}"] = summary.get(
            f"correction_rate_w{stable_window}", 0.0
        )
        compact[f"p50_commit_w{stable_window}"] = summary.get(
            f"p50_commit_fraction_w{stable_window}", 0.0
        )
    print(json.dumps(compact, ensure_ascii=True), flush=True)

aggregate: dict[str, Any] = {
    "num_requests": len(request_rows),
    "steps": sampler_config.steps,
    "max_new_tokens": sampler_config.max_new_tokens,
    "block_size": sampler_config.block_size,
    "use_cache": sampler_config.use_cache,
    "threshold": sampler_config.threshold,
    "stable_windows": stable_windows,
}
if request_rows:
    for key in request_rows[0]:
        if key in ("request_index", "decoded_text"):
            continue
        values = [row[key] for row in request_rows if isinstance(row.get(key), (int, float))]
        if values:
            aggregate[f"mean_{key}"] = statistics.mean(values)
            aggregate[f"p50_{key}"] = _percentile([float(value) for value in values], 0.50)
            aggregate[f"p90_{key}"] = _percentile([float(value) for value in values], 0.90)

output = {
    "aggregate": aggregate,
    "requests": request_rows,
}

prefix = Path(script_args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
summary_path = prefix.with_name(prefix.name + "_summary.json")
requests_path = prefix.with_name(prefix.name + "_requests.csv")
positions_path = prefix.with_name(prefix.name + "_positions.csv")
decoded_path = prefix.with_name(prefix.name + "_decoded.jsonl")
summary_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
_write_csv(requests_path, request_rows)
_write_csv(positions_path, position_rows)
with decoded_path.open("w", encoding="utf-8") as f:
    for row in decoded_rows:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

print(f"Saved summary: {summary_path}")
print(f"Saved request CSV: {requests_path}")
print(f"Saved position CSV: {positions_path}")
print("Aggregate:")
print(json.dumps(aggregate, ensure_ascii=True, indent=2))
