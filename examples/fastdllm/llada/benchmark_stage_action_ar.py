"""
Benchmark AR stage-local action extraction on multi-tool workflow prompts.

Run from repo root on a GPU server:
  source ~/.zshrc
  conda activate ~/miniconda3/envs/dllm
  python -u examples/fastdllm/llada/benchmark_stage_action_ar.py \
    --ar_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --input_path examples/fastdllm/llada/multitool_prefetch_prompts_3call_120.jsonl \
    --limit 20 \
    --max_new_tokens 64 \
    --output_prefix artifacts/nfe_stage_ablation/ar_stage_qwen2_5_7b_3call

This is the harness diagnostic for the NFE stage-ablation experiments. If a
strong AR model fails the same stage prompt/eval, the task or prompt is the
bottleneck rather than dLLM fidelity.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

from run_multitool_nfe_stage_ablation import _stage_prompt
from test_multitool_prefetch_signals import _call_score, _load_multitool_records
from test_tool_prefetch_signals import _percentile


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _dtype_from_arg(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decode_new_tokens(
    *,
    tokenizer: Any,
    generated_ids: torch.Tensor,
    prompt_token_count: int,
) -> str:
    new_ids = generated_ids[0, prompt_token_count:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


@dataclass
class StageResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int


def _run_stage(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> StageResult:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(rendered, list):
        input_ids = torch.tensor([rendered], dtype=torch.long)
    else:
        input_ids = rendered
    input_ids = input_ids.to(_device_of(model))
    prompt_tokens = int(input_ids.shape[-1])
    _sync_device(_device_of(model))
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    _sync_device(_device_of(model))
    latency_ms = (time.perf_counter() - start) * 1000.0
    return StageResult(
        text=_decode_new_tokens(
            tokenizer=tokenizer,
            generated_ids=generated,
            prompt_token_count=prompt_tokens,
        ),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=int(generated.shape[-1] - prompt_tokens),
    )


def _summarize(
    *,
    request_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "ar_model_name_or_path": args.ar_model_name_or_path,
        "input_path": args.input_path,
        "num_requests": len(request_rows),
        "num_stages": len(stage_rows),
        "max_new_tokens": args.max_new_tokens,
    }
    if request_rows:
        aggregate["request_all_ready_rate"] = (
            sum(bool(row["all_ready"]) for row in request_rows) / len(request_rows)
        )
        aggregate["mean_request_ready_count"] = statistics.mean(
            [float(row["ready_count"]) for row in request_rows]
        )
        aggregate["mean_request_latency_ms"] = statistics.mean(
            [float(row["total_latency_ms"]) for row in request_rows]
        )
        aggregate["p95_request_latency_ms"] = _percentile(
            [float(row["total_latency_ms"]) for row in request_rows],
            0.95,
        )
    for stage_index in sorted({int(row["stage_index"]) for row in stage_rows}):
        rows = [
            row for row in stage_rows if int(row["stage_index"]) == stage_index
        ]
        tag = f"stage{stage_index}"
        aggregate[f"{tag}_ready_rate"] = (
            sum(bool(row["ready"]) for row in rows) / max(len(rows), 1)
        )
        aggregate[f"{tag}_tool_correct_rate"] = (
            sum(bool(row["tool_correct"]) for row in rows) / max(len(rows), 1)
        )
        aggregate[f"{tag}_args_correct_rate"] = (
            sum(bool(row["args_correct"]) for row in rows) / max(len(rows), 1)
        )
        aggregate[f"{tag}_mean_latency_ms"] = statistics.mean(
            [float(row["latency_ms"]) for row in rows]
        )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_model_name_or_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    records = _load_multitool_records(args.input_path, args.limit)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.ar_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.ar_model_name_or_path,
        torch_dtype=_dtype_from_arg(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    ).eval()

    request_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []

    for request_index, record in enumerate(records):
        calls = list(record["calls"])
        stage_results = []
        for stage_index, call in enumerate(calls):
            messages = _stage_prompt(
                record,
                stage_index=stage_index,
                num_stages=len(calls),
            )
            result = _run_stage(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
            )
            score = _call_score(result.text, call)
            row = {
                "request_index": request_index,
                "stage_index": stage_index,
                "target_tool": call["tool"],
                "target_args": json.dumps(call["args"], sort_keys=True),
                "ready": bool(score["ready"]),
                "tool_correct": bool(score["tool_correct"]),
                "args_correct": bool(score["args_correct"]),
                "detected_tool": score["detected_tool"],
                "latency_ms": result.latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
            }
            stage_rows.append(row)
            stage_results.append({**row, "text": result.text})

        request_row = {
            "request_index": request_index,
            "num_stages": len(calls),
            "prompt": record["prompt"],
            "all_ready": all(bool(row["ready"]) for row in stage_results),
            "ready_count": sum(bool(row["ready"]) for row in stage_results),
            "total_latency_ms": sum(float(row["latency_ms"]) for row in stage_results),
            "total_output_tokens": sum(int(row["output_tokens"]) for row in stage_results),
        }
        request_rows.append(request_row)
        decoded_rows.append(
            {
                **request_row,
                "calls": calls,
                "stages": stage_results,
            }
        )
        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "ready_count": request_row["ready_count"],
                    "all_ready": request_row["all_ready"],
                    "total_latency_ms": request_row["total_latency_ms"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    stages_path = prefix.with_name(prefix.name + "_stages.csv")
    decoded_path = prefix.with_name(prefix.name + "_decoded.jsonl")
    aggregate = _summarize(
        request_rows=request_rows,
        stage_rows=stage_rows,
        args=args,
    )
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
    _write_csv(stages_path, stage_rows)
    with decoded_path.open("w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Saved summary: {summary_path}")
    print(f"Saved request CSV: {requests_path}")
    print(f"Saved stage CSV: {stages_path}")
    print(f"Saved decoded JSONL: {decoded_path}")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
