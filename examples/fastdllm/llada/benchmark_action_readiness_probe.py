"""
Benchmark an auxiliary AR probe against native dLLM action readiness.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/benchmark_action_readiness_probe.py \
    --ar_model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --input_path examples/fastdllm/llada/tool_prefetch_prompts_core.jsonl \
    --native_summary_path artifacts/tool_prefetch/llada_core_s128_summary.json \
    --limit 45 \
    --output_prefix artifacts/action_completeness/probe_vs_native_core

This is not a full SPORK implementation. It is a break-even microbenchmark for
the auxiliary prediction branch: how fast and accurately an AR probe can emit
the structured action, and how much extra model work it consumes relative to
native dLLM readiness already exposed by the main refinement trajectory.
"""

from __future__ import annotations

import argparse
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

from test_tool_prefetch_signals import (
    _format_prompt,
    _load_records,
    _normalize_inputs,
    _parse_float_list,
    _percentile,
    _score_extraction,
)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_native_rows(path: str | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(row["request_index"]): row
        for row in data.get("requests", [])
        if "request_index" in row
    }


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


def _decode_new_tokens(
    *,
    tokenizer: Any,
    generated_ids: torch.Tensor,
    prompt_token_count: int,
) -> str:
    new_ids = generated_ids[0, prompt_token_count:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _overlap_saving(
    *,
    ready_ms: float | None,
    generation_ms: float,
    tool_latency_ms: float,
) -> float:
    sequential_ms = generation_ms + tool_latency_ms
    if ready_ms is None:
        return 0.0
    speculative_ms = max(generation_ms, ready_ms + tool_latency_ms)
    return max(0.0, (sequential_ms - speculative_ms) / sequential_ms)


def _effective_latency(
    *,
    ready_ms: float | None,
    generation_ms: float,
    tool_latency_ms: float,
) -> float:
    if ready_ms is None:
        return generation_ms + tool_latency_ms
    return max(generation_ms, ready_ms + tool_latency_ms)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ProbeResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int


def _run_probe(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> ProbeResult:
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
    text = _decode_new_tokens(
        tokenizer=tokenizer,
        generated_ids=generated,
        prompt_token_count=prompt_tokens,
    )
    return ProbeResult(
        text=text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=int(generated.shape[-1] - prompt_tokens),
    )


def _aggregate(rows: list[dict[str, Any]], tool_latencies: list[float]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"num_requests": len(rows)}
    if not rows:
        return aggregate
    probe_ready = [row for row in rows if row["probe_ready"]]
    native_rows = [row for row in rows if row.get("native_generation_ms") is not None]
    aggregate["probe_ready_rate"] = len(probe_ready) / len(rows)
    aggregate["mean_probe_latency_ms"] = statistics.mean(
        [float(row["probe_latency_ms"]) for row in rows]
    )
    aggregate["p95_probe_latency_ms"] = _percentile(
        [float(row["probe_latency_ms"]) for row in rows],
        0.95,
    )
    aggregate["mean_probe_output_tokens"] = statistics.mean(
        [float(row["probe_output_tokens"]) for row in rows]
    )
    aggregate["mean_probe_prompt_tokens"] = statistics.mean(
        [float(row["probe_prompt_tokens"]) for row in rows]
    )
    if native_rows:
        aggregate["native_rows"] = len(native_rows)
        aggregate["mean_native_generation_ms"] = statistics.mean(
            [float(row["native_generation_ms"]) for row in native_rows]
        )
        aggregate["mean_native_ready_ms"] = statistics.mean(
            [float(row["native_ready_ms"]) for row in native_rows]
        )
        aggregate["mean_probe_over_native_generation"] = statistics.mean(
            [
                float(row["probe_latency_ms"]) / max(float(row["native_generation_ms"]), 1e-9)
                for row in native_rows
            ]
        )
        aggregate["probe_beats_native_ready_rate"] = (
            sum(
                1
                for row in native_rows
                if row["probe_ready"]
                and float(row["probe_latency_ms"]) < float(row["native_ready_ms"])
            )
            / len(native_rows)
        )
        for latency_ms in tool_latencies:
            key = int(latency_ms)
            aggregate[f"mean_no_spec_latency_tool{key}ms"] = statistics.mean(
                [float(row[f"no_spec_latency_tool{key}ms"]) for row in native_rows]
            )
            aggregate[f"mean_native_latency_tool{key}ms"] = statistics.mean(
                [float(row[f"native_latency_tool{key}ms"]) for row in native_rows]
            )
            aggregate[f"mean_probe_latency_tool{key}ms"] = statistics.mean(
                [float(row[f"probe_latency_tool{key}ms"]) for row in native_rows]
            )
            aggregate[f"mean_native_saving_tool{key}ms"] = statistics.mean(
                [float(row[f"native_saving_tool{key}ms"]) for row in native_rows]
            )
            aggregate[f"mean_probe_saving_tool{key}ms"] = statistics.mean(
                [float(row[f"probe_saving_tool{key}ms"]) for row in native_rows]
            )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_model_name_or_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--native_summary_path")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--tool_latencies_ms", default="50,100,300,500")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_prefix", default="artifacts/action_completeness/probe_vs_native")
    args = parser.parse_args()

    tool_latencies = _parse_float_list(args.tool_latencies_ms)
    records = _load_records(args.input_path, args.limit)
    native_rows = _load_native_rows(args.native_summary_path)

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

    request_rows = []
    decoded_rows = []
    for request_index, record in enumerate(records):
        messages = _format_prompt(record)
        target_tool = str(record["tool"])
        target_args = {str(key): str(value) for key, value in record["args"].items()}
        probe = _run_probe(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
        )
        score = _score_extraction(probe.text, target_tool, target_args)
        row: dict[str, Any] = {
            "request_index": request_index,
            "target_tool": target_tool,
            "target_args": json.dumps(target_args, sort_keys=True),
            "probe_ready": bool(score["ready"]),
            "probe_detected_tool": score["detected_tool"],
            "probe_latency_ms": probe.latency_ms,
            "probe_prompt_tokens": probe.prompt_tokens,
            "probe_output_tokens": probe.output_tokens,
        }
        native = native_rows.get(request_index)
        if native is not None and native.get("final_ready"):
            generation_ms = float(native["generation_ms"])
            native_fraction = native.get("dllm_stable_fraction")
            native_ready_ms = (
                generation_ms * float(native_fraction)
                if native_fraction is not None
                else None
            )
            row["native_generation_ms"] = generation_ms
            row["native_ready_fraction"] = native_fraction
            row["native_ready_ms"] = native_ready_ms
            for latency_ms in tool_latencies:
                key = int(latency_ms)
                row[f"no_spec_latency_tool{key}ms"] = generation_ms + latency_ms
                row[f"native_latency_tool{key}ms"] = _effective_latency(
                    ready_ms=native_ready_ms,
                    generation_ms=generation_ms,
                    tool_latency_ms=latency_ms,
                )
                row[f"probe_latency_tool{key}ms"] = _effective_latency(
                    ready_ms=probe.latency_ms if score["ready"] else None,
                    generation_ms=generation_ms,
                    tool_latency_ms=latency_ms,
                )
                row[f"native_saving_tool{key}ms"] = _overlap_saving(
                    ready_ms=native_ready_ms,
                    generation_ms=generation_ms,
                    tool_latency_ms=latency_ms,
                )
                row[f"probe_saving_tool{key}ms"] = _overlap_saving(
                    ready_ms=probe.latency_ms if score["ready"] else None,
                    generation_ms=generation_ms,
                    tool_latency_ms=latency_ms,
                )
        request_rows.append(row)
        decoded_rows.append(
            {
                "request_index": request_index,
                "target_tool": target_tool,
                "target_args": target_args,
                "probe_text": probe.text,
                "probe_score": score,
                "prompt": record["prompt"],
            }
        )
        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "probe_ready": score["ready"],
                    "probe_latency_ms": probe.latency_ms,
                    "probe_output_tokens": probe.output_tokens,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    aggregate = _aggregate(request_rows, tool_latencies)
    aggregate.update(
        {
            "ar_model_name_or_path": args.ar_model_name_or_path,
            "input_path": args.input_path,
            "native_summary_path": args.native_summary_path,
        }
    )

    prefix = Path(args.output_prefix)
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
