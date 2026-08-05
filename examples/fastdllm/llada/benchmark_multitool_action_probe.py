"""
Benchmark an auxiliary AR probe on multi-tool structured-action prompts.

Run from repo root on a GPU server:
  source .venv/bin/activate
  python -u examples/fastdllm/llada/benchmark_multitool_action_probe.py \
    --ar_model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --input_path examples/fastdllm/llada/multitool_prefetch_prompts_120.jsonl \
    --native_requests_csv artifacts/action_completeness/multitool_llada_scaleup_requests.csv \
    --limit 120 \
    --max_new_tokens 192 \
    --output_prefix artifacts/action_completeness/probe_vs_native_multitool_scaleup

This measures the real wall-clock cost and accuracy of an auxiliary AR branch
that tries to emit all structured actions for the same prompts used by the
native dLLM multi-tool readiness experiment. It is a probe-cost check, not a
full SPORK implementation.
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

from test_multitool_prefetch_signals import (
    _call_score,
    _format_prompt_for_style,
    _load_multitool_records,
    _parse_float_list,
    _percentile,
)


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


def _read_csv(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _native_by_request(path: str | None) -> dict[int, dict[str, str]]:
    rows = _read_csv(path)
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("bias_strength") not in (None, "", "0", "0.0"):
            continue
        try:
            request_index = int(row["request_index"])
        except (KeyError, ValueError):
            continue
        result.setdefault(request_index, row)
    return result


def _decode_new_tokens(
    *,
    tokenizer: Any,
    generated_ids: torch.Tensor,
    prompt_token_count: int,
) -> str:
    new_ids = generated_ids[0, prompt_token_count:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _effective_latency(
    *,
    generation_ms: float,
    ready_ms: float | None,
    tool_latency_ms: float,
) -> float:
    if ready_ms is None:
        return generation_ms + tool_latency_ms
    return max(generation_ms, ready_ms + tool_latency_ms)


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
    return ProbeResult(
        text=_decode_new_tokens(
            tokenizer=tokenizer,
            generated_ids=generated,
            prompt_token_count=prompt_tokens,
        ),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=int(generated.shape[-1] - prompt_tokens),
    )


def _summarize(rows: list[dict[str, Any]], tool_latencies: list[float]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"num_requests": len(rows)}
    if not rows:
        return aggregate
    aggregate["probe_all_ready_rate"] = (
        sum(bool(row["probe_all_ready"]) for row in rows) / len(rows)
    )
    aggregate["mean_probe_ready_count"] = statistics.mean(
        [float(row["probe_ready_count"]) for row in rows]
    )
    for key in [
        "probe_latency_ms",
        "probe_prompt_tokens",
        "probe_output_tokens",
        "native_generation_ms",
        "native_all_ready_ms",
    ]:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        if values:
            aggregate[f"mean_{key}"] = statistics.mean(values)
            aggregate[f"p50_{key}"] = _percentile(values, 0.50)
            aggregate[f"p90_{key}"] = _percentile(values, 0.90)
            aggregate[f"p95_{key}"] = _percentile(values, 0.95)
    native_rows = [row for row in rows if row.get("native_generation_ms") not in (None, "")]
    if native_rows:
        aggregate["native_rows"] = len(native_rows)
        aggregate["probe_beats_native_all_ready_rate"] = (
            sum(
                1
                for row in native_rows
                if row["probe_all_ready"]
                and row.get("native_all_ready_ms") not in (None, "")
                and float(row["probe_latency_ms"]) < float(row["native_all_ready_ms"])
            )
            / len(native_rows)
        )
        for tool_latency_ms in tool_latencies:
            key = int(tool_latency_ms)
            no_spec = []
            native = []
            probe = []
            for row in native_rows:
                generation_ms = float(row["native_generation_ms"])
                native_ready_ms = _maybe_float(row.get("native_all_ready_ms"))
                probe_ready_ms = float(row["probe_latency_ms"]) if row["probe_all_ready"] else None
                no_spec.append(generation_ms + tool_latency_ms)
                native.append(
                    _effective_latency(
                        generation_ms=generation_ms,
                        ready_ms=native_ready_ms,
                        tool_latency_ms=tool_latency_ms,
                    )
                )
                probe.append(
                    _effective_latency(
                        generation_ms=generation_ms,
                        ready_ms=probe_ready_ms,
                        tool_latency_ms=tool_latency_ms,
                    )
                )
            aggregate[f"tool{key}_mean_native_vs_probe_speedup"] = statistics.mean(
                p / n for p, n in zip(probe, native) if n > 0
            )
            aggregate[f"tool{key}_mean_native_vs_no_spec_speedup"] = statistics.mean(
                s / n for s, n in zip(no_spec, native) if n > 0
            )
            aggregate[f"tool{key}_mean_probe_vs_no_spec_speedup"] = statistics.mean(
                s / p for s, p in zip(no_spec, probe) if p > 0
            )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_model_name_or_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--native_requests_csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--tool_latencies_ms", default="100,300,500,1000,2000")
    parser.add_argument("--prompt_format", default="action_list", choices=["action_list", "json_array", "named_object"])
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    records = _load_multitool_records(args.input_path, args.limit)
    native_rows = _native_by_request(args.native_requests_csv)
    tool_latencies = _parse_float_list(args.tool_latencies_ms)

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
    decoded_rows: list[dict[str, Any]] = []
    for request_index, record in enumerate(records):
        messages = _format_prompt_for_style(record, prompt_format=args.prompt_format)
        probe = _run_probe(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
        )
        scores = [_call_score(probe.text, call) for call in record["calls"]]
        ready_count = sum(bool(score["ready"]) for score in scores)
        row: dict[str, Any] = {
            "request_index": request_index,
            "prompt_format": args.prompt_format,
            "num_calls": len(record["calls"]),
            "probe_all_ready": ready_count == len(record["calls"]),
            "probe_ready_count": ready_count,
            "probe_latency_ms": probe.latency_ms,
            "probe_prompt_tokens": probe.prompt_tokens,
            "probe_output_tokens": probe.output_tokens,
        }
        native = native_rows.get(request_index)
        if native is not None and native.get("final_all_ready") == "True":
            generation_ms = _maybe_float(native.get("generation_ms"))
            fraction = _maybe_float(native.get("dllm_all_stable_fraction"))
            row["native_generation_ms"] = generation_ms
            row["native_all_ready_fraction"] = fraction
            row["native_all_ready_ms"] = (
                generation_ms * fraction
                if generation_ms is not None and fraction is not None
                else None
            )
        request_rows.append(row)
        decoded_rows.append(
            {
                "request_index": request_index,
                "prompt": record["prompt"],
                "calls": record["calls"],
                "probe_text": probe.text,
                "scores": scores,
            }
        )
        print(
            json.dumps(
                {
                    "request_index": request_index,
                    "probe_ready_count": ready_count,
                    "probe_latency_ms": probe.latency_ms,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    aggregate = _summarize(request_rows, tool_latencies)
    aggregate.update(
        {
            "ar_model_name_or_path": args.ar_model_name_or_path,
            "input_path": args.input_path,
            "native_requests_csv": args.native_requests_csv,
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
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
