"""
Run a stage-level NFE ablation on multi-tool LLaDA workflow prompts.

Run from repo root on a GPU server:
  source ~/.zshrc
  conda activate ~/miniconda3/envs/dllm

  # If already inside an allocated GPU node:
  python -u examples/fastdllm/llada/run_multitool_nfe_stage_ablation.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/multitool_prefetch_prompts_3call_120.jsonl \
    --limit 20 \
    --high_steps 128 \
    --low_steps 32 \
    --max_new_tokens 64 \
    --block_size 32 \
    --use_cache prefix \
    --output_prefix artifacts/nfe_stage_ablation/multitool_3call_h128_l32

  # If submitting through Slurm from a login node:
  srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 \
    --time=03:00:00 python -u \
    examples/fastdllm/llada/run_multitool_nfe_stage_ablation.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/multitool_prefetch_prompts_3call_120.jsonl \
    --limit 20 \
    --high_steps 128 \
    --low_steps 32 \
    --max_new_tokens 64 \
    --block_size 32 \
    --use_cache prefix \
    --output_prefix artifacts/nfe_stage_ablation/multitool_3call_h128_l32

Do not pass --threshold or --factor for this ablation. In threshold/factor
mode the sampler ignores the fixed per-step transfer quota, so the --steps
argument is not a clean NFE knob.
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

from test_multitool_prefetch_signals import _call_score, _load_multitool_records
from test_tool_prefetch_signals import _decode_generated, _normalize_inputs, _percentile


TOOL_LINES = [
    "- flight_search(origin, destination, date)",
    "- weather(location, date)",
    "- calendar_api(calendar, start_date, end_date, keyword)",
    "- crm_api(company, role, city)",
]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stage_prompt(
    record: dict[str, Any],
    *,
    stage_index: int,
    num_stages: int,
) -> list[dict[str, str]]:
    content = (
        "You are executing one stage of a multi-tool workflow. "
        f"The user request requires {num_stages} independent read-only actions. "
        f"Return only action {stage_index + 1}, using the order in the user request. "
        "Do not include any other action.\n"
        "Available tools:\n"
        + "\n".join(TOOL_LINES)
        + "\n\n"
        "Return only this format:\n"
        f"ACTION {stage_index + 1}:\n"
        "TOOL: <tool_name>\n"
        "ARGS: key=value; key=value\n\n"
        f"User request: {record['prompt']}"
    )
    return [{"role": "user", "content": content}]


def _variant_schedules(num_stages: int) -> dict[str, set[int]]:
    schedules = {"all_high": set(), "all_low": set(range(num_stages))}
    for stage_index in range(num_stages):
        schedules[f"low_stage_{stage_index}"] = {stage_index}
    return schedules


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str = "examples/fastdllm/llada/multitool_prefetch_prompts_3call_120.jsonl"
    limit: int = 10
    seed: int = 42
    high_steps: int = 128
    low_steps: int = 32
    output_prefix: str = "artifacts/nfe_stage_ablation/multitool_3call_h128_l32"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path,
            "BASE_MODELS_DIR",
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    max_new_tokens: int = 64
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "prefix"
    threshold: float | None = None
    factor: float | None = None
    begin_suppress_tokens: list[int] | None = None


def _run_stage(
    *,
    sampler: dllm.pipelines.fastdllm.llada.FastdLLMLLaDASampler,
    tokenizer: Any,
    sampler_config: SamplerConfig,
    record: dict[str, Any],
    call: dict[str, Any],
    stage_index: int,
    num_stages: int,
    steps: int,
) -> dict[str, Any]:
    messages = _stage_prompt(
        record,
        stage_index=stage_index,
        num_stages=num_stages,
    )
    inputs = tokenizer.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = _normalize_inputs(inputs)[0]

    _sync_device(sampler.model.device)
    start = time.perf_counter()
    outputs = sampler.sample(
        inputs,
        config=sampler_config,
        return_dict=True,
        steps=steps,
    )
    _sync_device(sampler.model.device)
    generation_ms = (time.perf_counter() - start) * 1000.0

    final_sequence = outputs.sequences[0].tolist()
    final_text = _decode_generated(
        tokenizer=tokenizer,
        sequence=final_sequence,
        prompt_len=len(prompt_ids),
        max_new_tokens=sampler_config.max_new_tokens,
    )
    score = _call_score(final_text, call)
    return {
        "stage_index": stage_index,
        "steps": steps,
        "generation_ms": generation_ms,
        "final_text": final_text,
        "ready": bool(score["ready"]),
        "tool_correct": bool(score["tool_correct"]),
        "args_correct": bool(score["args_correct"]),
        "detected_tool": score["detected_tool"],
        "target_tool": call["tool"],
        "target_args": json.dumps(call["args"], sort_keys=True),
    }


def _summarize(
    *,
    request_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    high_steps: int,
    low_steps: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "num_variant_requests": len(request_rows),
        "num_stage_generations": len(stage_rows),
        "high_steps": high_steps,
        "low_steps": low_steps,
    }
    baseline_rows = [row for row in request_rows if row["variant"] == "all_high"]
    baseline_rate = (
        sum(bool(row["final_all_ready"]) for row in baseline_rows)
        / max(len(baseline_rows), 1)
    )
    aggregate["all_high_final_all_ready_rate"] = baseline_rate

    for variant in sorted({str(row["variant"]) for row in request_rows}):
        rows = [row for row in request_rows if row["variant"] == variant]
        final_rate = sum(bool(row["final_all_ready"]) for row in rows) / max(len(rows), 1)
        ready_counts = [float(row["final_ready_count"]) for row in rows]
        generation_ms_values = [float(row["total_generation_ms"]) for row in rows]
        aggregate[f"{variant}_num_requests"] = len(rows)
        aggregate[f"{variant}_final_all_ready_rate"] = final_rate
        aggregate[f"{variant}_final_all_ready_delta_vs_all_high"] = (
            final_rate - baseline_rate
        )
        aggregate[f"{variant}_mean_final_ready_count"] = statistics.mean(ready_counts)
        aggregate[f"{variant}_mean_generation_ms"] = statistics.mean(
            generation_ms_values
        )
        aggregate[f"{variant}_p95_generation_ms"] = _percentile(generation_ms_values, 0.95)
    baseline_generation = aggregate.get("all_high_mean_generation_ms")
    if baseline_generation:
        for variant in sorted({str(row["variant"]) for row in request_rows}):
            mean_generation = aggregate.get(f"{variant}_mean_generation_ms")
            if mean_generation:
                aggregate[f"{variant}_latency_speedup_vs_all_high"] = (
                    baseline_generation / mean_generation
                )

    for stage_index in sorted({int(row["stage_index"]) for row in stage_rows}):
        for steps in (high_steps, low_steps):
            rows = [
                row
                for row in stage_rows
                if int(row["stage_index"]) == stage_index and int(row["steps"]) == steps
            ]
            if not rows:
                continue
            tag = f"stage{stage_index}_steps{steps}"
            aggregate[f"{tag}_ready_rate"] = (
                sum(bool(row["ready"]) for row in rows) / len(rows)
            )
            aggregate[f"{tag}_tool_correct_rate"] = (
                sum(bool(row["tool_correct"]) for row in rows) / len(rows)
            )
            aggregate[f"{tag}_args_correct_rate"] = (
                sum(bool(row["args_correct"]) for row in rows) / len(rows)
            )
            aggregate[f"{tag}_mean_generation_ms"] = statistics.mean(
                [float(row["generation_ms"]) for row in rows]
            )
    return aggregate


def main() -> None:
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

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
    stage_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []

    for request_index, record in enumerate(records):
        calls = list(record["calls"])
        num_stages = len(calls)
        stage_results: dict[tuple[int, int], dict[str, Any]] = {}

        for stage_index, call in enumerate(calls):
            for steps in (script_args.high_steps, script_args.low_steps):
                result = _run_stage(
                    sampler=sampler,
                    tokenizer=tokenizer,
                    sampler_config=sampler_config,
                    record=record,
                    call=call,
                    stage_index=stage_index,
                    num_stages=num_stages,
                    steps=steps,
                )
                result.update(
                    {
                        "request_index": request_index,
                        "prompt": record["prompt"],
                        "num_stages": num_stages,
                    }
                )
                stage_results[(stage_index, steps)] = result
                stage_rows.append(
                    {key: value for key, value in result.items() if key != "final_text"}
                )

        for variant, low_stage_indexes in _variant_schedules(num_stages).items():
            selected = []
            for stage_index in range(num_stages):
                steps = (
                    script_args.low_steps
                    if stage_index in low_stage_indexes
                    else script_args.high_steps
                )
                selected.append(stage_results[(stage_index, steps)])

            final_ready = [bool(row["ready"]) for row in selected]
            total_generation_ms = sum(float(row["generation_ms"]) for row in selected)
            request_row = {
                "request_index": request_index,
                "variant": variant,
                "prompt": record["prompt"],
                "num_stages": num_stages,
                "high_steps": script_args.high_steps,
                "low_steps": script_args.low_steps,
                "nfe_schedule": ",".join(str(row["steps"]) for row in selected),
                "final_all_ready": all(final_ready),
                "final_ready_count": sum(final_ready),
                "total_generation_ms": total_generation_ms,
            }
            request_rows.append(request_row)
            decoded_rows.append(
                {
                    **request_row,
                    "stages": [
                        {
                            "stage_index": row["stage_index"],
                            "steps": row["steps"],
                            "ready": row["ready"],
                            "target_tool": row["target_tool"],
                            "target_args": row["target_args"],
                            "final_text": row["final_text"],
                        }
                        for row in selected
                    ],
                }
            )

            print(
                json.dumps(
                    {
                        "request_index": request_index,
                        "variant": variant,
                        "nfe_schedule": request_row["nfe_schedule"],
                        "final_ready_count": request_row["final_ready_count"],
                        "final_all_ready": request_row["final_all_ready"],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    requests_path = prefix.with_name(prefix.name + "_requests.csv")
    stages_path = prefix.with_name(prefix.name + "_stages.csv")
    decoded_path = prefix.with_name(prefix.name + "_decoded.jsonl")

    aggregate = _summarize(
        request_rows=request_rows,
        stage_rows=stage_rows,
        high_steps=script_args.high_steps,
        low_steps=script_args.low_steps,
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
    print("Aggregate:")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
