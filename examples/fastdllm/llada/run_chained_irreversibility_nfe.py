"""
Run isolated-vs-chained NFE ablations for request-varying irreversibility.

Run from repo root on a GPU server:
  source ~/.zshrc
  conda activate ~/miniconda3/envs/dllm

  python -u examples/fastdllm/llada/run_chained_irreversibility_nfe.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/chained_irreversibility_60.jsonl \
    --limit 30 \
    --high_steps 128 \
    --low_steps 16 \
    --max_new_tokens 48 \
    --block_size 24 \
    --use_cache prefix \
    --output_prefix artifacts/nfe_stage_ablation/chained_irrev_h128_l16

This script measures:
  A. isolated stage-local quality using oracle inputs
  B. chained final quality where structured fields from the irreversible stage
     are consumed by downstream prompts

The main diagnostic is propagation amplification:
  (all_high final success - low_stage_k final success)
  - (stage_k high local success - stage_k low local success)
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm

from generate_chained_irreversibility_prompts import CITY_TO_AIRPORT, PRIORITY_TO_SLA
from test_tool_prefetch_signals import _decode_generated, _normalize_inputs, _percentile


FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*[:=]\s*(.*?)\s*$")
STAGE_FIELDS = {
    0: ("city", "date", "priority"),
    1: ("airport", "sla", "date"),
    2: ("dispatch",),
}


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_records(path: str, limit: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "request" not in row or "targets" not in row or "hard_stage" not in row:
                raise ValueError(f"Unsupported chained record: {row}")
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = FIELD_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip().strip('"').strip("'").strip()
        value = value.rstrip(";,")
        fields[key] = value
    return fields


def _score_fields(
    *,
    text: str,
    expected: dict[str, str],
    required_fields: tuple[str, ...],
) -> dict[str, Any]:
    observed = _parse_fields(text)
    field_correct = {}
    for field in required_fields:
        field_correct[field] = _norm(observed.get(field, "")) == _norm(expected[field])
    return {
        "ready": all(field_correct.values()),
        "observed": observed,
        "field_correct": field_correct,
        "ready_count": sum(bool(value) for value in field_correct.values()),
        "required_count": len(required_fields),
    }


def _table_text() -> str:
    city_lines = [f"- {city}: {airport}" for city, airport in CITY_TO_AIRPORT.items()]
    priority_lines = [
        f"- {priority}: {sla}" for priority, sla in PRIORITY_TO_SLA.items()
    ]
    return (
        "City to airport table:\n"
        + "\n".join(city_lines)
        + "\n\nPriority to SLA table:\n"
        + "\n".join(priority_lines)
    )


def _stage0_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    content = (
        "Parse one travel dispatch request.\n"
        "Return only this format:\n"
        "CITY: <city>\n"
        "DATE: <yyyy-mm-dd>\n"
        "PRIORITY: <urgent|standard|economy>\n\n"
        f"Request: {record['request']}"
    )
    return [{"role": "user", "content": content}]


def _stage1_prompt(fields: dict[str, str]) -> list[dict[str, str]]:
    content = (
        "Lookup the airport and SLA for a parsed travel dispatch.\n"
        "Return only this format:\n"
        "AIRPORT: <airport_code>\n"
        "SLA: <S1|S2|S3>\n"
        "DATE: <yyyy-mm-dd>\n\n"
        f"{_table_text()}\n\n"
        f"CITY: {fields.get('city', 'UNKNOWN')}\n"
        f"DATE: {fields.get('date', 'UNKNOWN')}\n"
        f"PRIORITY: {fields.get('priority', 'UNKNOWN')}"
    )
    return [{"role": "user", "content": content}]


def _stage2_prompt(fields: dict[str, str]) -> list[dict[str, str]]:
    content = (
        "Finalize the dispatch code.\n"
        "The dispatch code is AIRPORT + '-' + DATE without dashes + '-' + SLA.\n"
        "Return only this format:\n"
        "DISPATCH: <airport>-<yyyymmdd>-<sla>\n\n"
        f"AIRPORT: {fields.get('airport', 'UNKNOWN')}\n"
        f"DATE: {fields.get('date', 'UNKNOWN')}\n"
        f"SLA: {fields.get('sla', 'UNKNOWN')}"
    )
    return [{"role": "user", "content": content}]


def _oracle_stage0(record: dict[str, Any]) -> dict[str, str]:
    targets = record["targets"]
    return {
        "city": str(targets["city"]),
        "date": str(targets["date"]),
        "priority": str(targets["priority"]),
    }


def _oracle_stage1(record: dict[str, Any]) -> dict[str, str]:
    targets = record["targets"]
    return {
        "airport": str(targets["airport"]),
        "sla": str(targets["sla"]),
        "date": str(targets["date"]),
    }


def _expected_for_stage(record: dict[str, Any], stage_index: int) -> dict[str, str]:
    targets = record["targets"]
    if stage_index == 0:
        return _oracle_stage0(record)
    if stage_index == 1:
        return _oracle_stage1(record)
    return {"dispatch": str(targets["dispatch"])}


def _merge_stage0(observed: dict[str, str]) -> dict[str, str]:
    return {
        "city": observed.get("city", "UNKNOWN"),
        "date": observed.get("date", "UNKNOWN"),
        "priority": observed.get("priority", "UNKNOWN"),
    }


def _merge_stage1(observed: dict[str, str]) -> dict[str, str]:
    return {
        "airport": observed.get("airport", "UNKNOWN"),
        "sla": observed.get("sla", "UNKNOWN"),
        "date": observed.get("date", "UNKNOWN"),
    }


def _variant_schedules() -> dict[str, set[int]]:
    return {
        "all_high": set(),
        "all_low": {0, 1, 2},
        "low_stage_0": {0},
        "low_stage_1": {1},
        "low_stage_2": {2},
    }


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str = "examples/fastdllm/llada/chained_irreversibility_60.jsonl"
    limit: int = 30
    seed: int = 42
    high_steps: int = 128
    low_steps: int = 16
    output_prefix: str = "artifacts/nfe_stage_ablation/chained_irrev_h128_l16"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path,
            "BASE_MODELS_DIR",
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    max_new_tokens: int = 48
    block_size: int = 24
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "prefix"
    threshold: float | None = None
    factor: float | None = None
    begin_suppress_tokens: list[int] | None = None


def _run_generation(
    *,
    sampler: Any,
    tokenizer: Any,
    sampler_config: SamplerConfig,
    messages: list[dict[str, str]],
    steps: int,
) -> tuple[str, float]:
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
    text = _decode_generated(
        tokenizer=tokenizer,
        sequence=outputs.sequences[0].tolist(),
        prompt_len=len(prompt_ids),
        max_new_tokens=sampler_config.max_new_tokens,
    )
    return text, generation_ms


def _run_isolated_stage(
    *,
    sampler: Any,
    tokenizer: Any,
    sampler_config: SamplerConfig,
    record: dict[str, Any],
    stage_index: int,
    steps: int,
) -> dict[str, Any]:
    if stage_index == 0:
        messages = _stage0_prompt(record)
    elif stage_index == 1:
        messages = _stage1_prompt(_oracle_stage0(record))
    else:
        messages = _stage2_prompt(_oracle_stage1(record))
    text, generation_ms = _run_generation(
        sampler=sampler,
        tokenizer=tokenizer,
        sampler_config=sampler_config,
        messages=messages,
        steps=steps,
    )
    score = _score_fields(
        text=text,
        expected=_expected_for_stage(record, stage_index),
        required_fields=STAGE_FIELDS[stage_index],
    )
    return {
        "stage_index": stage_index,
        "steps": steps,
        "mode": "isolated",
        "generation_ms": generation_ms,
        "text": text,
        **{key: value for key, value in score.items() if key != "observed"},
        "observed": json.dumps(score["observed"], sort_keys=True),
    }


def _run_chained_variant(
    *,
    sampler: Any,
    tokenizer: Any,
    sampler_config: SamplerConfig,
    record: dict[str, Any],
    variant: str,
    low_stages: set[int],
    high_steps: int,
    low_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hard_stage = int(record["hard_stage"])
    stage_rows = []
    generated_stage0: dict[str, str] | None = None
    generated_stage1: dict[str, str] | None = None

    for stage_index in range(3):
        steps = low_steps if stage_index in low_stages else high_steps
        if stage_index == 0:
            messages = _stage0_prompt(record)
        elif stage_index == 1:
            stage0_fields = (
                _merge_stage0(generated_stage0 or {})
                if hard_stage == 0
                else _oracle_stage0(record)
            )
            messages = _stage1_prompt(stage0_fields)
        else:
            stage1_fields = (
                _merge_stage1(generated_stage1 or {})
                if hard_stage in (0, 1)
                else _oracle_stage1(record)
            )
            messages = _stage2_prompt(stage1_fields)

        text, generation_ms = _run_generation(
            sampler=sampler,
            tokenizer=tokenizer,
            sampler_config=sampler_config,
            messages=messages,
            steps=steps,
        )
        score = _score_fields(
            text=text,
            expected=_expected_for_stage(record, stage_index),
            required_fields=STAGE_FIELDS[stage_index],
        )
        observed = score["observed"]
        if stage_index == 0:
            generated_stage0 = observed
        elif stage_index == 1:
            generated_stage1 = observed
        stage_rows.append(
            {
                "stage_index": stage_index,
                "steps": steps,
                "mode": "chained",
                "variant": variant,
                "generation_ms": generation_ms,
                "text": text,
                **{key: value for key, value in score.items() if key != "observed"},
                "observed": json.dumps(observed, sort_keys=True),
            }
        )

    final_success = bool(stage_rows[-1]["ready"])
    request_row = {
        "variant": variant,
        "hard_stage": hard_stage,
        "final_success": final_success,
        "stage_ready_count": sum(bool(row["ready"]) for row in stage_rows),
        "total_generation_ms": sum(float(row["generation_ms"]) for row in stage_rows),
    }
    return request_row, stage_rows


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def _summarize(
    *,
    isolated_rows: list[dict[str, Any]],
    chain_request_rows: list[dict[str, Any]],
    high_steps: int,
    low_steps: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "high_steps": high_steps,
        "low_steps": low_steps,
        "num_isolated_stage_rows": len(isolated_rows),
        "num_chain_variant_rows": len(chain_request_rows),
    }
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in chain_request_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    baseline = _mean_bool(by_variant.get("all_high", []), "final_success")
    aggregate["chain_all_high_success_rate"] = baseline
    for variant, rows in sorted(by_variant.items()):
        rate = _mean_bool(rows, "final_success")
        aggregate[f"chain_{variant}_success_rate"] = rate
        aggregate[f"chain_{variant}_drop_vs_all_high"] = baseline - rate
        aggregate[f"chain_{variant}_mean_generation_ms"] = statistics.mean(
            [float(row["total_generation_ms"]) for row in rows]
        )
        aggregate[f"chain_{variant}_p95_generation_ms"] = _percentile(
            [float(row["total_generation_ms"]) for row in rows],
            0.95,
        )
        for hard_stage in range(3):
            subset = [row for row in rows if int(row["hard_stage"]) == hard_stage]
            if subset:
                key = f"chain_{variant}_hard{hard_stage}"
                aggregate[f"{key}_success_rate"] = _mean_bool(subset, "final_success")
                base_subset = [
                    row
                    for row in by_variant.get("all_high", [])
                    if int(row["hard_stage"]) == hard_stage
                ]
                if base_subset:
                    aggregate[f"{key}_drop_vs_all_high"] = (
                        _mean_bool(base_subset, "final_success")
                        - aggregate[f"{key}_success_rate"]
                    )
    baseline_ms = aggregate.get("chain_all_high_mean_generation_ms")
    if baseline_ms:
        for variant in by_variant:
            mean_ms = aggregate.get(f"chain_{variant}_mean_generation_ms")
            if mean_ms:
                aggregate[f"chain_{variant}_speedup_vs_all_high"] = baseline_ms / mean_ms

    for stage_index in range(3):
        for steps in (high_steps, low_steps):
            rows = [
                row
                for row in isolated_rows
                if int(row["stage_index"]) == stage_index and int(row["steps"]) == steps
            ]
            key = f"isolated_stage{stage_index}_steps{steps}"
            aggregate[f"{key}_ready_rate"] = _mean_bool(rows, "ready")
            aggregate[f"{key}_mean_generation_ms"] = statistics.mean(
                [float(row["generation_ms"]) for row in rows]
            )

    for stage_index in range(3):
        high_ready = aggregate[f"isolated_stage{stage_index}_steps{high_steps}_ready_rate"]
        low_ready = aggregate[f"isolated_stage{stage_index}_steps{low_steps}_ready_rate"]
        isolated_drop = high_ready - low_ready
        chain_drop = aggregate.get(f"chain_low_stage_{stage_index}_drop_vs_all_high")
        if chain_drop is not None:
            aggregate[f"stage{stage_index}_isolated_drop"] = isolated_drop
            aggregate[f"stage{stage_index}_chain_drop"] = chain_drop
            aggregate[f"stage{stage_index}_propagation_amplification"] = (
                chain_drop - isolated_drop
            )
        for hard_stage in range(3):
            high_rows = [
                row
                for row in isolated_rows
                if int(row["stage_index"]) == stage_index
                and int(row["steps"]) == high_steps
                and int(row["hard_stage"]) == hard_stage
            ]
            low_rows = [
                row
                for row in isolated_rows
                if int(row["stage_index"]) == stage_index
                and int(row["steps"]) == low_steps
                and int(row["hard_stage"]) == hard_stage
            ]
            chain_hard_drop = aggregate.get(
                f"chain_low_stage_{stage_index}_hard{hard_stage}_drop_vs_all_high"
            )
            if high_rows and low_rows and chain_hard_drop is not None:
                hard_isolated_drop = _mean_bool(high_rows, "ready") - _mean_bool(
                    low_rows,
                    "ready",
                )
                aggregate[
                    f"stage{stage_index}_hard{hard_stage}_isolated_drop"
                ] = hard_isolated_drop
                aggregate[f"stage{stage_index}_hard{hard_stage}_chain_drop"] = (
                    chain_hard_drop
                )
                aggregate[
                    f"stage{stage_index}_hard{hard_stage}_propagation_amplification"
                ] = chain_hard_drop - hard_isolated_drop
    return aggregate


def main() -> None:
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)
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

    isolated_rows: list[dict[str, Any]] = []
    chain_request_rows: list[dict[str, Any]] = []
    chain_stage_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []

    for request_index, record in enumerate(records):
        for stage_index in range(3):
            for steps in (script_args.high_steps, script_args.low_steps):
                row = _run_isolated_stage(
                    sampler=sampler,
                    tokenizer=tokenizer,
                    sampler_config=sampler_config,
                    record=record,
                    stage_index=stage_index,
                    steps=steps,
                )
                row.update(
                    {
                        "request_index": request_index,
                        "hard_stage": int(record["hard_stage"]),
                        "request": record["request"],
                    }
                )
                isolated_rows.append({k: v for k, v in row.items() if k != "text"})
                decoded_rows.append({**row, "request_index": request_index})

        for variant, low_stages in _variant_schedules().items():
            request_row, stage_rows = _run_chained_variant(
                sampler=sampler,
                tokenizer=tokenizer,
                sampler_config=sampler_config,
                record=record,
                variant=variant,
                low_stages=low_stages,
                high_steps=script_args.high_steps,
                low_steps=script_args.low_steps,
            )
            request_row.update(
                {
                    "request_index": request_index,
                    "request": record["request"],
                    "targets": json.dumps(record["targets"], sort_keys=True),
                }
            )
            chain_request_rows.append(request_row)
            for stage_row in stage_rows:
                full_stage_row = {
                    **stage_row,
                    "request_index": request_index,
                    "hard_stage": int(record["hard_stage"]),
                    "request": record["request"],
                }
                chain_stage_rows.append(
                    {k: v for k, v in full_stage_row.items() if k != "text"}
                )
                decoded_rows.append(full_stage_row)
            print(
                json.dumps(
                    {
                        "request_index": request_index,
                        "hard_stage": int(record["hard_stage"]),
                        "variant": variant,
                        "final_success": request_row["final_success"],
                        "stage_ready_count": request_row["stage_ready_count"],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    aggregate = _summarize(
        isolated_rows=isolated_rows,
        chain_request_rows=chain_request_rows,
        high_steps=script_args.high_steps,
        low_steps=script_args.low_steps,
    )
    aggregate.update(
        {
            "input_path": script_args.input_path,
            "model_name_or_path": script_args.model_name_or_path,
            "limit": script_args.limit,
            "max_new_tokens": sampler_config.max_new_tokens,
            "block_size": sampler_config.block_size,
            "use_cache": sampler_config.use_cache,
        }
    )

    prefix = Path(script_args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    isolated_path = prefix.with_name(prefix.name + "_isolated_stages.csv")
    requests_path = prefix.with_name(prefix.name + "_chain_requests.csv")
    stages_path = prefix.with_name(prefix.name + "_chain_stages.csv")
    decoded_path = prefix.with_name(prefix.name + "_decoded.jsonl")
    summary_path.write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "chain_requests": chain_request_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(isolated_path, isolated_rows)
    _write_csv(requests_path, chain_request_rows)
    _write_csv(stages_path, chain_stage_rows)
    with decoded_path.open("w", encoding="utf-8") as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Saved summary: {summary_path}")
    print(f"Saved isolated stage CSV: {isolated_path}")
    print(f"Saved chain request CSV: {requests_path}")
    print(f"Saved chain stage CSV: {stages_path}")
    print(f"Saved decoded JSONL: {decoded_path}")
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
