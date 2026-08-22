"""Run deterministic Fast LLaDA preemption-state structural validation.

Run on a GPU server from the repository root:

    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python -u examples/fastdllm/llada/run_preemption_state_killtest.py \
      --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
      --cache_modes none,prefix,dual \
      --schedule_modes threshold,quota \
      --output_dir artifacts/preemption_state/structural_killtest

This script validates the real ``FastdLLMLLaDASampler.sample`` path. It does
not implement a preemption policy and does not report GPU offload costs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers

import dllm
from dllm.pipelines.fastdllm.llada import (
    FastdLLMLLaDAConfig,
    FastdLLMLLaDAPaused,
    FastdLLMLLaDAResumeState,
    FastdLLMLLaDASampler,
    FastdLLMLLaDASamplerConfig,
)


@dataclass
class StepSnapshot:
    block: int
    inner_step: int
    x: torch.Tensor
    transfer_index: torch.Tensor


COMPONENTS = (
    "x",
    "attention_mask",
    "current_schedule",
    "block_index",
    "inner_step",
    "past_key_values",
    "replace_position",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return int(value.numel() * value.element_size())


def _pkv_bytes(
    values: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> int:
    if values is None:
        return 0
    return sum(_tensor_bytes(tensor) for layer in values for tensor in layer)


def _component_bytes(state: FastdLLMLLaDAResumeState) -> dict[str, int]:
    return {
        "x": _tensor_bytes(state.x),
        "attention_mask": _tensor_bytes(state.attention_mask),
        "current_schedule": _tensor_bytes(state.num_transfer_tokens),
        "block_index": 0 if state.block_index is None else 8,
        "inner_step": 0 if state.inner_step is None else 8,
        "past_key_values": _pkv_bytes(state.past_key_values),
        "replace_position": _tensor_bytes(state.replace_position),
    }


def _to_cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(device="cpu").clone()


def _to_cpu_pkv(
    values: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    if values is None:
        return None
    return [tuple(_to_cpu_tensor(tensor) for tensor in layer) for layer in values]  # type: ignore[list-item]


def export_state(
    state: FastdLLMLLaDAResumeState,
    *,
    remove: Iterable[str] = (),
) -> FastdLLMLLaDAResumeState:
    """Materialize a detached CPU checkpoint with selected fields removed."""

    removed = set(remove)
    if "x" in removed:
        raise ValueError("x removal is not a valid resume checkpoint")
    return FastdLLMLLaDAResumeState(
        x=_to_cpu_tensor(state.x),  # type: ignore[arg-type]
        prompt_lens=tuple(state.prompt_lens),
        attention_mask=(
            None
            if "attention_mask" in removed
            else _to_cpu_tensor(state.attention_mask)
        ),
        block_index=(None if "block_index" in removed else state.block_index),
        inner_step=(None if "inner_step" in removed else state.inner_step),
        num_transfer_tokens=(
            None
            if "current_schedule" in removed
            else _to_cpu_tensor(state.num_transfer_tokens)
        ),
        past_key_values=(
            None
            if "past_key_values" in removed
            else _to_cpu_pkv(state.past_key_values)
        ),
        replace_position=(
            None
            if "replace_position" in removed
            else _to_cpu_tensor(state.replace_position)
        ),
        cache_mode=state.cache_mode,
    )


def _observer(target: list[StepSnapshot]):
    def observe(record: dict[str, Any]) -> None:
        target.append(
            StepSnapshot(
                block=int(record["block_index"]),
                inner_step=int(record["inner_step"]),
                x=record["x"].detach().cpu().clone(),
                transfer_index=record["transfer_index"].detach().cpu().clone(),
            )
        )

    return observe


def _output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    return output.sequences


def _trace_equal(
    expected: list[StepSnapshot], observed: list[StepSnapshot]
) -> tuple[bool, str]:
    if len(expected) != len(observed):
        return False, f"tail length expected={len(expected)} observed={len(observed)}"
    for index, (left, right) in enumerate(zip(expected, observed)):
        if (left.block, left.inner_step) != (right.block, right.inner_step):
            return False, f"step {index}: progress metadata differs"
        if not torch.equal(left.transfer_index, right.transfer_index):
            return False, f"step {index}: transfer positions differ"
        if not torch.equal(left.x, right.x):
            return False, f"step {index}: canvas differs"
    return True, ""


def _select_checkpoint_indices(
    trace: list[StepSnapshot], max_checkpoints: int
) -> list[int]:
    if not trace:
        return []
    n = len(trace)
    required = {0, n // 10, n // 4, n // 2, (3 * n) // 4, (9 * n) // 10, n - 1}
    boundaries = [i for i, step in enumerate(trace) if step.inner_step == 0]
    inners = [i for i, step in enumerate(trace) if step.inner_step in (1, 2, 3)]
    required.update(boundaries)
    required.update(inners[: max(0, max_checkpoints // 2)])
    selected = sorted(i for i in required if 0 <= i < n)
    if len(selected) <= max_checkpoints:
        return selected
    anchors = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    extras = [i for i in selected if i not in anchors]
    return sorted(anchors + extras[: max(0, max_checkpoints - len(anchors))])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _config(args: argparse.Namespace, cache_mode: str, schedule_mode: str):
    return FastdLLMLLaDASamplerConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=0.0,
        remasking="low_confidence",
        stochastic_transfer=False,
        threshold=(args.threshold if schedule_mode == "threshold" else None),
        factor=None,
        use_cache=cache_mode,
        return_dict=False,
    )


def _run_baseline(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
) -> tuple[torch.Tensor, list[StepSnapshot]]:
    trace: list[StepSnapshot] = []
    output = sampler.sample(inputs, config=config, step_observer=_observer(trace))
    return _output_tensor(output).detach().cpu().clone(), trace


def _capture_checkpoint(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
    coordinate: tuple[int, int],
) -> FastdLLMLLaDAResumeState:
    try:
        sampler.sample(inputs, config=config, pause_at=coordinate)
    except FastdLLMLLaDAPaused as paused:
        return paused.state
    raise RuntimeError(f"sampler did not pause at {coordinate}")


def _resume_and_compare(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
    state: FastdLLMLLaDAResumeState,
    expected_final: torch.Tensor,
    expected_tail: list[StepSnapshot],
) -> tuple[bool, bool, str]:
    trace: list[StepSnapshot] = []
    try:
        output = sampler.sample(
            inputs,
            config=config,
            resume_state=state,
            step_observer=_observer(trace),
        )
    except Exception as error:
        return False, False, f"resume error: {type(error).__name__}: {error}"
    exact_intermediate, note = _trace_equal(expected_tail, trace)
    exact_final = torch.equal(expected_final, _output_tensor(output).detach().cpu())
    if not exact_final and not note:
        note = "final canvas differs"
    return exact_intermediate, exact_final, note


def _variants(cache_mode: str, schedule_mode: str) -> dict[str, tuple[str, ...]]:
    variants: dict[str, tuple[str, ...]] = {
        "full": (),
        "no_attention_mask": ("attention_mask",),
        "no_current_schedule": ("current_schedule",),
        "no_block_index": ("block_index",),
        "no_inner_step": ("inner_step",),
    }
    if cache_mode != "none":
        variants["no_cache_state"] = ("past_key_values",)
    if cache_mode == "dual":
        variants["no_replace_position"] = ("replace_position",)
    semantic_removed = [
        "attention_mask",
        "current_schedule",
        "block_index",
        "past_key_values",
        "replace_position",
    ]
    if schedule_mode == "quota":
        semantic_removed.append("inner_step")
    variants["semantic_only"] = tuple(semantic_removed)
    return variants


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Explain why request preemption is hard for diffusion language models.",
    )
    parser.add_argument("--cache_modes", default="none,prefix,dual")
    parser.add_argument("--schedule_modes", default="threshold,quota")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_checkpoints", type=int, default=12)
    parser.add_argument(
        "--output_dir",
        default="artifacts/preemption_state/structural_killtest",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    transformers.set_seed(args.seed)
    model_path = dllm.utils.resolve_with_base_env(
        args.model_name_or_path, "BASE_MODELS_DIR"
    )
    model_config = FastdLLMLLaDAConfig.from_pretrained(model_path)
    model = dllm.utils.get_model(
        model_name_or_path=model_path,
        config=model_config,
    ).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=model_path)
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": args.prompt}]],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.as_tensor(inputs, dtype=torch.long)
    inputs = inputs.to(model.device)
    sampler = FastdLLMLLaDASampler(model=model, tokenizer=tokenizer)

    exact_rows: list[dict[str, Any]] = []
    captured: dict[tuple[str, str, int], FastdLLMLLaDAResumeState] = {}
    mode_failures: list[str] = []
    baseline_lengths: dict[tuple[str, str], int] = {}

    cache_modes = [item.strip() for item in args.cache_modes.split(",") if item.strip()]
    schedule_modes = [
        item.strip() for item in args.schedule_modes.split(",") if item.strip()
    ]
    for cache_mode in cache_modes:
        for schedule_mode in schedule_modes:
            print(f"baseline cache={cache_mode} schedule={schedule_mode}", flush=True)
            config = _config(args, cache_mode, schedule_mode)
            try:
                baseline_final, baseline_trace = _run_baseline(sampler, inputs, config)
            except Exception as error:
                mode_failures.append(
                    f"{cache_mode}/{schedule_mode}: {type(error).__name__}: {error}"
                )
                continue
            baseline_lengths[(cache_mode, schedule_mode)] = len(baseline_trace)
            indices = _select_checkpoint_indices(
                baseline_trace, max_checkpoints=args.max_checkpoints
            )
            for checkpoint_index in indices:
                expected = baseline_trace[checkpoint_index]
                coordinate = (expected.block, expected.inner_step)
                progress = checkpoint_index / max(1, len(baseline_trace))
                print(
                    f"  checkpoint progress={progress:.3f} block={coordinate[0]} "
                    f"inner={coordinate[1]}",
                    flush=True,
                )
                raw_state = _capture_checkpoint(sampler, inputs, config, coordinate)
                captured[(cache_mode, schedule_mode, checkpoint_index)] = raw_state
                full_bytes = _component_bytes(raw_state)

                for variant, removed in _variants(cache_mode, schedule_mode).items():
                    _sync(model.device)
                    export_start = time.perf_counter()
                    checkpoint = export_state(raw_state, remove=removed)
                    _sync(model.device)
                    export_ms = (time.perf_counter() - export_start) * 1000.0
                    checkpoint_bytes = sum(_component_bytes(checkpoint).values())
                    exact_intermediate, exact_final, note = _resume_and_compare(
                        sampler,
                        inputs,
                        config,
                        checkpoint,
                        baseline_final,
                        baseline_trace[checkpoint_index:],
                    )
                    exact_rows.append(
                        {
                            "cache_mode": cache_mode,
                            "schedule_mode": schedule_mode,
                            "checkpoint_progress": progress,
                            "checkpoint_index": checkpoint_index,
                            "block": coordinate[0],
                            "inner_step": coordinate[1],
                            "checkpoint_type": (
                                "block_boundary" if coordinate[1] == 0 else "inner_step"
                            ),
                            "variant": variant,
                            "removed_components": ";".join(removed),
                            "exact_intermediate": exact_intermediate,
                            "exact_final": exact_final,
                            "checkpoint_bytes": checkpoint_bytes,
                            "full_checkpoint_bytes": sum(full_bytes.values()),
                            "export_to_cpu_ms": export_ms,
                            "notes": note,
                        }
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "exact_resume_results.csv", exact_rows)

    results_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {
        (
            row["cache_mode"],
            row["schedule_mode"],
            int(row["checkpoint_index"]),
            row["variant"],
        ): row
        for row in exact_rows
    }
    inventory_rows: list[dict[str, Any]] = []
    required_by_mode: dict[tuple[str, str], set[str]] = defaultdict(lambda: {"x"})
    for key, state in captured.items():
        cache_mode, schedule_mode, checkpoint_index = key
        full_row = results_by_key[(cache_mode, schedule_mode, checkpoint_index, "full")]
        progress = float(full_row["checkpoint_progress"])
        sizes = _component_bytes(state)
        variant_names = {
            "attention_mask": "no_attention_mask",
            "current_schedule": "no_current_schedule",
            "block_index": "no_block_index",
            "inner_step": "no_inner_step",
            "past_key_values": "no_cache_state",
            "replace_position": "no_replace_position",
        }
        for component in COMPONENTS:
            if component == "x":
                required = True
                reconstructable = False
                evidence = "semantic canvas; removal is not a valid continuation"
            elif sizes[component] == 0:
                required = False
                reconstructable = True
                evidence = "component absent at this checkpoint"
            elif variant_names[component] not in _variants(cache_mode, schedule_mode):
                required = False
                reconstructable = True
                evidence = "component not used by this cache mode"
            else:
                row = results_by_key[
                    (
                        cache_mode,
                        schedule_mode,
                        checkpoint_index,
                        variant_names[component],
                    )
                ]
                reconstructable = bool(row["exact_intermediate"] and row["exact_final"])
                required = not reconstructable
                evidence = (
                    "removal preserved exact trajectory"
                    if reconstructable
                    else row["notes"] or "removal changed exact trajectory"
                )
            if required:
                required_by_mode[(cache_mode, schedule_mode)].add(component)
            inventory_rows.append(
                {
                    "cache_mode": cache_mode,
                    "schedule_mode": schedule_mode,
                    "progress": progress,
                    "state_component": component,
                    "required": required,
                    "reconstructable": reconstructable,
                    "bytes": sizes[component],
                    "evidence": evidence,
                }
            )
        inventory_rows.append(
            {
                "cache_mode": cache_mode,
                "schedule_mode": schedule_mode,
                "progress": progress,
                "state_component": "request_metadata_and_config",
                "required": False,
                "reconstructable": True,
                "bytes": 0,
                "evidence": "reused from immutable original request/config",
            }
        )
    _write_csv(output_dir / "minimal_state_inventory.csv", inventory_rows)

    scaling_rows: list[dict[str, Any]] = []
    for key, state in captured.items():
        cache_mode, schedule_mode, checkpoint_index = key
        row = results_by_key[(cache_mode, schedule_mode, checkpoint_index, "full")]
        sizes = _component_bytes(state)
        required = required_by_mode[(cache_mode, schedule_mode)]
        semantic_bytes = sum(
            size
            for component, size in sizes.items()
            if component in required
        )
        performance_bytes = (
            0
            if "past_key_values" in required
            else sizes["past_key_values"]
        )
        scaling_rows.append(
            {
                "cache_mode": cache_mode,
                "schedule_mode": schedule_mode,
                "prompt_length": int(inputs.shape[-1]),
                "generation_length": args.max_new_tokens,
                "block_size": args.block_size,
                "progress": row["checkpoint_progress"],
                "semantic_bytes": semantic_bytes,
                "performance_bytes": performance_bytes,
                "total_bytes": sum(sizes.values()),
            }
        )
    _write_csv(output_dir / "state_scaling.csv", scaling_rows)

    full_rows = [row for row in exact_rows if row["variant"] == "full"]
    full_exact = bool(full_rows) and all(
        row["exact_intermediate"] and row["exact_final"] for row in full_rows
    )
    if mode_failures or not full_exact:
        judgment = "NO-GO"
    else:
        judgment = "CONDITIONAL GO"
    report_lines = [
        "# Preemption State Structural Kill Test",
        "",
        "## CONFIRMED",
        "",
        f"- Real-sampler full-state exact resume passed: {full_exact}.",
        f"- Exact checkpoint rows: {len(full_rows)}.",
        "- Minimal-state conclusions are based on component-removal ablations.",
        "",
        "## NEGATIVE RESULT",
        "",
        "- No recovery strategy or scheduling crossover has been established.",
        "- Python CPU export timing is not treated as GPU offload cost.",
        "",
        "## UNKNOWN / NOT YET TESTED",
        "",
        "- GPU-to-CPU and CPU-to-GPU transfer distributions.",
        "- Cache rebuild versus restart costs across progress.",
        "- Prompt/generation/block-size scaling beyond this run.",
        "- Strategy winner and block-boundary waiting trade-off.",
    ]
    if mode_failures:
        report_lines.extend(["", "## MODE FAILURES", ""])
        report_lines.extend(f"- {failure}" for failure in mode_failures)
    report_lines.extend(["", judgment])
    (output_dir / "preemption_state_killtest.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "model": args.model_name_or_path,
        "baseline_steps": {
            f"{cache}/{schedule}": steps
            for (cache, schedule), steps in baseline_lengths.items()
        },
        "mode_failures": mode_failures,
        "full_exact": full_exact,
        "judgment": judgment,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
