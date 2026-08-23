"""Benchmark matched Fast LLaDA preemption recovery strategies.

Run the current-shape kill test first:

    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python -u examples/fastdllm/llada/benchmark_preemption_recovery.py \
      --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
      --request_shapes 25:128 \
      --cache_modes prefix,dual \
      --repetitions 10 \
      --output_dir artifacts/preemption_state/recovery_killtest

The script uses the production sampler pause/resume path. It characterizes
FULL OFFLOAD, SEMANTIC-ONLY, DROP, and KEEP; it does not schedule requests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

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


@dataclass(frozen=True)
class StepPoint:
    index: int
    block: int
    inner_step: int
    progress: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.numel() * tensor.element_size())


def _pkv_bytes(
    values: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> int:
    if values is None:
        return 0
    return sum(_tensor_bytes(tensor) for layer in values for tensor in layer)


def _state_bytes(state: FastdLLMLLaDAResumeState) -> int:
    return (
        _tensor_bytes(state.x)
        + _tensor_bytes(state.attention_mask)
        + _tensor_bytes(state.num_transfer_tokens)
        + _pkv_bytes(state.past_key_values)
        + _tensor_bytes(state.replace_position)
        + (0 if state.block_index is None else 8)
        + (0 if state.inner_step is None else 8)
    )


def _semantic_state(
    state: FastdLLMLLaDAResumeState,
    schedule_mode: str,
) -> FastdLLMLLaDAResumeState:
    at_boundary = int(state.inner_step or 0) == 0
    return FastdLLMLLaDAResumeState(
        x=state.x,
        prompt_lens=tuple(state.prompt_lens),
        block_index=None,
        inner_step=(
            int(state.inner_step or 0)
            if schedule_mode == "threshold" and not at_boundary
            else None
        ),
        attention_mask=None,
        num_transfer_tokens=None,
        past_key_values=None,
        replace_position=None,
        cache_mode=state.cache_mode,
    )


def _map_tensors(
    state: FastdLLMLLaDAResumeState,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> FastdLLMLLaDAResumeState:
    def one(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else transform(value)

    pkv = None
    if state.past_key_values is not None:
        pkv = [tuple(transform(tensor) for tensor in layer) for layer in state.past_key_values]  # type: ignore[list-item]
    return FastdLLMLLaDAResumeState(
        x=transform(state.x),
        prompt_lens=tuple(state.prompt_lens),
        block_index=state.block_index,
        inner_step=state.inner_step,
        attention_mask=one(state.attention_mask),
        num_transfer_tokens=one(state.num_transfer_tokens),
        past_key_values=pkv,
        replace_position=one(state.replace_position),
        cache_mode=state.cache_mode,
    )


def _allocate_like(
    state: FastdLLMLLaDAResumeState,
    *,
    device: torch.device | str,
    pinned: bool = False,
) -> FastdLLMLLaDAResumeState:
    target = torch.device(device)

    def allocate(tensor: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "size": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "device": target,
        }
        if target.type == "cpu":
            kwargs["pin_memory"] = pinned
        return torch.empty(**kwargs)

    return _map_tensors(state, allocate)


def _tensor_pairs(
    source: FastdLLMLLaDAResumeState,
    destination: FastdLLMLLaDAResumeState,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    yield source.x, destination.x
    for name in ("attention_mask", "num_transfer_tokens", "replace_position"):
        src = getattr(source, name)
        dst = getattr(destination, name)
        if src is not None:
            if dst is None:
                raise RuntimeError(f"destination is missing {name}")
            yield src, dst
    if source.past_key_values is not None:
        if destination.past_key_values is None:
            raise RuntimeError("destination is missing past_key_values")
        for src_layer, dst_layer in zip(
            source.past_key_values, destination.past_key_values
        ):
            yield from zip(src_layer, dst_layer)


def _copy_state(
    source: FastdLLMLLaDAResumeState,
    destination: FastdLLMLLaDAResumeState,
    *,
    non_blocking: bool,
) -> None:
    for src, dst in _tensor_pairs(source, destination):
        dst.copy_(src, non_blocking=non_blocking)


def _offload(
    state: FastdLLMLLaDAResumeState,
    *,
    pinned: bool,
    cuda_device: torch.device,
) -> tuple[FastdLLMLLaDAResumeState, float, float]:
    prep_start = time.perf_counter()
    cpu_state = _allocate_like(state, device="cpu", pinned=pinned)
    prep_ms = (time.perf_counter() - prep_start) * 1000.0
    _sync(cuda_device)
    start = time.perf_counter()
    _copy_state(state, cpu_state, non_blocking=pinned)
    _sync(cuda_device)
    d2h_ms = (time.perf_counter() - start) * 1000.0
    return cpu_state, prep_ms, d2h_ms


def _restore_to_gpu(
    state: FastdLLMLLaDAResumeState,
    *,
    cuda_device: torch.device,
    non_blocking: bool,
) -> tuple[FastdLLMLLaDAResumeState, float, float]:
    prep_start = time.perf_counter()
    gpu_state = _allocate_like(state, device=cuda_device)
    prep_ms = (time.perf_counter() - prep_start) * 1000.0
    _sync(cuda_device)
    start = time.perf_counter()
    _copy_state(state, gpu_state, non_blocking=non_blocking)
    _sync(cuda_device)
    h2d_ms = (time.perf_counter() - start) * 1000.0
    return gpu_state, prep_ms, h2d_ms


def _release_gpu_state(state: FastdLLMLLaDAResumeState | None) -> None:
    del state
    gc.collect()


def _capture(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
    coordinate: tuple[int, int],
    *,
    resume_state: FastdLLMLLaDAResumeState | None = None,
    model_call_observer: Callable[[dict[str, Any]], None] | None = None,
) -> FastdLLMLLaDAResumeState:
    try:
        sampler.sample(
            inputs,
            config=config,
            pause_at=coordinate,
            resume_state=resume_state,
            model_call_observer=model_call_observer,
        )
    except FastdLLMLLaDAPaused as paused:
        return paused.state
    raise RuntimeError(f"sampler did not pause at {coordinate}")


def _resume_to_end(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
    state: FastdLLMLLaDAResumeState,
) -> torch.Tensor:
    output = sampler.sample(inputs, config=config, resume_state=state)
    return output if isinstance(output, torch.Tensor) else output.sequences


def _baseline(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
) -> tuple[torch.Tensor, list[StepPoint], dict[tuple[int, int], float]]:
    coordinates: list[tuple[int, int]] = []
    model_times: dict[tuple[int, int], float] = {}

    def step_observer(record: dict[str, Any]) -> None:
        coordinates.append((int(record["block_index"]), int(record["inner_step"])))

    def model_observer(record: dict[str, Any]) -> None:
        if "block_index" in record and "step_index" in record:
            model_times[(int(record["block_index"]), int(record["step_index"]))] = float(
                record["model_call_latency_ms"]
            )

    output = sampler.sample(
        inputs,
        config=config,
        step_observer=step_observer,
        model_call_observer=model_observer,
    )
    tensor = output if isinstance(output, torch.Tensor) else output.sequences
    points = [
        StepPoint(
            index=index,
            block=block,
            inner_step=inner,
            progress=index / max(1, len(coordinates)),
        )
        for index, (block, inner) in enumerate(coordinates)
    ]
    return tensor.detach().cpu().clone(), points, model_times


def _representative_points(points: list[StepPoint], limit: int) -> list[StepPoint]:
    if not points:
        return []
    selected: dict[tuple[int, int], StepPoint] = {}
    by_block: dict[int, list[StepPoint]] = {}
    for point in points:
        by_block.setdefault(point.block, []).append(point)
    for block_points in by_block.values():
        boundary = next((point for point in block_points if point.inner_step == 0), None)
        inner = [point for point in block_points if point.inner_step > 0]
        if boundary is not None:
            selected[(boundary.block, boundary.inner_step)] = boundary
        if inner:
            for point in (inner[0], inner[len(inner) // 2], inner[-1]):
                selected[(point.block, point.inner_step)] = point
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        nearest = min(points, key=lambda point: abs(point.progress - fraction))
        selected[(nearest.block, nearest.inner_step)] = nearest
    ordered = sorted(selected.values(), key=lambda point: point.index)
    if len(ordered) <= limit:
        return ordered
    anchors = [
        min(ordered, key=lambda point: abs(point.progress - fraction))
        for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    ]
    unique = {(point.block, point.inner_step): point for point in anchors}
    for point in ordered:
        if len(unique) >= limit:
            break
        unique.setdefault((point.block, point.inner_step), point)
    return sorted(unique.values(), key=lambda point: point.index)


def _make_inputs(tokenizer: Any, prompt: str, target_length: int) -> torch.Tensor:
    base = tokenizer.apply_chat_template(
        [[{"role": "user", "content": prompt}]],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not isinstance(base, torch.Tensor):
        base = torch.as_tensor(base, dtype=torch.long)
    if base.dim() == 1:
        base = base.unsqueeze(0)
    ids = base[0]
    if ids.numel() < target_length:
        repeats = math.ceil(target_length / ids.numel())
        ids = ids.repeat(repeats)
    return ids[:target_length].unsqueeze(0)


def _config(
    *,
    cache_mode: str,
    schedule_mode: str,
    generation_length: int,
    block_size: int,
    threshold: float,
) -> FastdLLMLLaDASamplerConfig:
    return FastdLLMLLaDASamplerConfig(
        steps=generation_length,
        max_new_tokens=generation_length,
        block_size=block_size,
        temperature=0.0,
        remasking="low_confidence",
        stochastic_transfer=False,
        threshold=threshold if schedule_mode == "threshold" else None,
        factor=None,
        use_cache=cache_mode,
        return_dict=False,
    )


def _validate_recovery_paths(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: FastdLLMLLaDASamplerConfig,
    coordinate: tuple[int, int],
    baseline_final: torch.Tensor,
    schedule_mode: str,
    device: torch.device,
) -> None:
    for semantic_only in (False, True):
        captured = _capture(sampler, inputs, config, coordinate)
        state = _semantic_state(captured, schedule_mode) if semantic_only else captured
        cpu, _, _ = _offload(state, pinned=True, cuda_device=device)
        del state, captured
        _release_gpu_state(None)
        restored, _, _ = _restore_to_gpu(cpu, cuda_device=device, non_blocking=True)
        final = _resume_to_end(sampler, inputs, config, restored).detach().cpu()
        if not torch.equal(final, baseline_final):
            name = "semantic-only" if semantic_only else "full-offload"
            raise RuntimeError(f"{name} recovery is not exact at {coordinate}")
    restarted = sampler.sample(inputs, config=config)
    restarted_tensor = restarted if isinstance(restarted, torch.Tensor) else restarted.sequences
    if not torch.equal(restarted_tensor.detach().cpu(), baseline_final):
        raise RuntimeError(f"drop/restart is not deterministic at {coordinate}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        prompt, generation = item.strip().split(":", maxsplit=1)
        shapes.append((int(prompt), int(generation)))
    return shapes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Explain why request preemption is hard for diffusion language models.",
    )
    parser.add_argument("--request_shapes", default="25:128")
    parser.add_argument("--cache_modes", default="prefix,dual")
    parser.add_argument("--schedule_mode", choices=("threshold", "quota"), default="quota")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max_checkpoints", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="artifacts/preemption_state/recovery_killtest",
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
    sampler = FastdLLMLLaDASampler(model=model, tokenizer=tokenizer)
    device = model.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    boundary_jobs: list[dict[str, Any]] = []
    baseline_metadata: dict[str, Any] = {}
    cache_modes = [item.strip() for item in args.cache_modes.split(",") if item.strip()]

    for shape_index, (prompt_length, generation_length) in enumerate(
        _parse_shapes(args.request_shapes)
    ):
        request_id = f"shape{shape_index}_p{prompt_length}_g{generation_length}"
        inputs = _make_inputs(tokenizer, args.prompt, prompt_length).to(device)
        for cache_mode in cache_modes:
            config = _config(
                cache_mode=cache_mode,
                schedule_mode=args.schedule_mode,
                generation_length=generation_length,
                block_size=args.block_size,
                threshold=args.threshold,
            )
            print(f"warmup {request_id} cache={cache_mode}", flush=True)
            sampler.sample(inputs, config=config)
            baseline_final, all_points, model_times = _baseline(
                sampler, inputs, config
            )
            points = _representative_points(all_points, args.max_checkpoints)
            median_iteration_ms = statistics.median(model_times.values())
            baseline_metadata[f"{request_id}/{cache_mode}"] = {
                "steps": len(all_points),
                "median_iteration_ms": median_iteration_ms,
                "points": [
                    {
                        "block": point.block,
                        "inner_step": point.inner_step,
                        "progress": point.progress,
                    }
                    for point in points
                ],
            }

            for point in points:
                coordinate = (point.block, point.inner_step)
                print(
                    f"point {request_id} cache={cache_mode} block={point.block} "
                    f"inner={point.inner_step} progress={point.progress:.3f}",
                    flush=True,
                )
                _validate_recovery_paths(
                    sampler,
                    inputs,
                    config,
                    coordinate,
                    baseline_final,
                    args.schedule_mode,
                    device,
                )

                sample_state = _capture(sampler, inputs, config, coordinate)
                semantic_sample = _semantic_state(sample_state, args.schedule_mode)
                semantic_bytes = _state_bytes(semantic_sample)
                cache_bytes = _pkv_bytes(sample_state.past_key_values)
                full_bytes = _state_bytes(sample_state)
                del semantic_sample, sample_state
                _release_gpu_state(None)

                for pinned in (False, True):
                    strategy = (
                        "full_offload_pinned" if pinned else "full_offload_pageable"
                    )
                    gpu_state = _capture(sampler, inputs, config, coordinate)
                    for repetition in range(args.repetitions):
                        cpu_state, prep_ms, d2h_ms = _offload(
                            gpu_state, pinned=pinned, cuda_device=device
                        )
                        del gpu_state
                        _release_gpu_state(None)
                        restored, gpu_prep_ms, h2d_ms = _restore_to_gpu(
                            cpu_state,
                            cuda_device=device,
                            non_blocking=pinned,
                        )
                        _sync(device)
                        setup_start = time.perf_counter()
                        gpu_state = _capture(
                            sampler,
                            inputs,
                            config,
                            coordinate,
                            resume_state=restored,
                        )
                        del restored
                        _sync(device)
                        setup_ms = (time.perf_counter() - setup_start) * 1000.0
                        total = prep_ms + d2h_ms + gpu_prep_ms + h2d_ms + setup_ms
                        rows.append(
                            {
                                "request_id": request_id,
                                "cache_mode": cache_mode,
                                "prompt_length": prompt_length,
                                "generation_length": generation_length,
                                "block": point.block,
                                "inner_step": point.inner_step,
                                "progress": point.progress,
                                "semantic_bytes": semantic_bytes,
                                "cache_bytes": cache_bytes,
                                "strategy": strategy,
                                "cpu_memory_mode": "pinned" if pinned else "pageable",
                                "checkpoint_prepare_ms": prep_ms,
                                "d2h_ms": d2h_ms,
                                "gpu_restore_prepare_ms": gpu_prep_ms,
                                "h2d_ms": h2d_ms,
                                "semantic_restore_ms": 0.0,
                                "restore_setup_ms": setup_ms,
                                "rebuild_ms": 0.0,
                                "recompute_ms": 0.0,
                                "total_recovery_ms": total,
                                "gpu_bytes_retained": 0,
                                "full_state_bytes": full_bytes,
                                "median_iteration_ms": median_iteration_ms,
                                "final_exact": True,
                                "repetition": repetition,
                            }
                        )
                    del gpu_state, cpu_state
                    _release_gpu_state(None)

                semantic_gpu = _semantic_state(
                    _capture(sampler, inputs, config, coordinate), args.schedule_mode
                )
                for repetition in range(args.repetitions):
                    cpu_state, prep_ms, d2h_ms = _offload(
                        semantic_gpu, pinned=True, cuda_device=device
                    )
                    del semantic_gpu
                    _release_gpu_state(None)
                    restored, gpu_prep_ms, h2d_ms = _restore_to_gpu(
                        cpu_state, cuda_device=device, non_blocking=True
                    )
                    model_calls: list[dict[str, Any]] = []
                    _sync(device)
                    restore_start = time.perf_counter()
                    rebuilt = _capture(
                        sampler,
                        inputs,
                        config,
                        coordinate,
                        resume_state=restored,
                        model_call_observer=model_calls.append,
                    )
                    del restored
                    _sync(device)
                    restore_wall_ms = (time.perf_counter() - restore_start) * 1000.0
                    rebuild_ms = sum(
                        float(record["model_call_latency_ms"]) for record in model_calls
                    )
                    restore_setup_ms = max(0.0, restore_wall_ms - rebuild_ms)
                    total = (
                        prep_ms
                        + d2h_ms
                        + gpu_prep_ms
                        + h2d_ms
                        + restore_setup_ms
                        + rebuild_ms
                    )
                    rows.append(
                        {
                            "request_id": request_id,
                            "cache_mode": cache_mode,
                            "prompt_length": prompt_length,
                            "generation_length": generation_length,
                            "block": point.block,
                            "inner_step": point.inner_step,
                            "progress": point.progress,
                            "semantic_bytes": semantic_bytes,
                            "cache_bytes": cache_bytes,
                            "strategy": "semantic_only",
                            "cpu_memory_mode": "pinned",
                            "checkpoint_prepare_ms": prep_ms,
                            "d2h_ms": d2h_ms,
                            "gpu_restore_prepare_ms": gpu_prep_ms,
                            "h2d_ms": h2d_ms,
                            "semantic_restore_ms": restore_setup_ms,
                            "restore_setup_ms": restore_setup_ms,
                            "rebuild_ms": rebuild_ms,
                            "recompute_ms": 0.0,
                            "total_recovery_ms": total,
                            "gpu_bytes_retained": 0,
                            "full_state_bytes": full_bytes,
                            "median_iteration_ms": median_iteration_ms,
                            "final_exact": True,
                            "repetition": repetition,
                        }
                    )
                    semantic_gpu = _semantic_state(rebuilt, args.schedule_mode)
                    del rebuilt
                    _release_gpu_state(None)
                del semantic_gpu, cpu_state
                _release_gpu_state(None)

                for repetition in range(args.repetitions):
                    _sync(device)
                    start = time.perf_counter()
                    restarted_state = _capture(sampler, inputs, config, coordinate)
                    _sync(device)
                    recompute_ms = (time.perf_counter() - start) * 1000.0
                    rows.append(
                        {
                            "request_id": request_id,
                            "cache_mode": cache_mode,
                            "prompt_length": prompt_length,
                            "generation_length": generation_length,
                            "block": point.block,
                            "inner_step": point.inner_step,
                            "progress": point.progress,
                            "semantic_bytes": semantic_bytes,
                            "cache_bytes": cache_bytes,
                            "strategy": "drop_restart",
                            "cpu_memory_mode": "none",
                            "checkpoint_prepare_ms": 0.0,
                            "d2h_ms": 0.0,
                            "gpu_restore_prepare_ms": 0.0,
                            "h2d_ms": 0.0,
                            "semantic_restore_ms": 0.0,
                            "restore_setup_ms": 0.0,
                            "rebuild_ms": 0.0,
                            "recompute_ms": recompute_ms,
                            "total_recovery_ms": recompute_ms,
                            "gpu_bytes_retained": 0,
                            "full_state_bytes": full_bytes,
                            "median_iteration_ms": median_iteration_ms,
                            "final_exact": True,
                            "repetition": repetition,
                        }
                    )
                    del restarted_state
                    _release_gpu_state(None)

                keep_state = _capture(sampler, inputs, config, coordinate)
                for repetition in range(args.repetitions):
                    _sync(device)
                    start = time.perf_counter()
                    keep_state = _capture(
                        sampler,
                        inputs,
                        config,
                        coordinate,
                        resume_state=keep_state,
                    )
                    _sync(device)
                    setup_ms = (time.perf_counter() - start) * 1000.0
                    rows.append(
                        {
                            "request_id": request_id,
                            "cache_mode": cache_mode,
                            "prompt_length": prompt_length,
                            "generation_length": generation_length,
                            "block": point.block,
                            "inner_step": point.inner_step,
                            "progress": point.progress,
                            "semantic_bytes": semantic_bytes,
                            "cache_bytes": cache_bytes,
                            "strategy": "keep",
                            "cpu_memory_mode": "none",
                            "checkpoint_prepare_ms": 0.0,
                            "d2h_ms": 0.0,
                            "gpu_restore_prepare_ms": 0.0,
                            "h2d_ms": 0.0,
                            "semantic_restore_ms": 0.0,
                            "restore_setup_ms": setup_ms,
                            "rebuild_ms": 0.0,
                            "recompute_ms": 0.0,
                            "total_recovery_ms": setup_ms,
                            "gpu_bytes_retained": full_bytes,
                            "full_state_bytes": full_bytes,
                            "median_iteration_ms": median_iteration_ms,
                            "final_exact": True,
                            "repetition": repetition,
                        }
                    )
                del keep_state
                _release_gpu_state(None)

                next_boundary = next(
                    (
                        candidate
                        for candidate in all_points[point.index + 1 :]
                        if candidate.inner_step == 0 and candidate.block > point.block
                    ),
                    None,
                )
                if point.inner_step > 0 and next_boundary is not None:
                    boundary_jobs.append(
                        {
                            "request_id": request_id,
                            "cache_mode": cache_mode,
                            "prompt_length": prompt_length,
                            "generation_length": generation_length,
                            "config": config,
                            "inputs": inputs,
                            "point": point,
                            "next_boundary": next_boundary,
                            "semantic_bytes": semantic_bytes,
                            "cache_bytes": cache_bytes,
                            "median_iteration_ms": median_iteration_ms,
                        }
                    )

                _write_csv(output_dir / "recovery_costs.csv", rows)

    _write_csv(output_dir / "recovery_costs.csv", rows)
    (output_dir / "baseline_metadata.json").write_text(
        json.dumps(baseline_metadata, indent=2) + "\n", encoding="utf-8"
    )

    grouped: dict[tuple[str, str, int, int, str], list[float]] = {}
    for row in rows:
        if row["strategy"] == "keep":
            continue
        key = (
            row["request_id"],
            row["cache_mode"],
            int(row["block"]),
            int(row["inner_step"]),
            row["strategy"],
        )
        grouped.setdefault(key, []).append(float(row["total_recovery_ms"]))

    boundary_rows: list[dict[str, Any]] = []
    for job in boundary_jobs:
        point: StepPoint = job["point"]
        boundary: StepPoint = job["next_boundary"]
        strategy_medians = {
            strategy: statistics.median(values)
            for (request, cache, block, inner, strategy), values in grouped.items()
            if request == job["request_id"]
            and cache == job["cache_mode"]
            and block == point.block
            and inner == point.inner_step
            and strategy
            in ("full_offload_pinned", "semantic_only", "drop_restart")
        }
        immediate_strategy, immediate_ms = min(
            strategy_medians.items(), key=lambda item: item[1]
        )
        for repetition in range(args.repetitions):
            current = _capture(
                sampler,
                job["inputs"],
                job["config"],
                (point.block, point.inner_step),
            )
            _sync(device)
            start = time.perf_counter()
            boundary_state = _capture(
                sampler,
                job["inputs"],
                job["config"],
                (boundary.block, boundary.inner_step),
                resume_state=current,
            )
            del current
            _sync(device)
            wait_ms = (time.perf_counter() - start) * 1000.0
            semantic_boundary = _semantic_state(boundary_state, args.schedule_mode)
            cpu_state, prep_ms, d2h_ms = _offload(
                semantic_boundary, pinned=True, cuda_device=device
            )
            restored, gpu_prep_ms, h2d_ms = _restore_to_gpu(
                cpu_state, cuda_device=device, non_blocking=True
            )
            _sync(device)
            resume_start = time.perf_counter()
            _capture(
                sampler,
                job["inputs"],
                job["config"],
                (boundary.block, boundary.inner_step),
                resume_state=restored,
            )
            del restored
            _sync(device)
            resume_ms = (
                prep_ms
                + d2h_ms
                + gpu_prep_ms
                + h2d_ms
                + (time.perf_counter() - resume_start) * 1000.0
            )
            total_boundary = wait_ms + resume_ms
            boundary_rows.append(
                {
                    "request_id": job["request_id"],
                    "cache_mode": job["cache_mode"],
                    "block": point.block,
                    "inner_step": point.inner_step,
                    "progress": point.progress,
                    "distance_to_boundary_steps": boundary.index - point.index,
                    "distance_to_boundary_ms": wait_ms,
                    "immediate_strategy": immediate_strategy,
                    "immediate_recovery_ms": immediate_ms,
                    "boundary_wait_ms": wait_ms,
                    "boundary_checkpoint_bytes": _state_bytes(semantic_boundary),
                    "boundary_resume_ms": resume_ms,
                    "total_boundary_cost_ms": total_boundary,
                    "winner": (
                        "immediate" if immediate_ms <= total_boundary else "boundary"
                    ),
                    "median_iteration_ms": job["median_iteration_ms"],
                    "repetition": repetition,
                }
            )
    _write_csv(output_dir / "boundary_preemption.csv", boundary_rows)

    scaling_rows: list[dict[str, Any]] = []
    point_groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["request_id"],
            row["cache_mode"],
            int(row["block"]),
            int(row["inner_step"]),
        )
        point_groups.setdefault(key, []).append(row)
    for point_rows in point_groups.values():
        first = point_rows[0]
        strategy_values: dict[str, list[float]] = {}
        for row in point_rows:
            strategy_values.setdefault(row["strategy"], []).append(
                float(row["total_recovery_ms"])
            )
        scaling_rows.append(
            {
                "request_id": first["request_id"],
                "cache_mode": first["cache_mode"],
                "prompt_length": first["prompt_length"],
                "generation_length": first["generation_length"],
                "block": first["block"],
                "inner_step": first["inner_step"],
                "progress": first["progress"],
                "semantic_bytes": first["semantic_bytes"],
                "cache_bytes": first["cache_bytes"],
                "full_offload_ms": statistics.median(
                    strategy_values["full_offload_pinned"]
                ),
                "full_offload_p95_ms": _percentile(
                    strategy_values["full_offload_pinned"], 0.95
                ),
                "semantic_only_ms": statistics.median(
                    strategy_values["semantic_only"]
                ),
                "semantic_only_p95_ms": _percentile(
                    strategy_values["semantic_only"], 0.95
                ),
                "restart_ms": statistics.median(strategy_values["drop_restart"]),
                "restart_p95_ms": _percentile(
                    strategy_values["drop_restart"], 0.95
                ),
                "keep_resume_ms": statistics.median(strategy_values["keep"]),
                "keep_gpu_bytes": first["full_state_bytes"],
                "median_iteration_ms": first["median_iteration_ms"],
            }
        )
    _write_csv(output_dir / "state_cost_scaling.csv", scaling_rows)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
