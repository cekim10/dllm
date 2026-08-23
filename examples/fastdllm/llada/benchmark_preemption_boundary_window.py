"""Measure the profitable block-boundary deferral window for Fast LLaDA.

Run from the repository root:

    python -u examples/fastdllm/llada/benchmark_preemption_boundary_window.py \
      --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
      --prompt_lengths 512,2048,3968 \
      --generation_length 128 \
      --cache_modes prefix,dual \
      --distances 1,2,3,4 \
      --repetitions 10 \
      --output_dir artifacts/preemption_state/boundary_window_killtest
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm
from benchmark_preemption_recovery import (
    _baseline,
    _capture,
    _config,
    _make_inputs,
    _offload,
    _percentile,
    _pkv_bytes,
    _release_gpu_state,
    _restore_to_gpu,
    _resume_to_end,
    _state_bytes,
    _sync,
)
from dllm.pipelines.fastdllm.llada import (
    FastdLLMLLaDAConfig,
    FastdLLMLLaDASampler,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _full_roundtrip(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: Any,
    coordinate: tuple[int, int],
    state: Any,
    device: torch.device,
) -> tuple[Any, dict[str, float]]:
    cpu_state, checkpoint_prepare_ms, d2h_ms = _offload(
        state, pinned=True, cuda_device=device
    )
    del state
    _release_gpu_state(None)
    restored, restore_prepare_ms, h2d_ms = _restore_to_gpu(
        cpu_state, cuda_device=device, non_blocking=True
    )
    del cpu_state
    _sync(device)
    start = time.perf_counter()
    resumed = _capture(
        sampler,
        inputs,
        config,
        coordinate,
        resume_state=restored,
    )
    del restored
    _sync(device)
    restore_setup_ms = (time.perf_counter() - start) * 1000.0
    timings = {
        "checkpoint_prepare_ms": checkpoint_prepare_ms,
        "d2h_ms": d2h_ms,
        "restore_prepare_ms": restore_prepare_ms,
        "h2d_ms": h2d_ms,
        "restore_setup_ms": restore_setup_ms,
    }
    timings["roundtrip_ms"] = sum(timings.values())
    return resumed, timings


def _validate_exact(
    sampler: FastdLLMLLaDASampler,
    inputs: torch.Tensor,
    config: Any,
    point_coordinate: tuple[int, int],
    boundary_coordinate: tuple[int, int],
    baseline_final: torch.Tensor,
    device: torch.device,
) -> None:
    point_state = _capture(sampler, inputs, config, point_coordinate)
    restored_point, _ = _full_roundtrip(
        sampler, inputs, config, point_coordinate, point_state, device
    )
    immediate_final = _resume_to_end(
        sampler, inputs, config, restored_point
    ).detach().cpu()
    if not torch.equal(immediate_final, baseline_final):
        raise RuntimeError(f"immediate offload is not exact at {point_coordinate}")

    point_state = _capture(sampler, inputs, config, point_coordinate)
    boundary_state = _capture(
        sampler,
        inputs,
        config,
        boundary_coordinate,
        resume_state=point_state,
    )
    restored_boundary, _ = _full_roundtrip(
        sampler, inputs, config, boundary_coordinate, boundary_state, device
    )
    boundary_final = _resume_to_end(
        sampler, inputs, config, restored_boundary
    ).detach().cpu()
    if not torch.equal(boundary_final, baseline_final):
        raise RuntimeError(f"boundary deferral is not exact at {point_coordinate}")


def _summaries(rows: list[dict[str, Any]], block_size: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["request_id"], row["cache_mode"], int(row["distance_steps"]))
        groups.setdefault(key, []).append(row)

    summaries = []
    for group in groups.values():
        first = group[0]
        immediate = [float(row["immediate_offload_ms"]) for row in group]
        wait = [float(row["wait_to_boundary_ms"]) for row in group]
        boundary = [float(row["total_boundary_cost_ms"]) for row in group]
        margins = [float(row["margin_ms"]) for row in group]
        median_margin = statistics.median(margins)
        summaries.append(
            {
                "request_id": first["request_id"],
                "cache_mode": first["cache_mode"],
                "prompt_length": first["prompt_length"],
                "generation_length": first["generation_length"],
                "distance_steps": first["distance_steps"],
                "immediate_cache_bytes": first["immediate_cache_bytes"],
                "boundary_checkpoint_bytes": first["boundary_checkpoint_bytes"],
                "immediate_offload_median_ms": statistics.median(immediate),
                "immediate_offload_p95_ms": _percentile(immediate, 0.95),
                "wait_to_boundary_median_ms": statistics.median(wait),
                "boundary_total_median_ms": statistics.median(boundary),
                "boundary_total_p95_ms": _percentile(boundary, 0.95),
                "margin_median_ms": median_margin,
                "margin_p95_ms": _percentile(margins, 0.95),
                "winner": "boundary" if median_margin > 0 else "immediate",
                "profitable_repetition_fraction": sum(
                    float(row["margin_ms"]) > 0 for row in group
                )
                / len(group),
                "block_fraction": 1.0 / block_size,
                "median_iteration_ms": first["median_iteration_ms"],
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            int(row["prompt_length"]),
            row["cache_mode"],
            int(row["distance_steps"]),
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt", default="Explain dLLM request preemption.")
    parser.add_argument("--prompt_lengths", default="512,2048,3968")
    parser.add_argument("--generation_length", type=int, default=128)
    parser.add_argument("--cache_modes", default="prefix,dual")
    parser.add_argument("--distances", default="1,2,3,4")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--schedule_mode", choices=("threshold", "quota"), default="quota")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="artifacts/preemption_state/boundary_window_killtest",
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

    prompt_lengths = [int(value) for value in args.prompt_lengths.split(",")]
    distances = sorted({int(value) for value in args.distances.split(",")})
    cache_modes = [value.strip() for value in args.cache_modes.split(",")]
    if not distances or min(distances) < 1 or max(distances) >= args.block_size:
        raise ValueError("distances must be between 1 and block_size - 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    for prompt_length in prompt_lengths:
        inputs = _make_inputs(tokenizer, args.prompt, prompt_length).to(device)
        request_id = f"p{prompt_length}_g{args.generation_length}"
        for cache_mode in cache_modes:
            config = _config(
                cache_mode=cache_mode,
                schedule_mode=args.schedule_mode,
                generation_length=args.generation_length,
                block_size=args.block_size,
                threshold=args.threshold,
            )
            print(f"warmup {request_id} cache={cache_mode}", flush=True)
            sampler.sample(inputs, config=config)
            baseline_final, points, model_times = _baseline(sampler, inputs, config)
            boundaries = [point for point in points if point.inner_step == 0]
            if len(boundaries) < 2:
                raise RuntimeError("generation did not expose a complete block boundary")
            boundary = boundaries[1]
            by_index = {point.index: point for point in points}
            median_iteration_ms = statistics.median(model_times.values())
            metadata[f"{request_id}/{cache_mode}"] = {
                "boundary": [boundary.block, boundary.inner_step],
                "median_iteration_ms": median_iteration_ms,
            }

            for distance in distances:
                point = by_index[boundary.index - distance]
                point_coordinate = (point.block, point.inner_step)
                boundary_coordinate = (boundary.block, boundary.inner_step)
                print(
                    f"point {request_id} cache={cache_mode} distance={distance} "
                    f"coordinate={point_coordinate}",
                    flush=True,
                )
                _validate_exact(
                    sampler,
                    inputs,
                    config,
                    point_coordinate,
                    boundary_coordinate,
                    baseline_final,
                    device,
                )

                sample_point = _capture(sampler, inputs, config, point_coordinate)
                immediate_cache_bytes = _pkv_bytes(sample_point.past_key_values)
                immediate_full_bytes = _state_bytes(sample_point)
                sample_boundary = _capture(
                    sampler,
                    inputs,
                    config,
                    boundary_coordinate,
                    resume_state=sample_point,
                )
                boundary_cache_bytes = _pkv_bytes(sample_boundary.past_key_values)
                boundary_checkpoint_bytes = _state_bytes(sample_boundary)
                del sample_point, sample_boundary
                _release_gpu_state(None)
                if boundary_cache_bytes != 0:
                    raise RuntimeError("natural block boundary unexpectedly retains cache")

                for repetition in range(args.repetitions):
                    immediate_state = _capture(
                        sampler, inputs, config, point_coordinate
                    )
                    immediate_resumed, immediate = _full_roundtrip(
                        sampler,
                        inputs,
                        config,
                        point_coordinate,
                        immediate_state,
                        device,
                    )
                    del immediate_resumed
                    _release_gpu_state(None)

                    current = _capture(sampler, inputs, config, point_coordinate)
                    _sync(device)
                    wait_start = time.perf_counter()
                    boundary_state = _capture(
                        sampler,
                        inputs,
                        config,
                        boundary_coordinate,
                        resume_state=current,
                    )
                    del current
                    _sync(device)
                    wait_ms = (time.perf_counter() - wait_start) * 1000.0
                    boundary_resumed, boundary_timings = _full_roundtrip(
                        sampler,
                        inputs,
                        config,
                        boundary_coordinate,
                        boundary_state,
                        device,
                    )
                    del boundary_resumed
                    _release_gpu_state(None)

                    boundary_total_ms = wait_ms + boundary_timings["roundtrip_ms"]
                    margin_ms = immediate["roundtrip_ms"] - boundary_total_ms
                    raw_rows.append(
                        {
                            "request_id": request_id,
                            "cache_mode": cache_mode,
                            "prompt_length": prompt_length,
                            "generation_length": args.generation_length,
                            "block": point.block,
                            "inner_step": point.inner_step,
                            "distance_steps": distance,
                            "immediate_cache_bytes": immediate_cache_bytes,
                            "immediate_full_bytes": immediate_full_bytes,
                            "boundary_checkpoint_bytes": boundary_checkpoint_bytes,
                            "median_iteration_ms": median_iteration_ms,
                            "immediate_checkpoint_prepare_ms": immediate[
                                "checkpoint_prepare_ms"
                            ],
                            "immediate_d2h_ms": immediate["d2h_ms"],
                            "immediate_restore_prepare_ms": immediate[
                                "restore_prepare_ms"
                            ],
                            "immediate_h2d_ms": immediate["h2d_ms"],
                            "immediate_restore_setup_ms": immediate[
                                "restore_setup_ms"
                            ],
                            "immediate_offload_ms": immediate["roundtrip_ms"],
                            "wait_to_boundary_ms": wait_ms,
                            "boundary_checkpoint_prepare_ms": boundary_timings[
                                "checkpoint_prepare_ms"
                            ],
                            "boundary_d2h_ms": boundary_timings["d2h_ms"],
                            "boundary_restore_prepare_ms": boundary_timings[
                                "restore_prepare_ms"
                            ],
                            "boundary_h2d_ms": boundary_timings["h2d_ms"],
                            "boundary_restore_setup_ms": boundary_timings[
                                "restore_setup_ms"
                            ],
                            "boundary_roundtrip_ms": boundary_timings[
                                "roundtrip_ms"
                            ],
                            "total_boundary_cost_ms": boundary_total_ms,
                            "margin_ms": margin_ms,
                            "winner": "boundary" if margin_ms > 0 else "immediate",
                            "final_exact": True,
                            "repetition": repetition,
                        }
                    )
                _write_csv(output_dir / "boundary_window_raw.csv", raw_rows)

    summary_rows = _summaries(raw_rows, args.block_size)
    _write_csv(output_dir / "boundary_window_summary.csv", summary_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
