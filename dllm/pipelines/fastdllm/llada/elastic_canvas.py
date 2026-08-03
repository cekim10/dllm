"""
Utilities for elastic-canvas serving experiments.

Run from repo root:
  python -m py_compile dllm/pipelines/fastdllm/llada/elastic_canvas.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ElasticCanvasConfig:
    fixed_canvas: int
    initial_canvas: int = 32
    page_size: int = 32
    batch_size: int = 8
    steps: int = 256


def round_canvas_length(
    useful_length: int,
    *,
    initial_canvas: int,
    page_size: int,
    max_canvas: int,
) -> int:
    if useful_length < 0:
        raise ValueError(f"useful_length must be non-negative, got {useful_length}")
    if initial_canvas <= 0:
        raise ValueError(f"initial_canvas must be positive, got {initial_canvas}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if max_canvas <= 0:
        raise ValueError(f"max_canvas must be positive, got {max_canvas}")

    needed = max(1, useful_length)
    rounded = max(initial_canvas, math.ceil(needed / page_size) * page_size)
    return min(max_canvas, rounded)


def summarize_elastic_canvas(
    useful_lengths: list[int],
    config: ElasticCanvasConfig,
) -> dict[str, object]:
    if not useful_lengths:
        raise ValueError("useful_lengths must not be empty")
    if config.fixed_canvas <= 0:
        raise ValueError(f"fixed_canvas must be positive, got {config.fixed_canvas}")
    if config.steps <= 0:
        raise ValueError(f"steps must be positive, got {config.steps}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")

    elastic_lengths = [
        round_canvas_length(
            length,
            initial_canvas=config.initial_canvas,
            page_size=config.page_size,
            max_canvas=config.fixed_canvas,
        )
        for length in useful_lengths
    ]
    n = len(useful_lengths)
    fixed_tokens = n * config.fixed_canvas
    useful_tokens = sum(min(length, config.fixed_canvas) for length in useful_lengths)
    elastic_tokens = sum(elastic_lengths)
    fixed_step_tokens = fixed_tokens * config.steps
    elastic_step_tokens = elastic_tokens * config.steps
    fixed_attention_units = n * (config.fixed_canvas**2) * config.steps
    elastic_attention_units = sum(length**2 for length in elastic_lengths) * config.steps

    fixed_batch_dense_tokens = 0
    elastic_batch_dense_tokens = 0
    elastic_batch_packed_tokens = 0
    for start in range(0, n, config.batch_size):
        batch = elastic_lengths[start : start + config.batch_size]
        fixed_batch_dense_tokens += len(batch) * config.fixed_canvas
        elastic_batch_dense_tokens += len(batch) * max(batch)
        elastic_batch_packed_tokens += sum(batch)

    growth_events = [
        max(0, (length - config.initial_canvas + config.page_size - 1) // config.page_size)
        for length in elastic_lengths
    ]
    copied_tokens = [
        sum(config.initial_canvas + i * config.page_size for i in range(events))
        for events in growth_events
    ]

    return {
        "num_requests": n,
        "fixed_canvas": config.fixed_canvas,
        "initial_canvas": config.initial_canvas,
        "page_size": config.page_size,
        "batch_size": config.batch_size,
        "steps": config.steps,
        "useful_tokens": useful_tokens,
        "fixed_allocated_tokens": fixed_tokens,
        "elastic_allocated_tokens": elastic_tokens,
        "fixed_padding_tokens": fixed_tokens - useful_tokens,
        "elastic_padding_tokens": elastic_tokens - useful_tokens,
        "fixed_useful_ratio": useful_tokens / fixed_tokens,
        "elastic_useful_ratio": useful_tokens / elastic_tokens
        if elastic_tokens > 0
        else 0.0,
        "elastic_vs_fixed_token_ratio": elastic_tokens / fixed_tokens,
        "oracle_token_volume_ratio": elastic_step_tokens / fixed_step_tokens,
        "oracle_token_volume_reduction": 1.0
        - (elastic_step_tokens / fixed_step_tokens),
        "oracle_attention_volume_ratio": elastic_attention_units
        / fixed_attention_units,
        "oracle_attention_volume_reduction": 1.0
        - (elastic_attention_units / fixed_attention_units),
        "fixed_batch_dense_tokens": fixed_batch_dense_tokens,
        "elastic_batch_dense_tokens": elastic_batch_dense_tokens,
        "elastic_batch_packed_tokens": elastic_batch_packed_tokens,
        "elastic_dense_padding_tokens": elastic_batch_dense_tokens
        - elastic_batch_packed_tokens,
        "packed_vs_elastic_dense_token_ratio": elastic_batch_packed_tokens
        / elastic_batch_dense_tokens
        if elastic_batch_dense_tokens > 0
        else 0.0,
        "total_growth_events": sum(growth_events),
        "total_reallocation_copied_tokens": sum(copied_tokens),
        "per_request_elastic_lengths": elastic_lengths,
        "per_request_growth_events": growth_events,
    }
