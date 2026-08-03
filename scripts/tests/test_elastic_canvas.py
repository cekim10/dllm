"""
Unit tests for elastic-canvas simulation utilities.

Run from repo root:
  pytest /Users/cekim/Desktop/git/dllm/scripts/tests/test_elastic_canvas.py -v
"""

import pytest

from dllm.pipelines.fastdllm.llada.elastic_canvas import (
    ElasticCanvasConfig,
    round_canvas_length,
    summarize_elastic_canvas,
)


def test_round_canvas_length_respects_initial_page_and_cap():
    assert (
        round_canvas_length(
            3, initial_canvas=32, page_size=32, max_canvas=256
        )
        == 32
    )
    assert (
        round_canvas_length(
            33, initial_canvas=32, page_size=32, max_canvas=256
        )
        == 64
    )
    assert (
        round_canvas_length(
            300, initial_canvas=32, page_size=32, max_canvas=256
        )
        == 256
    )


def test_summarize_elastic_canvas_reports_oracle_savings():
    summary = summarize_elastic_canvas(
        [8, 32, 33, 120],
        ElasticCanvasConfig(
            fixed_canvas=256,
            initial_canvas=32,
            page_size=32,
            batch_size=2,
            steps=128,
        ),
    )

    assert summary["per_request_elastic_lengths"] == [32, 32, 64, 128]
    assert summary["fixed_allocated_tokens"] == 1024
    assert summary["elastic_allocated_tokens"] == 256
    assert summary["oracle_token_volume_reduction"] == pytest.approx(0.75)
    assert summary["oracle_attention_volume_reduction"] > 0.9
