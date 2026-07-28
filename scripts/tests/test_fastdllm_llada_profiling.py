"""
Unit tests for Fast-dLLM LLaDA block-usefulness profiling.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /Users/cekim/Desktop/git/dllm/scripts/tests/test_fastdllm_llada_profiling.py -v
"""

from collections import namedtuple

import torch

from dllm.pipelines.fastdllm.llada.profiling import BlockUsefulnessProfiler
from dllm.pipelines.fastdllm.llada.sampler import (
    FastdLLMLLaDASampler,
    FastdLLMLLaDASamplerConfig,
)


DummyOutput = namedtuple("DummyOutput", ["logits", "past_key_values", "hidden_states"])


class DummyTokenizer:
    mask_token_id = 99
    bos_token_id = 1
    eos_token_id = 0


class DummyModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.forward_calls = []

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        output_hidden_states=False,
        block_observer=None,
        observer_context=None,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        vocab_size = 128
        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        logits[..., 7] = 10.0

        hidden0 = input_ids.unsqueeze(-1).repeat(1, 1, 2).to(torch.float32)
        hidden1 = hidden0 + 1.0
        hidden2 = hidden1 + 1.0

        self.forward_calls.append(observer_context)
        if block_observer is not None:
            block_observer(
                observer_context={
                    **observer_context,
                    "layer_index": 0,
                    "layer_latency_ms": 0.1,
                },
                hidden_before=hidden0,
                hidden_after=hidden1,
            )
            block_observer(
                observer_context={
                    **observer_context,
                    "layer_index": 1,
                    "layer_latency_ms": 0.2,
                },
                hidden_before=hidden1,
                hidden_after=hidden2,
            )

        return DummyOutput(
            logits=logits,
            past_key_values=None,
            hidden_states=(hidden0, hidden1, hidden2) if output_hidden_states else None,
        )


def test_sampler_emits_block_observer_context():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    sampler = FastdLLMLLaDASampler(model=model, tokenizer=tokenizer)
    profiler = BlockUsefulnessProfiler()
    config = FastdLLMLLaDASamplerConfig(
        max_new_tokens=2,
        max_length=3,
        block_size=2,
        steps=2,
        temperature=0.0,
        remasking="low_confidence",
        use_cache=None,
    )

    outputs = sampler.sample(
        inputs=[[1]],
        config=config,
        return_dict=True,
        block_observer=profiler,
    )

    assert outputs.sequences.shape == (1, 3)
    assert len(model.forward_calls) == 2
    assert model.forward_calls[0]["cache_mode"] == "none"
    assert model.forward_calls[0]["step_index"] == 0
    assert model.forward_calls[1]["step_index"] == 1
    assert model.forward_calls[0]["block_ranges"] == ((1, 3),)
    assert len(profiler.records) == 4

    second_step_records = [
        record for record in profiler.records if record["step_index"] == 1
    ]
    assert second_step_records
    assert all(
        record["inter_step_output_delta_norm"] is not None
        for record in second_step_records
    )
