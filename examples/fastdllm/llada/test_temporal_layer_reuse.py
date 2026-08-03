"""
Compare baseline Fast-dLLM prefix decoding against cross-refinement prefix-layer reuse.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  python -u examples/fastdllm/llada/test_temporal_layer_reuse.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

import dllm
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.fastdllm.llada.sampler import (
    _get_transfer_index,
    _trim_past_key_values,
)


def _load_messages(
    input_path: str | None, prompt: str, limit: int
) -> list[list[dict[str, str]]]:
    if input_path is None:
        return [[{"role": "user", "content": prompt}]]

    path = Path(input_path)
    messages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                record = json.loads(line)
                if "messages" in record:
                    messages.append(record["messages"])
                else:
                    text = (
                        record.get("prompt")
                        or record.get("text")
                        or record.get("input")
                    )
                    if text is None:
                        raise ValueError(f"Unsupported JSONL record: {record}")
                    messages.append([{"role": "user", "content": text}])
            else:
                messages.append([{"role": "user", "content": line}])
            if len(messages) >= limit:
                break
    return messages


def _apply_suppressions(
    logits: torch.Tensor,
    suppress_tokens: list[int] | None,
    begin_suppress_tokens: list[int] | None,
) -> None:
    if suppress_tokens:
        for tid in suppress_tokens:
            logits[:, :, tid] = -torch.inf
    if begin_suppress_tokens:
        for tid in begin_suppress_tokens:
            logits[:, :, tid] = -torch.inf


def _count_token_differences(
    a: torch.Tensor, b: torch.Tensor, prompt_lens: list[int]
) -> list[int]:
    diffs = []
    for idx, prompt_len in enumerate(prompt_lens):
        diffs.append(int((a[idx, prompt_len:] != b[idx, prompt_len:]).sum().item()))
    return diffs


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str | None = None
    prompt: str = "Explain why caching helps diffusion language model serving."
    limit: int = 1
    seed: int = 42
    output_path: str = "artifacts/block_usefulness/reuse_test.json"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.llada.FastdLLMLLaDASamplerConfig):
    steps: int = 256
    max_new_tokens: int = 256
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "prefix"
    threshold: float | None = 0.9
    factor: float | None = None
    begin_suppress_tokens: list[int] | None = None
    reuse_prefix_layers: int = 8
    reuse_after_step: int = 3
    refresh_interval: int = 4


def run_prefix_decode(
    model,
    tokenizer,
    scheduler,
    inputs: torch.Tensor,
    config: SamplerConfig,
    reuse_prefix_layers: int = 0,
    reuse_after_step: int = 10**9,
    refresh_interval: int = 1,
):
    if config.use_cache != "prefix":
        raise ValueError("This script currently supports use_cache='prefix' only.")
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)

    prompt_lens = [int((row != tokenizer.pad_token_id).sum().item()) for row in inputs]
    if len(set(prompt_lens)) != 1:
        raise ValueError(
            f"Prefix-cache replay requires equal prompt lengths, got {prompt_lens}"
        )
    prompt_len = prompt_lens[0]
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    T = prompt_len + config.max_new_tokens
    B = inputs.shape[0]

    x = torch.full((B, T), eos_id, dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=model.device)
    for i in range(B):
        x[i, :prompt_len] = inputs[i, :prompt_len]
        x[i, prompt_len:T] = mask_id
        attention_mask[i, :T] = 1

    histories = [x.clone()]
    num_blocks = (config.max_new_tokens + config.block_size - 1) // config.block_size
    steps_per_block = (config.steps + num_blocks - 1) // num_blocks
    executed_layer_steps = 0
    full_layer_steps = 0
    refresh_steps = 0
    reuse_steps = 0

    def should_refresh(step_index: int) -> bool:
        if reuse_prefix_layers <= 0:
            return True
        if step_index < reuse_after_step:
            return True
        if refresh_interval <= 1:
            return True
        return (step_index - reuse_after_step) % refresh_interval == 0

    _sync_device(model.device)
    start_time = time.perf_counter()
    for block_index in range(num_blocks):
        s = prompt_len + block_index * config.block_size
        e = min(s + config.block_size, T)
        if s >= e:
            continue
        block_len = e - s
        block_mask_index = torch.zeros(
            (B, config.block_size), dtype=torch.bool, device=x.device
        )
        block_mask_index[:, :block_len] = x[:, s:e] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index,
            steps=steps_per_block,
            scheduler=scheduler,
            stochastic=config.stochastic_transfer,
        )

        out_full = model(
            x,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=reuse_prefix_layers > 0,
        )
        logits_full = out_full.logits
        _apply_suppressions(
            logits_full, config.suppress_tokens, config.begin_suppress_tokens
        )
        if config.right_shift_logits:
            logits_full = torch.cat([logits_full[:, :1], logits_full[:, :-1]], dim=1)

        mask_allowed = torch.zeros_like(x, dtype=torch.bool)
        mask_allowed[:, s:e] = x[:, s:e] == mask_id
        if mask_allowed.sum() > 0:
            quota = None if config.threshold is not None else num_transfer_tokens[:, 0]
            x0, transfer_idx = _get_transfer_index(
                logits=logits_full,
                temperature=config.temperature,
                remasking=config.remasking,
                mask_index=mask_allowed,
                x=x,
                num_transfer_tokens=quota,
                threshold=config.threshold,
                factor=config.factor,
            )
            x = torch.where(transfer_idx, x0, x)
            histories.append(x.clone())

        past_key_values = out_full.past_key_values
        if past_key_values is None:
            raise RuntimeError("Model did not return past_key_values with use_cache=True")
        past_key_values = _trim_past_key_values(past_key_values, s)
        cached_prefix_hidden = None
        if reuse_prefix_layers > 0:
            assert out_full.hidden_states is not None
            cached_prefix_hidden = out_full.hidden_states[reuse_prefix_layers].detach()

        effective_steps = num_transfer_tokens.size(1)
        for step_index in range(1, effective_steps):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            x_suffix = x[:, s:]
            mask_suffix = x_suffix == mask_id
            if x_suffix.size(1) > block_len:
                mask_suffix[:, block_len:] = False
            if mask_suffix.sum() == 0:
                break

            full_layer_steps += model.config.n_layers
            if should_refresh(step_index):
                out_suf = model(
                    x_suffix,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=reuse_prefix_layers > 0,
                )
                refresh_steps += 1
                executed_layer_steps += model.config.n_layers
                if reuse_prefix_layers > 0:
                    assert out_suf.hidden_states is not None
                    cached_prefix_hidden = out_suf.hidden_states[
                        reuse_prefix_layers
                    ].detach()
            else:
                if cached_prefix_hidden is None:
                    raise RuntimeError("Missing cached prefix hidden state for reuse.")
                out_suf = model(
                    attention_mask=attention_mask,
                    inputs_embeds=cached_prefix_hidden,
                    past_key_values=past_key_values[reuse_prefix_layers:],
                    use_cache=True,
                    output_hidden_states=False,
                    start_layer=reuse_prefix_layers,
                    hidden_state_input=True,
                )
                reuse_steps += 1
                executed_layer_steps += model.config.n_layers - reuse_prefix_layers

            logits_suf = out_suf.logits
            _apply_suppressions(
                logits_suf, config.suppress_tokens, config.begin_suppress_tokens
            )
            if config.right_shift_logits:
                logits_suf = torch.cat([logits_suf[:, :1], logits_suf[:, :-1]], dim=1)

            quota = (
                None
                if (config.threshold is not None or config.factor is not None)
                else num_transfer_tokens[:, step_index]
            )
            x0_suf, transfer_suf = _get_transfer_index(
                logits=logits_suf,
                temperature=config.temperature,
                remasking=config.remasking,
                mask_index=mask_suffix,
                x=x_suffix,
                num_transfer_tokens=quota,
                threshold=config.threshold,
                factor=config.factor,
            )

            x_suffix_new = torch.where(transfer_suf, x0_suf, x_suffix)
            x = torch.cat([x[:, :s], x_suffix_new], dim=1)
            histories.append(x.clone())

    _sync_device(model.device)
    elapsed_seconds = time.perf_counter() - start_time
    return {
        "sequences": x,
        "histories": histories,
        "elapsed_seconds": elapsed_seconds,
        "refresh_steps": refresh_steps,
        "reuse_steps": reuse_steps,
        "executed_layer_steps": executed_layer_steps,
        "full_layer_steps": full_layer_steps,
        "layer_execution_ratio": (
            executed_layer_steps / full_layer_steps if full_layer_steps > 0 else 1.0
        ),
    }


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

messages = _load_messages(
    input_path=script_args.input_path,
    prompt=script_args.prompt,
    limit=script_args.limit,
)
fastdllm_config = dllm.pipelines.fastdllm.llada.FastdLLMLLaDAConfig.from_pretrained(
    script_args.model_name_or_path
)
model = dllm.utils.get_model(model_args=script_args, config=fastdllm_config).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    padding=True,
    return_tensors="pt",
).to(model.device)
scheduler = dllm.core.schedulers.LinearAlphaScheduler()

baseline = run_prefix_decode(
    model=model,
    tokenizer=tokenizer,
    scheduler=scheduler,
    inputs=inputs,
    config=sampler_config,
    reuse_prefix_layers=0,
)
reuse = run_prefix_decode(
    model=model,
    tokenizer=tokenizer,
    scheduler=scheduler,
    inputs=inputs,
    config=sampler_config,
    reuse_prefix_layers=sampler_config.reuse_prefix_layers,
    reuse_after_step=sampler_config.reuse_after_step,
    refresh_interval=sampler_config.refresh_interval,
)

baseline_texts = dllm.utils.sample_trim(
    tokenizer, baseline["sequences"].tolist(), inputs.tolist()
)
reuse_texts = dllm.utils.sample_trim(
    tokenizer, reuse["sequences"].tolist(), inputs.tolist()
)
token_differences = _count_token_differences(
    baseline["sequences"], reuse["sequences"], [int((row != tokenizer.pad_token_id).sum().item()) for row in inputs]
)
results = {
    "model_name_or_path": script_args.model_name_or_path,
    "reuse_prefix_layers": sampler_config.reuse_prefix_layers,
    "reuse_after_step": sampler_config.reuse_after_step,
    "refresh_interval": sampler_config.refresh_interval,
    "baseline_elapsed_seconds": baseline["elapsed_seconds"],
    "reuse_elapsed_seconds": reuse["elapsed_seconds"],
    "speedup": baseline["elapsed_seconds"] / reuse["elapsed_seconds"]
    if reuse["elapsed_seconds"] > 0
    else None,
    "reuse_layer_execution_ratio": reuse["layer_execution_ratio"],
    "refresh_steps": reuse["refresh_steps"],
    "reuse_steps": reuse["reuse_steps"],
    "exact_match_per_request": [
        baseline_texts[i] == reuse_texts[i] for i in range(len(baseline_texts))
    ],
    "token_differences_per_request": token_differences,
    "baseline_texts": baseline_texts,
    "reuse_texts": reuse_texts,
}

output_path = Path(script_args.output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=True, indent=2)

print(json.dumps({k: v for k, v in results.items() if k not in ("baseline_texts", "reuse_texts")}, ensure_ascii=True, indent=2))
print("Baseline outputs:")
for idx, text in enumerate(baseline_texts):
    print(f"[{idx}] {text.strip() if text.strip() else '<empty>'}")
print("Reuse outputs:")
for idx, text in enumerate(reuse_texts):
    print(f"[{idx}] {text.strip() if text.strip() else '<empty>'}")
