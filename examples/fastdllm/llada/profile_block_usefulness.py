"""
Profile request- and step-dependent block usefulness for Fast-dLLM LLaDA.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  python -u examples/fastdllm/llada/profile_block_usefulness.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import transformers

import dllm


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


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    input_path: str | None = None
    prompt: str = "Explain why caching helps diffusion language model serving."
    limit: int = 8
    seed: int = 42
    output_prefix: str = "artifacts/block_usefulness/profile"

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
sampler = dllm.pipelines.fastdllm.llada.FastdLLMLLaDASampler(
    model=model, tokenizer=tokenizer
)
profiler = dllm.pipelines.fastdllm.llada.BlockUsefulnessProfiler()

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

outputs = sampler.sample(
    inputs,
    config=sampler_config,
    return_dict=True,
    block_observer=profiler,
)

save_paths = profiler.save(script_args.output_prefix)
decoded = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs.tolist())
summary = profiler.build_summary()

print(f"Saved records: {save_paths['records_jsonl']}")
print(f"Saved summary: {save_paths['summary_json']}")
print(f"Collected {summary['num_records']} request-step-layer records.")
print("Top 10 smallest inter-step deltas (candidate converged blocks):")
rows = [
    row for row in profiler.records if row["inter_step_output_delta_norm"] is not None
]
rows.sort(key=lambda row: row["inter_step_output_delta_norm"])
for row in rows[:10]:
    print(
        json.dumps(
            {
                "request_index": row["request_index"],
                "block_index": row["block_index"],
                "layer_index": row["layer_index"],
                "step_index": row["step_index"],
                "cache_mode": row["cache_mode"],
                "inter_step_output_delta_norm": row["inter_step_output_delta_norm"],
                "inter_step_output_cosine": row["inter_step_output_cosine"],
                "layer_latency_ms": row["layer_latency_ms"],
            },
            ensure_ascii=True,
        )
    )

print("Decoded outputs:")
for idx, text in enumerate(decoded):
    print(f"[{idx}] {text.strip() if text.strip() else '<empty>'}")
