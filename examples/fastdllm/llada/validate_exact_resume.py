"""
Validate exact arbitrary-step resume for Fast-dLLM LLaDA no-cache decoding.

Run from the repository root on a GPU machine:

  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  python -u examples/fastdllm/llada/validate_exact_resume.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --prompt "Explain why request preemption is hard for diffusion LMs." \
    --steps 128 \
    --max_new_tokens 128 \
    --block_size 32 \
    --use_cache none \
    --output_prefix artifacts/preemption_state/llada_resume_none_s128

This is a kill-test utility, not a serving scheduler. It isolates the
per-request generation state needed to stop and resume a single request exactly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.fastdllm.llada import FastdLLMLLaDAConfig
from dllm.pipelines.fastdllm.llada.sampler import (
    _get_transfer_index,
)


@dataclass
class ResumeSamplerConfig:
    steps: int
    max_new_tokens: int
    block_size: int
    temperature: float
    remasking: str
    stochastic_transfer: bool
    threshold: float | None
    factor: float | None
    right_shift_logits: bool


@dataclass
class ResumeState:
    x: torch.Tensor
    prompt_lens: tuple[int, ...]
    attention_mask: torch.Tensor
    block_index: int
    inner_step: int
    current_schedule: torch.Tensor | None
    config: ResumeSamplerConfig


@dataclass
class StepRecord:
    global_step: int
    block_index: int
    inner_step: int
    block_start: int
    block_end: int
    mask_count_before: int
    transfer_count: int
    x_after: torch.Tensor
    transfer_index: torch.Tensor
    model_latency_ms: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _state_bytes(state: ResumeState) -> dict[str, int]:
    return {
        "x": _tensor_bytes(state.x),
        "attention_mask": _tensor_bytes(state.attention_mask),
        "current_schedule": _tensor_bytes(state.current_schedule),
        "metadata_estimate": 8
        * (
            len(state.prompt_lens)
            + len(fields(ResumeSamplerConfig))
            + 2  # block_index, inner_step
        ),
    }


def _state_bytes_total(state: ResumeState) -> int:
    return sum(_state_bytes(state).values())


def _clone_state(state: ResumeState) -> ResumeState:
    return ResumeState(
        x=state.x.clone(),
        prompt_lens=tuple(state.prompt_lens),
        attention_mask=state.attention_mask.clone(),
        block_index=int(state.block_index),
        inner_step=int(state.inner_step),
        current_schedule=(
            None if state.current_schedule is None else state.current_schedule.clone()
        ),
        config=copy.deepcopy(state.config),
    )


def _serialize_state(state: ResumeState) -> dict[str, Any]:
    """Create a real detached checkpoint payload.

    Tensor payloads are cloned to CPU to force D2H serialization. The caller can
    move them back during restore. This keeps Phase 2 honest without starting a
    full save/restore benchmark.
    """
    payload = {
        "x": state.x.detach().cpu().clone(),
        "prompt_lens": tuple(state.prompt_lens),
        "attention_mask": state.attention_mask.detach().cpu().clone(),
        "block_index": int(state.block_index),
        "inner_step": int(state.inner_step),
        "current_schedule": (
            None
            if state.current_schedule is None
            else state.current_schedule.detach().cpu().clone()
        ),
        "config": copy.deepcopy(state.config),
    }
    return payload


def _restore_state(payload: dict[str, Any], device: torch.device) -> ResumeState:
    return ResumeState(
        x=payload["x"].to(device=device, non_blocking=False),
        prompt_lens=tuple(int(x) for x in payload["prompt_lens"]),
        attention_mask=payload["attention_mask"].to(device=device, non_blocking=False),
        block_index=int(payload["block_index"]),
        inner_step=int(payload["inner_step"]),
        current_schedule=(
            None
            if payload["current_schedule"] is None
            else payload["current_schedule"].to(device=device, non_blocking=False)
        ),
        config=copy.deepcopy(payload["config"]),
    )


class NoCacheLLaDAResumeRunner:
    """Single-request/stateful no-cache Fast-dLLM LLaDA loop."""

    def __init__(self, model: Any, tokenizer: Any, scheduler: Any, state: ResumeState):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.state = state
        self.global_step = 0

    @classmethod
    def from_inputs(
        cls,
        *,
        model: Any,
        tokenizer: Any,
        scheduler: Any,
        inputs: torch.Tensor,
        config: ResumeSamplerConfig,
    ) -> "NoCacheLLaDAResumeRunner":
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(device=model.device, dtype=torch.long)
        prompt_lens = tuple(int(row.numel()) for row in inputs)
        max_prompt_len = max(prompt_lens)
        max_length = max_prompt_len + int(config.max_new_tokens)
        B = int(inputs.shape[0])
        T = int(max_length)
        x = torch.full(
            (B, T),
            tokenizer.eos_token_id,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=model.device)

        for row, prompt in enumerate(inputs):
            prompt_len = int(prompt.numel())
            x[row, :prompt_len] = prompt
            gen_end = min(prompt_len + int(config.max_new_tokens), T)
            x[row, prompt_len:gen_end] = tokenizer.mask_token_id
            attention_mask[row, :gen_end] = 1

        state = ResumeState(
            x=x,
            prompt_lens=prompt_lens,
            attention_mask=attention_mask,
            block_index=0,
            inner_step=0,
            current_schedule=None,
            config=config,
        )
        return cls(model=model, tokenizer=tokenizer, scheduler=scheduler, state=state)

    def _num_blocks(self) -> int:
        return math.ceil(self.state.config.max_new_tokens / self.state.config.block_size)

    def _steps_per_block(self) -> int:
        return math.ceil(self.state.config.steps / self._num_blocks())

    def _block_ranges(self, block_index: int) -> list[tuple[int, int, int]]:
        ranges = []
        T = int(self.state.x.shape[1])
        for prompt_len in self.state.prompt_lens:
            start = int(prompt_len) + block_index * self.state.config.block_size
            end = min(
                start + self.state.config.block_size,
                int(prompt_len) + self.state.config.max_new_tokens,
                T,
            )
            ranges.append((start, end, max(0, end - start)))
        return ranges

    def _ensure_schedule(self) -> None:
        if self.state.current_schedule is not None:
            return
        block_mask = torch.zeros(
            (self.state.x.shape[0], self.state.config.block_size),
            dtype=torch.bool,
            device=self.state.x.device,
        )
        for row, (start, end, width) in enumerate(
            self._block_ranges(self.state.block_index)
        ):
            if width > 0:
                block_mask[row, :width] = (
                    self.state.x[row, start:end] == self.tokenizer.mask_token_id
                )
        self.state.current_schedule = get_num_transfer_tokens(
            mask_index=block_mask,
            steps=self._steps_per_block(),
            scheduler=self.scheduler,
            stochastic=self.state.config.stochastic_transfer,
        )

    def _advance_to_work(self) -> bool:
        while self.state.block_index < self._num_blocks():
            mask_allowed = self._mask_allowed()
            if mask_allowed.any():
                self._ensure_schedule()
                return True
            self.state.block_index += 1
            self.state.inner_step = 0
            self.state.current_schedule = None
        return False

    def _mask_allowed(self) -> torch.Tensor:
        mask_allowed = torch.zeros_like(self.state.x, dtype=torch.bool)
        for row, (start, end, width) in enumerate(
            self._block_ranges(self.state.block_index)
        ):
            if width > 0:
                mask_allowed[row, start:end] = (
                    self.state.x[row, start:end] == self.tokenizer.mask_token_id
                )
        return mask_allowed

    def step(self) -> StepRecord | None:
        if not self._advance_to_work():
            return None

        cfg = self.state.config
        mask_allowed = self._mask_allowed()
        mask_count_before = int(mask_allowed.sum().item())
        block_ranges = self._block_ranges(self.state.block_index)
        block_start = min(start for start, _, width in block_ranges if width > 0)
        block_end = max(end for _, end, width in block_ranges if width > 0)

        _sync(self.model.device)
        start_time = time.perf_counter()
        out = self.model(
            input_ids=self.state.x,
            attention_mask=self.state.attention_mask,
        )
        _sync(self.model.device)
        model_latency_ms = (time.perf_counter() - start_time) * 1000.0
        logits = out.logits
        if cfg.right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if self.state.current_schedule is None:
            raise RuntimeError("Missing current_schedule after _advance_to_work().")
        if cfg.threshold is None and cfg.factor is None:
            if self.state.inner_step >= self.state.current_schedule.shape[1]:
                quota = self.state.current_schedule[:, -1]
            else:
                quota = self.state.current_schedule[:, self.state.inner_step]
        else:
            quota = None

        x0, transfer_index = _get_transfer_index(
            logits=logits,
            temperature=cfg.temperature,
            remasking=cfg.remasking,
            mask_index=mask_allowed,
            x=self.state.x,
            num_transfer_tokens=quota,
            threshold=cfg.threshold,
            factor=cfg.factor,
            transfer_bias=None,
        )
        self.state.x = torch.where(transfer_index, x0, self.state.x)

        record = StepRecord(
            global_step=int(self.global_step),
            block_index=int(self.state.block_index),
            inner_step=int(self.state.inner_step),
            block_start=int(block_start),
            block_end=int(block_end),
            mask_count_before=mask_count_before,
            transfer_count=int(transfer_index.sum().item()),
            x_after=self.state.x.clone(),
            transfer_index=transfer_index.clone(),
            model_latency_ms=float(model_latency_ms),
        )
        self.global_step += 1
        self.state.inner_step += 1

        if not self._mask_allowed().any():
            self.state.block_index += 1
            self.state.inner_step = 0
            self.state.current_schedule = None

        return record

    def run_to_end(self) -> list[StepRecord]:
        records = []
        while True:
            record = self.step()
            if record is None:
                break
            records.append(record)
        return records

    def run_until_before_step(self, target_step: int) -> list[StepRecord]:
        records = []
        while self.global_step < target_step:
            record = self.step()
            if record is None:
                break
            records.append(record)
        return records


def _records_equal(expected: StepRecord, observed: StepRecord) -> tuple[bool, str]:
    fields_to_check = [
        "block_index",
        "inner_step",
        "block_start",
        "block_end",
        "mask_count_before",
        "transfer_count",
    ]
    for name in fields_to_check:
        if getattr(expected, name) != getattr(observed, name):
            return False, f"{name}: expected={getattr(expected, name)} observed={getattr(observed, name)}"
    if not torch.equal(expected.transfer_index, observed.transfer_index):
        return False, "transfer_index mismatch"
    if not torch.equal(expected.x_after, observed.x_after):
        return False, "x_after mismatch"
    return True, ""


def _checkpoint_indices(records: list[StepRecord]) -> list[int]:
    if not records:
        return []
    candidates = {
        0,
        max(0, len(records) // 4),
        max(0, len(records) // 2),
        max(0, (3 * len(records)) // 4),
        max(0, len(records) - 1),
    }
    # Add block-boundary-adjacent checkpoints.
    for i, record in enumerate(records):
        if i > 0 and records[i - 1].block_index != record.block_index:
            candidates.add(i - 1)  # immediately before a boundary transition
            candidates.add(i)  # immediately after entering next block
        if record.inner_step in (1, 2, 3):
            candidates.add(i)
    return sorted(i for i in candidates if 0 <= i < len(records))


def _inventory_rows(state: ResumeState) -> list[dict[str, Any]]:
    rows = []
    tensor_fields = {
        "x": state.x,
        "attention_mask": state.attention_mask,
        "current_schedule": state.current_schedule,
    }
    for name, tensor in tensor_fields.items():
        if tensor is None:
            rows.append(
                {
                    "field": name,
                    "shape": None,
                    "dtype": None,
                    "device": None,
                    "bytes": 0,
                    "changes_every_step": name == "current_schedule",
                    "required_for_exact_resume": name in ("x", "current_schedule"),
                    "required_only_for_performance": False,
                    "can_be_recomputed": name == "attention_mask",
                    "recompute_cost_known": False,
                }
            )
            continue
        rows.append(
            {
                "field": name,
                "shape": tuple(int(dim) for dim in tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": _tensor_bytes(tensor),
                "changes_every_step": name == "x",
                "required_for_exact_resume": name in ("x", "current_schedule"),
                "required_only_for_performance": False,
                "can_be_recomputed": name == "attention_mask",
                "recompute_cost_known": False,
            }
        )
    rows.extend(
        [
            {
                "field": "block_index",
                "shape": (),
                "dtype": "int",
                "device": "cpu_metadata",
                "bytes": 8,
                "changes_every_step": False,
                "required_for_exact_resume": True,
                "required_only_for_performance": False,
                "can_be_recomputed": False,
                "recompute_cost_known": False,
            },
            {
                "field": "inner_step",
                "shape": (),
                "dtype": "int",
                "device": "cpu_metadata",
                "bytes": 8,
                "changes_every_step": True,
                "required_for_exact_resume": True,
                "required_only_for_performance": False,
                "can_be_recomputed": False,
                "recompute_cost_known": False,
            },
            {
                "field": "prompt_lens/generation_metadata/config",
                "shape": (len(state.prompt_lens),),
                "dtype": "metadata",
                "device": "cpu_metadata",
                "bytes": _state_bytes(state)["metadata_estimate"],
                "changes_every_step": False,
                "required_for_exact_resume": True,
                "required_only_for_performance": False,
                "can_be_recomputed": False,
                "recompute_cost_known": False,
            },
        ]
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _derived_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def _normalize_inputs(inputs: Any) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return inputs
    return torch.as_tensor(inputs, dtype=torch.long)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Explain why request preemption is hard for diffusion language models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--stochastic_transfer", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--right_shift_logits", action="store_true")
    parser.add_argument("--use_cache", default="none", choices=["none", "prefix", "dual"])
    parser.add_argument(
        "--output_prefix",
        default="artifacts/preemption_state/llada_resume_none_s128",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.use_cache != "none":
        raise SystemExit(
            "This Phase 1/2 validator currently proves no-cache exact resume only. "
            "Run --use_cache none first; prefix/dual cache exact-performance state "
            "must be characterized after the semantic no-cache proof passes."
        )
    if args.temperature != 0.0 or args.stochastic_transfer:
        raise SystemExit(
            "Deterministic proof first: use --temperature 0 and omit --stochastic_transfer."
        )

    transformers.set_seed(args.seed)
    model_name_or_path = dllm.utils.resolve_with_base_env(
        args.model_name_or_path, "BASE_MODELS_DIR"
    )
    model_config = FastdLLMLLaDAConfig.from_pretrained(model_name_or_path)
    model = dllm.utils.get_model(
        model_name_or_path=model_name_or_path,
        config=model_config,
    ).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=model_name_or_path)

    messages = [[{"role": "user", "content": args.prompt}]]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    inputs = _normalize_inputs(inputs).to(model.device)

    cfg = ResumeSamplerConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=args.temperature,
        remasking=args.remasking,
        stochastic_transfer=args.stochastic_transfer,
        threshold=args.threshold,
        factor=args.factor,
        right_shift_logits=args.right_shift_logits,
    )

    baseline_runner = NoCacheLLaDAResumeRunner.from_inputs(
        model=model,
        tokenizer=tokenizer,
        scheduler=LinearAlphaScheduler(),
        inputs=inputs,
        config=cfg,
    )
    baseline_records = baseline_runner.run_to_end()
    final_baseline = baseline_runner.state.x.clone()
    checkpoint_indices = _checkpoint_indices(baseline_records)

    result_rows: list[dict[str, Any]] = []
    inventory_state: ResumeState | None = None
    for checkpoint_idx in checkpoint_indices:
        interrupted_runner = NoCacheLLaDAResumeRunner.from_inputs(
            model=model,
            tokenizer=tokenizer,
            scheduler=LinearAlphaScheduler(),
            inputs=inputs,
            config=cfg,
        )
        pre_records = interrupted_runner.run_until_before_step(checkpoint_idx)
        if len(pre_records) != checkpoint_idx:
            raise RuntimeError(
                f"Could not reach checkpoint step {checkpoint_idx}; reached {len(pre_records)}."
            )
        state_before = _clone_state(interrupted_runner.state)
        if inventory_state is None or _state_bytes_total(state_before) > _state_bytes_total(
            inventory_state
        ):
            inventory_state = _clone_state(state_before)
        _sync(model.device)
        ser_start = time.perf_counter()
        payload = _serialize_state(state_before)
        restored_state = _restore_state(payload, model.device)
        _sync(model.device)
        ser_ms = (time.perf_counter() - ser_start) * 1000.0

        resumed_runner = NoCacheLLaDAResumeRunner(
            model=model,
            tokenizer=tokenizer,
            scheduler=LinearAlphaScheduler(),
            state=restored_state,
        )
        resumed_runner.global_step = checkpoint_idx
        resumed_records = resumed_runner.run_to_end()

        ok = True
        divergence = ""
        expected_tail = baseline_records[checkpoint_idx:]
        if len(expected_tail) != len(resumed_records):
            ok = False
            divergence = (
                f"tail length mismatch expected={len(expected_tail)} "
                f"observed={len(resumed_records)}"
            )
        else:
            for rel_idx, (expected, observed) in enumerate(
                zip(expected_tail, resumed_records)
            ):
                equal, reason = _records_equal(expected, observed)
                if not equal:
                    ok = False
                    divergence = (
                        f"first divergent resumed step rel={rel_idx} "
                        f"global={checkpoint_idx + rel_idx}: {reason}"
                    )
                    break
        final_equal = torch.equal(final_baseline, resumed_runner.state.x)
        ok = ok and final_equal
        if not final_equal and not divergence:
            divergence = "final x mismatch"

        bytes_by_field = _state_bytes(state_before)
        result_rows.append(
            {
                "checkpoint_step": checkpoint_idx,
                "progress_fraction": checkpoint_idx / max(len(baseline_records), 1),
                "block_index": state_before.block_index,
                "inner_step": state_before.inner_step,
                "is_block_boundary_checkpoint": state_before.current_schedule is None,
                "ok": ok,
                "final_equal": final_equal,
                "baseline_tail_steps": len(expected_tail),
                "resumed_tail_steps": len(resumed_records),
                "serialize_restore_ms": ser_ms,
                "state_bytes_total": sum(bytes_by_field.values()),
                "state_bytes_x": bytes_by_field["x"],
                "state_bytes_attention_mask": bytes_by_field["attention_mask"],
                "state_bytes_current_schedule": bytes_by_field["current_schedule"],
                "divergence": divergence,
            }
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(_derived_path(output_prefix, "_checkpoints.csv"), result_rows)
    _write_csv(
        _derived_path(output_prefix, "_state_inventory.csv"),
        _inventory_rows(inventory_state or baseline_runner.state),
    )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "use_cache": args.use_cache,
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "num_baseline_steps": len(baseline_records),
        "checkpoint_count": len(checkpoint_indices),
        "all_checkpoints_ok": all(row["ok"] for row in result_rows),
        "checkpoint_indices": checkpoint_indices,
        "state_bytes_representative": _state_bytes(
            inventory_state or baseline_runner.state
        ),
        "state_bytes_representative_total": _state_bytes_total(
            inventory_state or baseline_runner.state
        ),
        "block_boundary_checkpoints": sum(
            1 for row in result_rows if row["is_block_boundary_checkpoint"]
        ),
        "decoded": tokenizer.decode(
            final_baseline[0, inputs.shape[1] :],
            skip_special_tokens=True,
        ),
        "go_judgment": (
            "CONDITIONAL_GO_EXACT_RESUME"
            if all(row["ok"] for row in result_rows)
            else "NO_GO_EXACT_RESUME_FAILED"
        ),
    }
    with _derived_path(output_prefix, "_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
