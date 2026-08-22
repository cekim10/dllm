"""
reference: https://github.com/NVlabs/Fast-dLLM/blob/main/llada/generate.py
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, List, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


def _trim_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    upto: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Keep only KV up to sequence index `upto` (exclusive) along seq_len dim (-2).
    Assumes each K/V is shaped like [B, H, S, D].
    """
    new_pkv = []
    for layer_kv in past_key_values:
        # layer_kv is usually (k, v)
        new_layer = tuple(t[:, :, :upto] for t in layer_kv)
        new_pkv.append(new_layer)  # type: ignore[arg-type]
    return new_pkv


def _get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,  # (B, L) bool
    x: torch.Tensor,  # (B, L) long
    num_transfer_tokens: Optional[torch.Tensor] = None,  # (B,) long (top-k mode)
    threshold: Optional[float] = None,  # threshold mode
    factor: Optional[float] = None,  # dynamic mode (highest priority)
    transfer_bias: Optional[torch.Tensor] = None,  # (B, L) additive score bias
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x0:            (B, L) long — proposed tokens
        transfer_index:(B, L) bool — which positions to update

    Priority:
      if factor is not None: dynamic schedule
      elif threshold is not None: threshold mode
      else: top-k mode (num_transfer_tokens required)
    """
    # 1) Propose tokens (greedy / gumbel-max)
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

    # 2) Confidence (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float32), dim=-1)
        conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
            -1
        )  # (B, L) float32
    elif remasking == "random":
        conf = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    # Only propose changes on masked positions
    x0 = torch.where(mask_index, x0, x)

    # Use a very negative value for non-mask positions so they never get selected
    neg = torch.finfo(conf.dtype).min
    confidence = torch.where(
        mask_index, conf, torch.tensor(neg, device=conf.device, dtype=conf.dtype)
    )  # (B, L)
    if transfer_bias is not None:
        confidence = confidence + transfer_bias.to(
            device=confidence.device,
            dtype=confidence.dtype,
        )

    # --------------------------
    # A) Dynamic factor schedule
    # --------------------------
    if factor is not None:
        B, L = confidence.shape
        values, idx = torch.sort(confidence, dim=1, descending=True)  # (B, L)

        # rank r = 1..L : thr[r] = 1 - factor/(r+1), but force rank-1 always selectable with -1
        ranks = torch.arange(
            1, L + 1, device=confidence.device, dtype=values.dtype
        )  # (L,)
        factor_t = torch.tensor(
            float(factor), device=confidence.device, dtype=values.dtype
        )
        thr = 1.0 - (factor_t / (ranks + 1.0))  # (L,)
        thr[0] = -1.0

        accept = values >= thr.unsqueeze(0)  # (B, L) bool
        k = accept.sum(dim=1).to(torch.long)  # (B,)

        # never select more than masked count
        n_masked = mask_index.sum(dim=1).to(torch.long)
        k = torch.minimum(k, n_masked)

        cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        select_sorted = cols < k.unsqueeze(1)  # (B, L)

        transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
        transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
        transfer_index = transfer_int.bool() & mask_index
        return x0, transfer_index

    # --------------------------
    # B) Threshold mode
    # --------------------------
    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)

        # force at least one transfer per row if there is any mask and none selected
        has_mask = mask_index.any(dim=1)  # (B,)
        selected = transfer_index.any(dim=1)  # (B,)
        need_force = has_mask & (~selected)
        if need_force.any():
            max_idx = torch.argmax(
                confidence, dim=1, keepdim=True
            )  # (B,1) — safe because non-masks are neg
            force = torch.zeros_like(transfer_index).scatter_(1, max_idx, True)
            transfer_index = (transfer_index | force) & mask_index

        return x0, transfer_index

    # --------------------------
    # C) Top-k (quota) mode
    # --------------------------
    if num_transfer_tokens is None:
        raise ValueError(
            "num_transfer_tokens must be provided when threshold is None and factor is None"
        )

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)

    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device
    )
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)  # (B, L)
    B, L = confidence.shape

    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1)  # (B, L)

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


@dataclass
class FastdLLMLLaDASamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = None
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0  # Unused within Fast-dLLM
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False

    # Can be "prefix" or "dual" or None
    use_cache: str | None = None

    # Remasking knobs (match NVLabs generate.py behavior)
    threshold: float | None = None
    factor: float | None = None


@dataclass
class FastdLLMLLaDAResumeState:
    """Per-request state captured immediately before a refinement step.

    Immutable request configuration remains in ``FastdLLMLLaDASamplerConfig``.
    Optional fields let the validation harness remove and reconstruct candidate
    state components instead of assuming every Python local is required.
    """

    x: torch.Tensor
    prompt_lens: tuple[int, ...]
    block_index: int | None
    inner_step: int | None
    attention_mask: torch.Tensor | None = None
    num_transfer_tokens: torch.Tensor | None = None
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] | None = None
    replace_position: torch.Tensor | None = None
    cache_mode: str = "none"


class FastdLLMLLaDAPaused(RuntimeError):
    """Control-flow exception carrying a real sampler checkpoint."""

    def __init__(self, state: FastdLLMLLaDAResumeState):
        super().__init__(
            f"Fast LLaDA paused before block={state.block_index}, "
            f"inner_step={state.inner_step}"
        )
        self.state = state


@dataclass
class FastdLLMLLaDASampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]], torch.Tensor],
        config: Optional[FastdLLMLLaDASamplerConfig] = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fast-dLLM v1 sampler.
        Supports:
          - use_cache=None: baseline (no cache)
          - use_cache="prefix": prefix cache
          - use_cache="dual": dual cache (requires model forward supports replace_position)
        """
        if config is None:
            config = FastdLLMLLaDASamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        use_cache = kwargs.get("use_cache", config.use_cache)
        threshold = kwargs.get("threshold", config.threshold)
        factor = kwargs.get("factor", config.factor)
        block_observer = kwargs.get("block_observer")
        model_call_observer = kwargs.get("model_call_observer")
        transfer_bias = kwargs.get("transfer_bias")
        canvas_update_hook = kwargs.get("canvas_update_hook")
        resume_state: FastdLLMLLaDAResumeState | None = kwargs.get("resume_state")
        pause_at: tuple[int, int] | None = kwargs.get("pause_at")
        step_observer: Callable[[dict[str, Any]], None] | None = kwargs.get(
            "step_observer"
        )

        if use_cache == "none":
            use_cache = None
        if use_cache not in (None, "prefix", "dual"):
            raise RuntimeError(
                f"Unknown use_cache mode: {use_cache}. Expected None, 'prefix', or 'dual'."
            )

        assert block_size >= 1
        assert steps >= 1
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Normalize inputs -> list[1D LongTensor] -----
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            inputs_list = [
                row.to(device=self.model.device, dtype=torch.long) for row in inputs
            ]
        else:
            # list of lists or list of tensors
            if len(inputs) == 0:
                raise ValueError("inputs is empty")
            if isinstance(inputs[0], list):
                inputs_list = [
                    torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs  # type: ignore[arg-type]
                ]
            else:
                inputs_list = [p.to(device=self.model.device, dtype=torch.long) for p in inputs]  # type: ignore[arg-type]

        prompt_lens = [p.shape[0] for p in inputs_list]
        B = len(inputs_list)
        max_prompt_len = max(prompt_lens)

        # If right_shift_logits and a sequence has length 0, replace that sequence with [bos] (match your MDLM style)
        if right_shift_logits:
            fixed = []
            for p in inputs_list:
                if p.numel() == 0:
                    fixed.append(
                        torch.tensor(
                            [bos_id], device=self.model.device, dtype=torch.long
                        )
                    )
                else:
                    fixed.append(p)
            inputs_list = fixed
            prompt_lens = [p.shape[0] for p in inputs_list]
            max_prompt_len = max(prompt_lens)

        # determine final T
        if max_new_tokens is not None:
            if max_length is None:
                max_length = max_prompt_len + max_new_tokens
            else:
                # respect explicit max_length
                max_new_tokens = max_length - max_prompt_len
        else:
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be set.")
            max_new_tokens = max_length - max_prompt_len

        T = int(max_length)

        # ----- Build canvas x and attention_mask -----
        # x is right-padded with EOS; prompt left-aligned; generation tail initialized as [MASK]
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)

        for i, p in enumerate(inputs_list):
            pl = p.shape[0]
            x[i, :pl] = p
            gen_end = min(pl + max_new_tokens, T)
            x[i, pl:gen_end] = mask_id
            attention_mask[i, :gen_end] = 1

        if resume_state is not None:
            if tuple(prompt_lens) != tuple(resume_state.prompt_lens):
                raise ValueError(
                    "resume_state prompt lengths do not match the supplied request: "
                    f"state={resume_state.prompt_lens}, request={tuple(prompt_lens)}"
                )
            expected_cache_mode = "none" if use_cache is None else use_cache
            if resume_state.cache_mode != expected_cache_mode:
                raise ValueError(
                    "resume_state cache mode does not match sampler configuration: "
                    f"state={resume_state.cache_mode!r}, config={expected_cache_mode!r}"
                )
            if tuple(resume_state.x.shape) != tuple(x.shape):
                raise ValueError(
                    "resume_state canvas shape does not match request configuration: "
                    f"state={tuple(resume_state.x.shape)}, expected={tuple(x.shape)}"
                )
            x = resume_state.x.to(device=self.model.device, dtype=torch.long).clone()
            if resume_state.attention_mask is not None:
                attention_mask = resume_state.attention_mask.to(
                    device=self.model.device, dtype=torch.long
                ).clone()

        histories = [x.clone()] if return_dict else None
        observer_call_index = 0

        # ----- Block scheduling -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        canvas_update_index = 0

        # Cache modes assume a single shared prompt length (like NVLabs reference code)
        # Fast-dLLM cache modes require batchsize = 1 or equal prompt lengths
        if use_cache is None:
            prompt_len = None
        else:
            if len(set(prompt_lens)) != 1:
                raise ValueError(
                    f"use_cache={use_cache!r} requires equal prompt lengths in batch. "
                    f"Got prompt_lens={prompt_lens}. "
                    f"Either batch by prompt length or set use_cache=None."
                )
            else:
                prompt_len = prompt_lens[0]

        # Helper: apply token suppressions to logits (in-place)
        def _apply_suppressions(logits_: torch.Tensor):
            if suppress_tokens:
                for tid in suppress_tokens:
                    logits_[:, :, tid] = -torch.inf
            if begin_suppress_tokens:
                # Simple interpretation: always suppress these tokens (you can specialize if needed)
                for tid in begin_suppress_tokens:
                    logits_[:, :, tid] = -torch.inf

        def _make_observer_context(
            *,
            phase: str,
            cache_mode: str,
            block_index: int,
            step_index: int,
            block_ranges: list[tuple[int, int]],
            mask_counts: list[int],
        ) -> dict[str, Any] | None:
            nonlocal observer_call_index
            if block_observer is None and model_call_observer is None:
                return None
            context = {
                "phase": phase,
                "cache_mode": cache_mode,
                "block_index": int(block_index),
                "step_index": int(step_index),
                "steps_per_block": int(steps_per_block),
                "num_blocks": int(num_blocks),
                "block_size": int(block_size),
                "block_ranges": tuple(
                    (int(start), int(end)) for start, end in block_ranges
                ),
                "mask_counts": tuple(int(count) for count in mask_counts),
                "prompt_lens": tuple(int(length) for length in prompt_lens),
                "model_call_index": int(observer_call_index),
                "sequence_length": int(T),
            }
            observer_call_index += 1
            return context

        def _sync_device() -> None:
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)

        def _call_model(**model_kwargs):
            if model_call_observer is None:
                return self.model(**model_kwargs)

            context = dict(model_kwargs.get("observer_context") or {})
            input_tensor = model_kwargs.get("input_ids")
            if input_tensor is None:
                input_tensor = model_kwargs.get("inputs_embeds")
            if input_tensor is not None:
                context["model_batch_size"] = int(input_tensor.shape[0])
                context["model_query_length"] = int(input_tensor.shape[1])
            attention_mask = model_kwargs.get("attention_mask")
            if attention_mask is not None:
                context["attention_mask_tokens"] = int(attention_mask.sum().item())
                context["attention_mask_length"] = int(attention_mask.shape[-1])
            past_key_values = model_kwargs.get("past_key_values")
            context["has_past_key_values"] = past_key_values is not None
            if past_key_values is not None and len(past_key_values) > 0:
                context["past_length"] = int(past_key_values[0][0].shape[-2])
            replace_position = model_kwargs.get("replace_position")
            if replace_position is not None:
                context["replace_position_count"] = int(replace_position.sum().item())
            context["use_cache"] = bool(model_kwargs.get("use_cache", False))

            _sync_device()
            if self.model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.model.device)
                memory_before = torch.cuda.memory_allocated(self.model.device)
            else:
                memory_before = 0
            start = time.perf_counter()
            output = self.model(**model_kwargs)
            _sync_device()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if self.model.device.type == "cuda":
                memory_peak = torch.cuda.max_memory_allocated(self.model.device)
                memory_after = torch.cuda.memory_allocated(self.model.device)
            else:
                memory_peak = 0
                memory_after = 0
            context["model_call_latency_ms"] = float(elapsed_ms)
            context["memory_before_bytes"] = int(memory_before)
            context["memory_after_bytes"] = int(memory_after)
            context["memory_peak_bytes"] = int(memory_peak)
            context["memory_peak_delta_bytes"] = int(max(0, memory_peak - memory_before))
            model_call_observer(context)
            return output

        def _bias_slice(start: int, end: int | None = None) -> torch.Tensor | None:
            if transfer_bias is None:
                return None
            if end is None:
                return transfer_bias[:, start:]
            return transfer_bias[:, start:end]

        def _apply_canvas_update_hook(
            x_: torch.Tensor,
            *,
            phase: str,
            cache_mode: str,
            block_index: int,
            step_index: int,
        ) -> torch.Tensor:
            nonlocal canvas_update_index
            if canvas_update_hook is None:
                canvas_update_index += 1
                return x_
            context = {
                "phase": phase,
                "cache_mode": cache_mode,
                "block_index": int(block_index),
                "step_index": int(step_index),
                "canvas_update_index": int(canvas_update_index),
            }
            canvas_update_index += 1
            hooked = canvas_update_hook(x_, context)
            return x_ if hooked is None else hooked

        cache_mode_label = "none" if use_cache is None else use_cache

        def _move_past_key_values(
            values: List[Tuple[torch.Tensor, torch.Tensor]] | None,
        ) -> List[Tuple[torch.Tensor, torch.Tensor]] | None:
            if values is None:
                return None
            return [
                tuple(tensor.to(device=self.model.device) for tensor in layer)  # type: ignore[misc]
                for layer in values
            ]

        def _derive_block_index() -> int:
            for candidate in range(num_blocks):
                for row, pl in enumerate(prompt_lens):
                    start = pl + candidate * block_size
                    end = min(start + block_size, pl + max_new_tokens, T)
                    if start < end and (x[row, start:end] == mask_id).any():
                        return candidate
            return num_blocks

        start_block = 0
        if resume_state is not None:
            start_block = (
                _derive_block_index()
                if resume_state.block_index is None
                else int(resume_state.block_index)
            )

        def _checkpoint_state(
            *,
            block_index: int,
            inner_step: int,
            schedule: torch.Tensor | None,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] | None,
            replace_position: torch.Tensor | None,
        ) -> FastdLLMLLaDAResumeState:
            return FastdLLMLLaDAResumeState(
                x=x.detach(),
                prompt_lens=tuple(int(length) for length in prompt_lens),
                attention_mask=attention_mask.detach(),
                block_index=int(block_index),
                inner_step=int(inner_step),
                num_transfer_tokens=(
                    None if schedule is None else schedule.detach()
                ),
                past_key_values=past_key_values,
                replace_position=(
                    None if replace_position is None else replace_position.detach()
                ),
                cache_mode=cache_mode_label,
            )

        def _maybe_pause(
            *,
            block_index: int,
            inner_step: int,
            schedule: torch.Tensor | None,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
            replace_position: torch.Tensor | None = None,
        ) -> None:
            if pause_at != (block_index, inner_step):
                return
            raise FastdLLMLLaDAPaused(
                _checkpoint_state(
                    block_index=block_index,
                    inner_step=inner_step,
                    schedule=schedule,
                    past_key_values=past_key_values,
                    replace_position=replace_position,
                )
            )

        def _observe_step(
            *,
            phase: str,
            block_index: int,
            inner_step: int,
            transfer_index: torch.Tensor,
        ) -> None:
            if step_observer is None:
                return
            step_observer(
                {
                    "phase": phase,
                    "cache_mode": cache_mode_label,
                    "block_index": int(block_index),
                    "inner_step": int(inner_step),
                    "x": x.detach(),
                    "transfer_index": transfer_index.detach(),
                }
            )

        # =============================
        # Main block loop
        # =============================
        for b in range(start_block, num_blocks):
            # Compute block boundaries
            if prompt_len is not None:
                # cache modes: shared boundaries
                s = prompt_len + b * block_size
                e = min(s + block_size, prompt_len + max_new_tokens, T)
                if s >= e:
                    continue
                block_len = e - s

                # Build block_mask_index for scheduling (B, block_size), padded with False
                block_mask_index = torch.zeros(
                    (B, block_size), dtype=torch.bool, device=x.device
                )
                block_mask_index[:, :block_len] = x[:, s:e] == mask_id

            else:
                # no-cache mode: per-sample boundaries
                # Build a block_mask_index (B, block_size) with per-sample widths
                block_mask_index = torch.zeros(
                    (B, block_size), dtype=torch.bool, device=x.device
                )
                widths = []
                for j in range(B):
                    start_j = prompt_lens[j] + b * block_size
                    end_j = min(
                        start_j + block_size, prompt_lens[j] + max_new_tokens, T
                    )
                    width_j = max(0, end_j - start_j)
                    widths.append((start_j, end_j, width_j))
                    if width_j > 0:
                        block_mask_index[j, :width_j] = x[j, start_j:end_j] == mask_id

            resume_here = resume_state is not None and b == start_block
            resume_inner_step = 0
            if resume_here:
                if resume_state.inner_step is not None:
                    resume_inner_step = int(resume_state.inner_step)
                elif threshold is not None or factor is not None:
                    if prompt_len is not None:
                        block_is_untouched = bool(
                            (block_mask_index.sum(dim=1) == block_len).all()
                        )
                    else:
                        block_is_untouched = all(
                            int(block_mask_index[row].sum().item()) == width
                            for row, (_, _, width) in enumerate(widths)
                        )
                    if block_is_untouched:
                        resume_inner_step = 0
                    else:
                        raise ValueError(
                            "inner_step is not derivable from a threshold/dynamic canvas"
                        )
                else:
                    # In deterministic quota mode, completed transfer count maps
                    # exactly to one cumulative quota prefix.
                    original_mask = torch.zeros_like(block_mask_index)
                    if prompt_len is not None:
                        original_mask[:, :block_len] = True
                    else:
                        for row, (_, _, width) in enumerate(widths):
                            original_mask[row, :width] = True
                    derivation_schedule = (
                        resume_state.num_transfer_tokens.to(device=x.device)
                        if resume_state.num_transfer_tokens is not None
                        else get_num_transfer_tokens(
                            mask_index=original_mask,
                            steps=steps_per_block,
                            scheduler=self.scheduler,
                            stochastic=False,
                        )
                    )
                    completed = original_mask.sum(dim=1) - block_mask_index.sum(dim=1)
                    if bool((completed == 0).all()):
                        resume_inner_step = 0
                    else:
                        cumulative = derivation_schedule.cumsum(dim=1)
                        matches = (cumulative == completed.unsqueeze(1)).all(dim=0)
                        indices = matches.nonzero(as_tuple=False).flatten()
                        if indices.numel() == 0:
                            raise ValueError(
                                "inner_step cannot be reconstructed from quota progress"
                            )
                        resume_inner_step = int(indices[0].item()) + 1

            if resume_inner_step == 0:
                _maybe_pause(
                    block_index=b,
                    inner_step=0,
                    schedule=None,
                )

            # Quotas are based on the block-start mask. For a deterministic
            # mid-block resume they can be reconstructed from request metadata.
            schedule_mask_index = block_mask_index
            if resume_here and resume_inner_step > 0:
                schedule_mask_index = torch.zeros_like(block_mask_index)
                if prompt_len is not None:
                    schedule_mask_index[:, :block_len] = True
                else:
                    for row, (_, _, width) in enumerate(widths):
                        schedule_mask_index[row, :width] = True

            if (
                resume_here
                and resume_state.num_transfer_tokens is not None
            ):
                num_transfer_tokens = resume_state.num_transfer_tokens.to(
                    device=x.device
                )
            else:
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=schedule_mask_index,
                    steps=steps_per_block,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
            effective_steps = num_transfer_tokens.size(1)

            # -------------------------
            # Mode 1: No cache
            # -------------------------
            if use_cache is None:
                i = resume_inner_step
                while True:
                    # mask only within current block (per-sample)
                    mask_allowed = torch.zeros_like(x, dtype=torch.bool)

                    for j in range(B):
                        start_j, end_j, width_j = widths[j]
                        if width_j > 0:
                            # only masked positions in current block
                            mask_allowed[j, start_j:end_j] = (
                                x[j, start_j:end_j] == mask_id
                            )

                    if mask_allowed.sum() == 0:
                        break

                    if i > 0:
                        _maybe_pause(
                            block_index=b,
                            inner_step=i,
                            schedule=num_transfer_tokens,
                        )

                    block_ranges = [(start_j, end_j) for start_j, end_j, _ in widths]
                    mask_counts = [
                        int(mask_allowed[j, start_j:end_j].sum().item())
                        for j, (start_j, end_j, _) in enumerate(widths)
                    ]
                    out = _call_model(
                        input_ids=x,
                        attention_mask=attention_mask,
                        output_hidden_states=block_observer is not None,
                        block_observer=block_observer,
                        observer_context=_make_observer_context(
                            phase="refine",
                            cache_mode="none",
                            block_index=b,
                            step_index=i,
                            block_ranges=block_ranges,
                            mask_counts=mask_counts,
                        ),
                    )
                    logits = out.logits
                    _apply_suppressions(logits)

                    if right_shift_logits:
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    quota = None if threshold is not None else num_transfer_tokens[:, i]
                    x0, transfer_idx = _get_transfer_index(
                        logits=logits,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_allowed,
                        x=x,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                        transfer_bias=transfer_bias,
                    )

                    x = torch.where(transfer_idx, x0, x)
                    x = _apply_canvas_update_hook(
                        x,
                        phase="refine",
                        cache_mode="none",
                        block_index=b,
                        step_index=i,
                    )
                    _observe_step(
                        phase="refine",
                        block_index=b,
                        inner_step=i,
                        transfer_index=transfer_idx,
                    )
                    i += 1

                    if histories is not None:
                        histories.append(x.clone())

                continue  # next block

            # -------------------------
            # Mode 2: Prefix cache
            # -------------------------
            if use_cache == "prefix":
                shared_ranges = [(s, e) for _ in range(B)]
                past_key_values = (
                    _move_past_key_values(resume_state.past_key_values)
                    if resume_here
                    else None
                )

                if resume_inner_step == 0:
                    # Warm cache on full x once per block and consume step 0.
                    out_full = _call_model(
                        input_ids=x,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_hidden_states=block_observer is not None,
                        block_observer=block_observer,
                        observer_context=_make_observer_context(
                            phase="warmup",
                            cache_mode="prefix",
                            block_index=b,
                            step_index=0,
                            block_ranges=shared_ranges,
                            mask_counts=[
                                int((x[row, s:e] == mask_id).sum().item())
                                for row in range(B)
                            ],
                        ),
                    )
                    logits_full = out_full.logits
                    past_key_values = out_full.past_key_values

                    _apply_suppressions(logits_full)
                    if right_shift_logits:
                        logits_full = torch.cat(
                            [logits_full[:, :1], logits_full[:, :-1]], dim=1
                        )

                    mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                    mask_allowed[:, s:e] = x[:, s:e] == mask_id
                    if mask_allowed.sum() > 0:
                        quota = (
                            None
                            if threshold is not None
                            else num_transfer_tokens[:, 0]
                        )
                        x0, transfer_idx = _get_transfer_index(
                            logits=logits_full,
                            temperature=temperature,
                            remasking=remasking,
                            mask_index=mask_allowed,
                            x=x,
                            num_transfer_tokens=quota,
                            threshold=threshold,
                            factor=factor,
                            transfer_bias=transfer_bias,
                        )

                        x = torch.where(transfer_idx, x0, x)
                        x = _apply_canvas_update_hook(
                            x,
                            phase="warmup",
                            cache_mode="prefix",
                            block_index=b,
                            step_index=0,
                        )
                        _observe_step(
                            phase="warmup",
                            block_index=b,
                            inner_step=0,
                            transfer_index=transfer_idx,
                        )
                        if histories is not None:
                            histories.append(x.clone())

                    if past_key_values is None:
                        raise RuntimeError(
                            "Model did not return past_key_values with use_cache=True"
                        )
                    past_key_values = _trim_past_key_values(past_key_values, s)
                elif past_key_values is None:
                    # Prefix K/V at deeper layers can depend on the bidirectional
                    # suffix. Recreate the exact block-start canvas before trim.
                    block_start_x = x.clone()
                    block_start_x[:, s:e] = mask_id
                    rebuilt = _call_model(
                        input_ids=block_start_x,
                        attention_mask=attention_mask,
                        use_cache=True,
                    ).past_key_values
                    if rebuilt is None:
                        raise RuntimeError(
                            "Model did not return past_key_values while rebuilding prefix cache"
                        )
                    past_key_values = _trim_past_key_values(rebuilt, s)

                # Refinement steps on suffix with prefix cache
                i = max(1, resume_inner_step)
                while True:
                    if (x[:, s:e] == mask_id).sum() == 0:
                        break

                    x_suffix = x[:, s:]  # (B, T-s)
                    mask_suffix = x_suffix == mask_id
                    # restrict to current block only
                    if x_suffix.size(1) > block_len:
                        mask_suffix[:, block_len:] = False

                    if mask_suffix.sum() == 0:
                        break

                    _maybe_pause(
                        block_index=b,
                        inner_step=i,
                        schedule=num_transfer_tokens,
                        past_key_values=past_key_values,
                    )

                    suffix_ranges = [
                        (0, min(block_len, int(x_suffix.shape[1]))) for _ in range(B)
                    ]
                    out_suf = _call_model(
                        input_ids=x_suffix,
                        attention_mask=attention_mask,  # full-length mask is OK for this model
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=block_observer is not None,
                        block_observer=block_observer,
                        observer_context=_make_observer_context(
                            phase="refine",
                            cache_mode="prefix",
                            block_index=b,
                            step_index=i,
                            block_ranges=suffix_ranges,
                            mask_counts=[
                                int(mask_suffix[row, :block_len].sum().item())
                                for row in range(B)
                            ],
                        ),
                    )
                    logits_suf = out_suf.logits
                    _apply_suppressions(logits_suf)

                    if right_shift_logits:
                        logits_suf = torch.cat(
                            [logits_suf[:, :1], logits_suf[:, :-1]], dim=1
                        )

                    quota = (
                        None
                        if (threshold is not None or factor is not None)
                        else num_transfer_tokens[:, i]
                    )
                    x0_suf, transfer_suf = _get_transfer_index(
                        logits=logits_suf,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_suffix,
                        x=x_suffix,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                        transfer_bias=_bias_slice(s),
                    )

                    x_suffix_new = torch.where(transfer_suf, x0_suf, x_suffix)
                    x = torch.cat([x[:, :s], x_suffix_new], dim=1)
                    x = _apply_canvas_update_hook(
                        x,
                        phase="refine",
                        cache_mode="prefix",
                        block_index=b,
                        step_index=i,
                    )
                    _observe_step(
                        phase="refine",
                        block_index=b,
                        inner_step=i,
                        transfer_index=transfer_suf,
                    )

                    i += 1
                    if histories is not None:
                        histories.append(x.clone())

                continue  # next block

            # -------------------------
            # Mode 3: Dual cache
            # -------------------------
            if use_cache == "dual":
                shared_ranges = [(s, e) for _ in range(B)]
                # replace_position mask for this block (B, T)
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, s:e] = True
                past_key_values = (
                    _move_past_key_values(resume_state.past_key_values)
                    if resume_here
                    else None
                )

                if resume_inner_step == 0:
                    # Warm cache on the block-start canvas and consume step 0.
                    out_full = _call_model(
                        input_ids=x,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_hidden_states=block_observer is not None,
                        block_observer=block_observer,
                        observer_context=_make_observer_context(
                            phase="warmup",
                            cache_mode="dual",
                            block_index=b,
                            step_index=0,
                            block_ranges=shared_ranges,
                            mask_counts=[
                                int((x[row, s:e] == mask_id).sum().item())
                                for row in range(B)
                            ],
                        ),
                    )
                    logits_full = out_full.logits
                    past_key_values = out_full.past_key_values
                    if past_key_values is None:
                        raise RuntimeError(
                            "Model did not return past_key_values with use_cache=True"
                        )

                    _apply_suppressions(logits_full)
                    if right_shift_logits:
                        logits_full = torch.cat(
                            [logits_full[:, :1], logits_full[:, :-1]], dim=1
                        )

                    mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                    mask_allowed[:, s:e] = x[:, s:e] == mask_id
                    if mask_allowed.sum() > 0:
                        quota = (
                            None
                            if threshold is not None
                            else num_transfer_tokens[:, 0]
                        )
                        x0, transfer_idx = _get_transfer_index(
                            logits=logits_full,
                            temperature=temperature,
                            remasking=remasking,
                            mask_index=mask_allowed,
                            x=x,
                            num_transfer_tokens=quota,
                            threshold=threshold,
                            factor=factor,
                            transfer_bias=transfer_bias,
                        )

                        x = torch.where(transfer_idx, x0, x)
                        x = _apply_canvas_update_hook(
                            x,
                            phase="warmup",
                            cache_mode="dual",
                            block_index=b,
                            step_index=0,
                        )
                        _observe_step(
                            phase="warmup",
                            block_index=b,
                            inner_step=0,
                            transfer_index=transfer_idx,
                        )
                        if histories is not None:
                            histories.append(x.clone())
                elif past_key_values is None:
                    # The dual cache is the cache of the block-start canvas.
                    # Recreate that canvas by remasking the current block.
                    block_start_x = x.clone()
                    block_start_x[:, s:e] = mask_id
                    past_key_values = _call_model(
                        input_ids=block_start_x,
                        attention_mask=attention_mask,
                        use_cache=True,
                    ).past_key_values
                    if past_key_values is None:
                        raise RuntimeError(
                            "Model did not return past_key_values while rebuilding dual cache"
                        )

                # Use for loop here for better compilation performance according to original implementation
                for i_step in range(max(1, resume_inner_step), effective_steps):
                    blk = x[:, s:e]
                    mask_blk = blk == mask_id
                    if mask_blk.sum() == 0:
                        break

                    _maybe_pause(
                        block_index=b,
                        inner_step=i_step,
                        schedule=num_transfer_tokens,
                        past_key_values=past_key_values,
                        replace_position=replace_position,
                    )

                    # This requires model forward supports replace_position (as in your first modeling_llada.py)
                    out_blk = _call_model(
                        input_ids=blk,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        replace_position=replace_position,
                        output_hidden_states=block_observer is not None,
                        block_observer=block_observer,
                        observer_context=_make_observer_context(
                            phase="refine",
                            cache_mode="dual",
                            block_index=b,
                            step_index=i_step,
                            block_ranges=[(0, int(blk.shape[1])) for _ in range(B)],
                            mask_counts=[
                                int(mask_blk[row].sum().item()) for row in range(B)
                            ],
                        ),
                    )
                    logits_blk = out_blk.logits
                    _apply_suppressions(logits_blk)

                    if right_shift_logits:
                        logits_blk = torch.cat(
                            [logits_blk[:, :1], logits_blk[:, :-1]], dim=1
                        )

                    quota = (
                        None
                        if threshold is not None
                        else num_transfer_tokens[:, i_step]
                    )
                    x0_blk, transfer_blk = _get_transfer_index(
                        logits=logits_blk,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_blk,
                        x=blk,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                        transfer_bias=_bias_slice(s, e),
                    )

                    blk_new = torch.where(transfer_blk, x0_blk, blk)
                    x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
                    x = _apply_canvas_update_hook(
                        x,
                        phase="refine",
                        cache_mode="dual",
                        block_index=b,
                        step_index=i_step,
                    )
                    _observe_step(
                        phase="refine",
                        block_index=b,
                        inner_step=i_step,
                        transfer_index=transfer_blk,
                    )

                    if histories is not None:
                        histories.append(x.clone())

                continue  # next block

            raise ValueError(f"Unknown use_cache mode: {use_cache!r}")

        # ----- Output format -----
        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]]],
        config: FastdLLMLLaDASamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
