# Action Completeness Next Experiments

This file records the decisive experiments for the structured-action direction.

## 1. Call-Count Scaling

Run the native dLLM readiness experiment for 1, 2, 3, and 5 independent actions.
The expected paper signal is weak for one call and stronger as the final
serialized action moves later in AR output order.

```bash
for C in 1 2 3 5; do
  python -u examples/fastdllm/llada/test_multitool_prefetch_signals.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path examples/fastdllm/llada/multitool_prefetch_prompts_${C}call_120.jsonl \
    --limit 120 \
    --tool_latencies_ms 100,300,500,1000,2000 \
    --prompt_format action_list \
    --bias_strengths 0 \
    --steps 128 \
    --max_new_tokens 192 \
    --block_size 48 \
    --use_cache prefix \
    --threshold 0.9 \
    --output_prefix artifacts/action_completeness/multitool_llada_${C}call_scaleup

  python -u examples/fastdllm/llada/sweep_multitool_probe_cost.py \
    --requests_csv artifacts/action_completeness/multitool_llada_${C}call_scaleup_requests.csv \
    --calls_csv artifacts/action_completeness/multitool_llada_${C}call_scaleup_calls.csv \
    --tool_latencies_ms 100,300,500,1000,2000 \
    --probe_costs_ms 0,50,100,200,350,500 \
    --output_prefix artifacts/action_completeness/multitool_llada_${C}call_scaleup_probe_sweep
done
```

Decision criteria:

- Strong: speedup versus AR verified probe rises with call count, and 3/5-call
  cases stay above 1.3x at realistic effective probe cost.
- Weak: only 5-call synthetic cases improve, or final all-ready rate falls.
- No-go: one-call and multi-call look similar, meaning there is no serialization
  penalty to exploit.

## 2. AR Probe Design Space

Run auxiliary probes with different model sizes and output scopes. The goal is
to map the cost/accuracy curve; cheap probes should be faster but less
dispatchable.

```bash
for MODEL in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  SAFE_MODEL=$(echo "$MODEL" | tr '/.' '__')
  for MODE in full_actions tool_names first_call; do
    python -u examples/fastdllm/llada/benchmark_multitool_action_probe.py \
      --ar_model_name_or_path "$MODEL" \
      --input_path examples/fastdllm/llada/multitool_prefetch_prompts_3call_120.jsonl \
      --native_requests_csv artifacts/action_completeness/multitool_llada_3call_scaleup_requests.csv \
      --limit 120 \
      --probe_mode "$MODE" \
      --max_new_tokens 192 \
      --output_prefix artifacts/action_completeness/probe_${SAFE_MODEL}_${MODE}_3call_scaleup
  done
done
```

Plot:

- x-axis: probe latency.
- y-axis: full-action accuracy and first-call/tool-name accuracy.
- dLLM native point: zero extra generation, final all-ready rate, false-start
  rate, and native all-ready time from the main trajectory.

## 3. Second dLLM Model

Repeat a smaller `n=20-30` run on Dream or another available dLLM. This is not
for final numbers; it answers whether the readiness signal is LLaDA-specific.

## 4. NFE Stage Sensitivity

Run the stage-level fidelity ablation on the 3-call workflow proxy. This treats
each required tool action as one workflow stage, generates each stage with high
and low NFE once, then compares `all_high`, `low_stage_0`, `low_stage_1`,
`low_stage_2`, and `all_low` schedules.

```bash
python -u examples/fastdllm/llada/run_multitool_nfe_stage_ablation.py \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
  --input_path examples/fastdllm/llada/multitool_prefetch_prompts_3call_120.jsonl \
  --limit 20 \
  --high_steps 128 \
  --low_steps 32 \
  --max_new_tokens 64 \
  --block_size 32 \
  --use_cache prefix \
  --output_prefix artifacts/nfe_stage_ablation/multitool_3call_h128_l32
```

Use the direct `python` command when already inside an allocated GPU node. Use
`srun` only on Slurm clusters where the command is available from a login node.
Do not pass `--threshold` or `--factor` for this ablation; those modes ignore
the fixed transfer quota, so `--steps` is not a clean NFE knob.

Decision criteria:

- Strong: one `low_stage_i` schedule has a materially larger drop in
  `final_all_ready_rate` than the others while `all_high` remains high.
- Weak: all low-stage schedules degrade by similar amounts.
- No-go: `all_high` is already poor, or `low_steps` causes every stage to fail.

## 5. Metrics to Report

- final all-ready rate.
- per-call ready/stable fraction by call index.
- dLLM false-start rate.
- AR optimistic versus AR verified readiness.
- probe latency, output tokens, and full-action accuracy.
- speedup under tool latency and tool capacity assumptions.
- NFE stage sensitivity: `low_stage_i_final_all_ready_delta_vs_all_high`,
  per-stage high/low ready rate, and per-stage high/low generation time.
