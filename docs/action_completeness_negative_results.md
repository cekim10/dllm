# Action Completeness Negative Results

This document preserves the dLLM serving experiments that did not survive as
main mechanisms. The goal is to keep the failure modes reusable for a workshop
paper, related-work defense, or a future full-paper motivation section.

## No-Go Mechanisms

| Mechanism | Primary artifact | Result | Interpretation |
| --- | --- | --- | --- |
| Layer/block reuse | `artifacts/block_usefulness/profile_summary.json` | Small compute changes caused large trajectory changes | Quality control becomes an inference-algorithm problem, not a clean serving optimization |
| Elastic canvas growth | `artifacts/elastic_canvas/expansion_deadline_summary.json` | Medium/long requests needed full canvas from step 0 | Response canvas length acts as a conditioning variable, not only a buffer capacity |
| Exact canvas queues | `artifacts/elastic_canvas/serve_sweep_packed/combined_summary.json` | Packed mixed fixed-canvas recovered almost all exact-queue gain | Shape-decoupled packed execution, not exact queue separation, was the real source of gain |
| Refresh-cap admission | `artifacts/refresh_storm/dual_open_r2p0_slo12_same_reject_m512_summary.json` | SLO goodput gains mostly came from early rejection | Refresh-aware admission did not clearly beat generic same-rejection admission |
| General semantic readiness | `artifacts/semantic_readiness/llada_retrieval_s128_summary.json` | Retrieval readiness was weak versus AR prefix | Prefix-like retrieval queries do not strongly benefit from whole-sequence refinement |
| Naive in-flight bind | `artifacts/action_completeness/inflight_binding_core_s128_summary.json` | Bind at 0.25 succeeded 4/10; later binds failed 0/10 | Overwriting result tokens does not repair an answer trajectory formed around placeholders |
| Reserved result slot | `artifacts/action_completeness/reserved_slot_binding_core_s128_summary.json` | Correctness survived, but speedup was about 0.39x | Waiting on result slots stalls because downstream answer text semantically depends on tool results |
| Selective remask | `artifacts/action_completeness/result_dependency_core_s128_summary.json` | Mean answer-changed fraction 79.8%; speedup only above about 1s tool latency | Most answer tokens depend on the result, so partial repair rarely beats restart for short tools |
| Unordered multi-tool formats | `artifacts/action_completeness/multitool_json_array_steer_s128_summary.json`, `artifacts/action_completeness/multitool_named_object_steer_s128_summary.json` | JSON array final all-ready 90%; named object 50%; steering reduced correctness | Format changes did not create reliable simultaneous multi-call emergence |

## Surviving Signal

The strongest remaining result is compact structured action readiness from the
native dLLM trajectory:

- Single-tool core prompts expose stable actions around mid-generation with no
  extra generation branch.
- Multi-tool `action_list` prompts preserve correctness and allow incremental
  per-call dispatch.
- When AR needs a verified auxiliary probe, the dLLM path can avoid that extra
  generation stream.

The claim should not be "dLLMs reveal all actions simultaneously." The stronger
and more defensible claim is:

> Native dLLM refinement exposes stable structured actions incrementally, while
> AR needs either late prefix visibility or an auxiliary probe to obtain
> comparable verified readiness.

## Current Decision Criteria

Continue the tool/action direction only if the scale-up result satisfies both:

- `n >= 100` keeps dLLM-vs-AR-verified speedup above 1.3x for realistic probe
  costs.
- The measured or defensible AR probe cost remains non-trivial under concurrent
  serving, ideally at least 100 ms effective cost or visible GPU interference.

If probe cost is effectively free, single-request latency is not enough. The
remaining path is batch-level tool capacity and concurrent agent serving.
