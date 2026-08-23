# dLLM Preemption Recovery Decision Space

## CONFIRMED

- Matched recovery points: 96 across 4 request shape(s).
- Full-offload point-median recovery range: 0.67–90.29 ms; point p95 range: 0.69–651.04 ms.
- Semantic-only point-median total recovery range: 6.92–415.06 ms; cache-rebuild component: 0.00–401.44 ms.
- Semantic-only point p95 range: 6.98–486.16 ms.
- Drop/restart point-median recomputation range: 0.50–12222.00 ms; point p95 range: 0.50–12228.18 ms.
- Meaningful measured strategy winners: full_offload_pinned.
- Prefix winners: full_offload_pinned.
- Dual winners: full_offload_pinned.
- Boundary decisions observed: none.

## NEGATIVE RESULTS

- One strategy was the only meaningful winner across all sampled points: full_offload_pinned.
- Non-meaningful raw winners (margin below half an iteration): drop_restart.
- Boundary deferral was not measured in this run.

## REMAINING UNKNOWNS

- KEEP opportunity cost under real multi-request memory contention.
- Generalization beyond the tested model and request shapes.
- Immediate-versus-boundary recovery for the scaled request shapes.

## Explicit Answers

- Q1: Full-state offload point medians span 0.67–90.29 ms over measured cache sizes.
- Q2: Semantic-only total recovery point medians span 6.92–415.06 ms; measured cache-rebuild work spans 0.00–401.44 ms.
- Q3: Lost-progress recomputation point medians span 0.50–12222.00 ms.
- Q4: Raw winners across progress are drop_restart, full_offload_pinned; meaningful winners are full_offload_pinned.
- Q5: Prefix winners are ['full_offload_pinned']; dual winners are ['full_offload_pinned'].
- Q6: Boundary outcomes are not measured in this run.
- Q7: Measurements cover 4 request shape(s): [(25, 128), (25, 256), (512, 128), (2048, 128)].
- Q8: Overall measured decision-space judgment is CONDITIONAL GO.

CONDITIONAL GO
