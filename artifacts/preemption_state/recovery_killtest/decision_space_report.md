# dLLM Preemption Recovery Decision Space

## CONFIRMED

- Matched recovery points: 24 across 1 request shape(s).
- Full-offload point-median recovery range: 0.68–8.29 ms; point p95 range: 0.70–47.83 ms.
- Semantic-only point-median total recovery range: 7.06–53.13 ms; cache-rebuild component: 0.00–39.10 ms.
- Semantic-only point p95 range: 7.12–56.18 ms.
- Drop/restart point-median recomputation range: 0.51–5931.01 ms; point p95 range: 0.51–5939.34 ms.
- Meaningful measured strategy winners: full_offload_pinned.
- Prefix winners: full_offload_pinned.
- Dual winners: full_offload_pinned.
- Boundary decisions observed: immediate.

## NEGATIVE RESULTS

- No single strategy won every sampled point.
- Non-meaningful raw winners (margin below half an iteration): drop_restart.
- Waiting for a block boundary never beat immediate recovery.

## REMAINING UNKNOWNS

- KEEP opportunity cost under real multi-request memory contention.
- Generalization beyond the tested model and request shapes.

## Explicit Answers

- Q1: Full-state offload point medians span 0.68–8.29 ms over measured cache sizes.
- Q2: Semantic-only total recovery point medians span 7.06–53.13 ms; measured cache-rebuild work spans 0.00–39.10 ms.
- Q3: Lost-progress recomputation point medians span 0.51–5931.01 ms.
- Q4: Raw winners across progress are drop_restart, full_offload_pinned; meaningful winners are full_offload_pinned.
- Q5: Prefix winners are ['full_offload_pinned']; dual winners are ['full_offload_pinned'].
- Q6: Boundary outcomes are ['immediate'].
- Q7: Measurements cover 1 request shape(s): [(25, 128)].
- Q8: Overall measured decision-space judgment is CONDITIONAL GO.

CONDITIONAL GO
