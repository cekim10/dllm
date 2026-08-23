# Preemption State Structural Kill Test

## CONFIRMED

- Production-sampler full-state exact resume: 72/72.
- Semantic-only exact resume: 72/72.
- Prefix/dual cache-rebuild exact resume: 48/48.
- Every resumed canvas, transfer position, and final output was bit-exact.
- No additional hidden sampler state was required.

### Minimal Semantic State

- Quota inner-step: 1,224 B.
- Quota block boundary: 1,224 B.
- Threshold inner-step: 1,232 B.
- Threshold block boundary: 1,224 B.
- Attention mask, transfer schedule, block index, and replace position were reconstructable.
- Threshold inner-step checkpoints require the 8-byte inner-step index.

### Derived Performance State

- Prefix cache by block: 13.1 MB -> 29.9 MB -> 46.7 MB -> 63.4 MB.
- Dual cache during inner steps: 80.2 MB.
- Block-boundary cache: 0 B before warmup.
- Maximum prefix full/semantic separation: 51,831x.
- Maximum dual full/semantic separation: 65,538x.

## NEGATIVE RESULTS

- Cache state is not required for correctness; preserving it is purely a performance choice.
- Dual-cache size does not vary materially within a block.
- No recovery-strategy crossover has yet been measured.

## REMAINING UNKNOWNS

- Full-state D2H/H2D transfer cost distributions.
- Exact cache-rebuild cost versus restart cost across progress.
- Immediate preemption versus waiting for a block boundary.
- Persistence across prompt and generation lengths.

CONDITIONAL GO
