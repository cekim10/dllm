# Preemption State Structural Kill Test

## CONFIRMED

- Real-sampler full-state exact resume passed: True.
- Exact checkpoint rows: 5.
- Minimal-state conclusions are based on component-removal ablations.

## NEGATIVE RESULT

- No recovery strategy or scheduling crossover has been established.
- Python CPU export timing is not treated as GPU offload cost.

## UNKNOWN / NOT YET TESTED

- GPU-to-CPU and CPU-to-GPU transfer distributions.
- Cache rebuild versus restart costs across progress.
- Prompt/generation/block-size scaling beyond this run.
- Strategy winner and block-boundary waiting trade-off.

CONDITIONAL GO
