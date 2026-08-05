# Structured-Action Safety Protocol

The runtime must not rely on empirical false-start rates for correctness.
Speculative execution is only safe if external actions are classified and
validated before effects become visible.

## Action Classes

| Class | Examples | Speculative behavior |
| --- | --- | --- |
| Read-only | weather lookup, flight search, CRM/calendar read, retrieval | Dispatch as soon as the action is stable and required fields are complete |
| Idempotent write | cache warmup, draft document creation with stable idempotency key | Prepare speculatively; commit only after final output validates |
| Non-idempotent write | payment, email send, ticket purchase, deletion | Never execute speculatively; optionally pre-validate parameters only |

## Lifecycle

1. `FORMING`: some tool or argument fields are visible in the dLLM refinement
   history.
2. `DISPATCHABLE`: required fields are complete and stable for the configured
   number of refinement snapshots.
3. `IN_FLIGHT`: a read-only or prepare-only external request has been issued.
4. `SUPERSEDED`: a later dLLM snapshot changes a field before final validation.
5. `VERIFIED`: final output exactly matches the speculative action.
6. `COMMITTED`: the result is exposed to downstream answer generation or to the
   user.
7. `REJECTED`: speculative result is discarded and normal execution is used.

## Validation Rule

Speculative results are never committed directly. The final decoded structured
action is the authority:

- exact match: reuse speculative result.
- mismatch: discard speculative result and run the final action normally.
- partial match: reuse only cacheable read-only subresults, never side effects.

## Paper Claim

The safety claim should be structural, not empirical:

> Native dLLM readiness determines when speculation may start; final structured
> action validation determines whether speculative results may be committed.

The measured false-start rate supports efficiency, but correctness must not
depend on it being zero.
