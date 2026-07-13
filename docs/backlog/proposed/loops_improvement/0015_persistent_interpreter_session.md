# Proposed: persistent interpreter session per run (A2)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: A2

## ADR status
- Governing ADRs: None
- ADR impact: Needs new ADR when promoted (execution trust model — pair with 0016)

## Context
The real capability gap vs OpenHands/smolagents: their Python execution keeps
state across steps (variables, imports, loaded data persist — Jupyter-style
kernel; smolagents remote executors literally run a Jupyter kernel). Our
execute_python is a stateless subprocess per call (sandbox/local.py) — the
model must re-import/reload everything each step; data-heavy work impractical.

## Current code reality
- sandbox/local.py: new subprocess per call, 10s timeout, "development-only".
- Runtime inventory (c1110/c1111): NO kernel/executor lane exists or is in
  flight anywhere; shell lane (0220 sessions) is the closest trust-boundary
  precedent (run-id namespace stamping, teardown).
- SEAM AGREED (c1112): the executor is RUNTIME's build (durable effect:
  idempotency-keyed, replay-safe, approval-gateable); adapters consume.

## Proposed direction
Runtime builds a session-scoped kernel executor (one interpreter per run,
reset at run end) as a durable effect beside the 0220 shell machinery; my lane
co-designs the effect contract and adapts the loops to target it.

## Why it might matter
Highest-value NEW capability in the whole proposal set for coding/data tasks.

## Promotion criteria
Maintainer green-light; then a joint spec with runtime (they offered, c1111).

## Validation ideas
Two-step kernel test: step 1 defines a variable, step 2 reads it; crash-replay
does not double-execute (idempotency key).

## Non-goals
Building the executor adapter-side (would re-implement isolation per loop —
rejected in the seam agreement).
