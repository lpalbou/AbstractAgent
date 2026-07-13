# Planned: MemAct dead knobs — review_mode/plan_mode accepted, ignored (C4)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Proposal ID: C4

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The silent fall-open class this workspace keeps closing: a knob that accepts a
value and does nothing is worse than no knob.

## Current code reality
- agents/memact.py ~71-73, ~177-179: `plan_mode`, `review_mode`,
  `review_max_rounds` accepted and persisted into run vars.
- memact_runtime.py: ZERO reads of any of them — no plan node, no review nodes.
- docs/agents.md:41 additionally claims ReAct ignores review_mode (stale — it
  has honored it since backlog 0217; fixed in docs during the 2026-07-12
  coredoc pass — verify).

## Problem
`MemActAgent(review_mode=True)` silently does nothing.

## What we want to do
RAISE-OR-DELETE, loudly: preferred = remove the parameters from MemActAgent
(they never worked; no consumer passes them — integration audit found zero
callers); alternative = raise ValueError naming the unsupported knob.

## Scope
agents/memact.py signature + docs.

## Non-goals
NOT building a MemAct verifier (that would be parity investment against the
keep-experimental recommendation).

## Validation
Constructor test: passing review_mode/plan_mode raises (or the params are gone
and mypy/tests confirm no references).

## Progress checklist
- [ ] Remove or raise
- [ ] docs/agents.md alignment
- [ ] Test + changelog
