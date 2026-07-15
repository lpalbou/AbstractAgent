# Planned: MemAct dead knobs — review_mode/plan_mode accepted, ignored (C4)

## Metadata
- Created: 2026-07-12
- Status: Completed
- Completed: 2026-07-13
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
- [x] Remove or raise
- [x] docs/agents.md alignment
- [x] Test + changelog

## Completion report
- Date: 2026-07-13
- Outcome: REMOVED (the preferred shape) — `plan_mode`/`review_mode`/
  `review_max_rounds` deleted from `MemActAgent.__init__` and `start()`, the
  `_runtime` entries no longer written. Passing any of them raises TypeError
  (the raise-or-delete requirement satisfied by the signature itself).
  Verified pre-removal: zero reads in memact_runtime.py AND logic/memact.py
  (grep clean — all three genuinely dead, not just the review pair).
- docs/agents.md: the stale line 41 ("ReAct ... does not apply them")
  replaced with the accurate split — review_mode HONORED by the ReAct
  verifier (0217) with 0027 failure containment; plan_mode stored-but-unread
  by the ReAct adapter (plan rides the always-on update_plan tool +
  check_plan gate); MemAct section documents the removal. Note for backlog
  0024 (dead-surface pruning): ReactAgent's own `plan_mode` param is the same
  accepted-but-unread class — named there, out of this item's scope.
- Validation: TypeError pins + a source-level no-live-reference check in
  tests/test_sibling_loop_defects_batch.py; full suite 156 passed / 2
  skipped.
