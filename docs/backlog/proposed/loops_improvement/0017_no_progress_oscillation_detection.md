# Proposed: no-progress + oscillation detection in the loop (B1)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: B1

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Converged best practice: hash (tool, args), stop/steer on repeats; two
identical calls = strong stuck signal, three = decisive; A->B->A->B oscillation
likewise. Composes with the ruled 100 calls/turn ceiling (belt to its braces).

## Current code reality
- react_runtime.py ~1338-1405: duplicate-batch guard WARNS and skips execution,
  but only for SIDE-EFFECT tools (write_file/execute_command/comms) — read-only
  repeat loops spin uncounted.
- max_iterations conclusion path exists and is the right terminal (loud
  synthesis, never silent).

## Proposed direction
Track a (tool,args)-hash repeat streak + simple oscillation window across
cycles; past threshold, force the EXISTING conclusion path with a named reason
(never silent). Emit a hook event so hosts observe it.

## Promotion criteria
Maintainer green-light on the B-series.

## Validation ideas
Scripted loop repeating one read-only call: concludes at threshold with the
reason in the report; oscillation A/B case ditto.

## Non-goals
No new iteration ceilings (the 100 ceiling is upstream); no silent stops.
