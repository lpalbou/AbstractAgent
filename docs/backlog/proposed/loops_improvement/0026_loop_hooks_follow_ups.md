# Proposed: loop-hooks follow-ups from the adversarial review (hooks wave)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: (hooks-wave follow-ups; pre-dates the A-D set)

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The 2026-07-12 loop-hooks ship (adapters/loop_hooks.py + wiring in all three
adapters) closed the adversaries' P1s (cross-run steer leakage, follow-up
invisibility, slow-handler DoS). These are the named P2/P3 leftovers.

## The follow-ups
- Emit `parse_tool_calls` on CodeAct/MemAct so `tool_proposed` fires on all
  three loops (today ReAct-only).
- A real mutation pin against the `act` payload (handlers get copies; one test
  asserts payload copy depth on the hot path).
- Discard queued steering on run FAILURE/cancel, not only at clean terminals
  (today only done/max_iterations discard).
- A ReAct run-init emit (siblings emit an init step; ReAct starts at reason).
- The silent conclude-phase drain: injections arriving during the forced
  conclusion cycle are folded but can no longer influence — either surface
  or discard loudly.

## Promotion criteria
Next hooks build wave, or the first external hook consumer hitting one.

## Non-goals
New hook vocabulary; async handler execution (rejected in design — sync +
budget is the contract).
