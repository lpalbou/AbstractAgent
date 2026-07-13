# Proposed: deterministic verification at turn end (B3)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: B3

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
review_mode is model-critiques-model. Converged practice adds DETERMINISTIC
gates (tests pass, schema validates, artifact exists) — do not trust the
model's claim of completion. The hooks layer already carries the mechanism:
a turn_end hook can steer.

## Current code reality
- LoopHooks steer: a handler returning an injection folds into the durable
  inbox — but at TURN END the run completes before the steer lands (terminal
  discard, hook_steer_discarded), so verification-shaped steering cannot
  extend a run today.
- review_mode (react) exists, opt-in, LLM-based.

## Proposed direction
First-class turn-end verdict: a designated hook (or _runtime.goal_check
contract) evaluated BEFORE terminal completion; verdict "not done + reason"
converts to durable steering and the loop continues (bounded — reuse
review_max_rounds-style budget). Makes the existing mechanism a contract
instead of a race.

## Promotion criteria
Maintainer green-light on the B-series; one real consumer (fleet harness or
abstractcode gate).

## Validation ideas
Hook verdict test: goal_check fails once with a reason -> loop runs one more
cycle carrying the reason -> passes -> completes; budget exhaustion completes
loudly.

## Non-goals
Replacing review_mode; unbounded verification loops.
