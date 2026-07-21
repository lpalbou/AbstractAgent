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

## Status note (2026-07-15 — workflow-layer resolution evidence)
The "one real consumer" arrived and chose a DIFFERENT home: flow's
coding-agent workflow (operator-tasked evaluation, commons c2412) realizes
exactly this verdict class at the WORKFLOW layer — an independent-verifier
SUBFLOW (fresh-context agent executing build/run gates) loops the builder
agent-node with failure-specific reprompting across bounded rounds. My
loop-owner verdict (c2414): the pattern is BLESSED there, and its landing
at the graph layer is evidence AGAINST building the in-loop verdict hook —
composition keeps the loop lean, the graph owns the gate, and the fleet
datum (+103% tokens for in-seat verification beside an external check)
argues the same way. This item stays proposed for the genuinely IN-LOOP
case (a host with no workflow layer wanting a deterministic gate), but the
bar for promotion is now higher: a consumer that cannot use the workflow
shape.

## Status note (2026-07-17 — execution preference shipped as the in-loop half)
The R-Type experiment (commons c2725/c2735/c2736: LLM review blessed three
runtime-dead artifacts; only execution caught them) resolved the in-loop
question WITHOUT the verdict hook: the verifier already forces tool calls, so
the shipped seam is declaration-driven — tools carrying an `executor` tag
(`generation_params.EXECUTOR_TAGS`) are named in the ReAct/CodeAct verifier
prompts with the rule "an artifact that was never executed is not verified;
propose the executor call". Deterministic ground truth enters through the
EXISTING review budget rather than a new gate; the hook-based verdict remains
unbuilt and this item's higher promotion bar stands.

## Validation ideas
Hook verdict test: goal_check fails once with a reason -> loop runs one more
cycle carrying the reason -> passes -> completes; budget exhaustion completes
loudly.

## Non-goals
Replacing review_mode; unbounded verification loops.
