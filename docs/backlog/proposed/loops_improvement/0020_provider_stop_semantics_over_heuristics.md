# Proposed: trust provider stop semantics over content heuristics (B4)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: B4

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The deferred-action nudge (_looks_like_deferred_action) is a content heuristic
built for prompted models ("I will now..." with no tool call -> re-ask). On
native tool-calling providers an explicit finish_reason=stop with zero tool
calls is a trustworthy termination signal; the heuristic then produces false
re-nudges (code seat's 9-cycle streaming divergence brushed this class).

## Current code reality
- react_runtime.py ~1443-1455: followthrough retry, fence-blind, gated by
  _runtime.check_plan (default ON).
- finish_reason is already parsed (~1300-1303) and used for the truncation
  retry.

## Proposed direction
Gate the followthrough heuristic OFF when the provider returned an explicit
finish_reason == "stop" AND the run's provider declares native tool support
(tools were offered natively this cycle). Keep it for prompted/fenced lanes.

## Promotion criteria
Maintainer green-light on the B-series.

## Validation ideas
Two scripted cases: native stop + deferred-sounding prose -> NO nudge, final
answer accepted; prompted lane -> nudge unchanged.

## Non-goals
Removing the heuristic (prompted local models still need it).
