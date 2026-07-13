# Proposed: structured handoff artifact for long runs — not compaction (B2)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: B2

## ADR status
- Governing ADRs: ADR-0026-class workspace rule (no silent truncation) — respected by design
- ADR impact: May need ADR when promoted (context-management policy)

## Context
Industry finding: Sourcegraph Amp RETIRED compaction in favor of explicit
handoff artifacts; consensus is context as STRUCTURED STATE, not compressed
text. Our loop deliberately sends full context forever (init disables all
trimming — the ADR-0026 lesson); correct, but a 60-cycle run eventually hurts.

## Current code reality
- react_runtime.py init: max_history_messages=-1 etc. (full context policy).
- scratchpad already holds cycles/plan/facts durably — most of a handoff
  artifact exists.

## Proposed direction
An OPT-IN, EXPLICIT, LABELED handoff verb: the loop writes a structured
progress state (task, facts, decisions, next steps) as an artifact; a FRESH
run seeds from it. Host-invoked, never automatic; no trimming of the live run
ever. Pairs with delegate_agent and the fleet's partitioned work.

## Promotion criteria
Maintainer green-light; a real long-run consumer (fleet or abstractcode)
asking for it.

## Validation ideas
Handoff round-trip: run A writes artifact; run B seeded from it answers a
question that requires run A's facts without run A's transcript.

## Non-goals
Automatic compaction; mid-run context surgery; anything silent.
