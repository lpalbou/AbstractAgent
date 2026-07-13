# Proposed: MemAct public-surface honesty — experimental label or root export (D1)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: D1

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
MemAct's half-state: not exported at package root (unlike CodeAct), docs say
"import from abstractagent.agents.memact", one opt-in consumer (abstractcode
selector), dead knobs (planned 0013), same strict-provider bug (planned 0011).
The keep-recommendation: genuinely distinct architecture (Letta-style per-turn
memory rewriting; machinery lives runtime-side and is separable), worth
keeping EXPLICITLY EXPERIMENTAL.

## Proposed direction
Either root-export it AND label experimental in docs/agents.md + README, or
keep the deep import and say "experimental" plainly. Fix the 0013/0011 debt
first so the label is honest about status, not an excuse.

## Promotion criteria
Maintainer preference; cheap either way.

## Non-goals
Parity investment; deprecation (it has a real consumer and a real distinct idea).
