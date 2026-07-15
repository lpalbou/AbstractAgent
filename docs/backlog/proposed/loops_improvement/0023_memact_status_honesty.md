# Proposed: MemAct public-surface honesty — experimental label or root export (D1)

## Metadata
- Created: 2026-07-12
- Status: Mostly executed (docs half)
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

## Execution notes (2026-07-14)
The "say experimental plainly" half EXECUTED: docs/agents.md (section header +
status paragraph + table row), README.md feature line, docs/api.md note — all
label MemAct experimental with the rationale (distinct architecture, one
consumer, less mileage) and frame the deep import as the label made structural.
The 0013/0011 debt this item gated on is PAID (0013 dead knobs removed then
shimmed deprecated-ignored; 0011 strict-provider transcripts fixed batch 8).
Also corrected agents.md's stale "raises TypeError" claim to the shipped
deprecated-ignored shim behavior. REMAINING (maintainer preference): the
root-export decision — export + label vs keep the deep import permanently.
