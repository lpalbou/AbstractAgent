# Completed: hook taxonomy multi-emit + report accumulation (audit residue)

## Metadata
- Created: 2026-07-13
- Status: Completed
- Completed: 2026-07-15
- Proposal ID: audit-2026-07-13 P2-4 / P2-7

## Context
The 2026-07-13 production-readiness fable5 audit (focused seams pass) left two
P2 findings deliberately unfixed — both are contract-design questions, not
surgical defects, and fixing them casually could break consumers of the hook
taxonomy or the report format.

## Findings (verbatim from the audit)
1. **`turn_end` multi-emit (P2-4)**: ReAct's `max_iterations` node emits its
   step on EVERY node entry, and the node re-enters itself (effect-issuing
   entry, parsing entry, markup-retry entry). `loop_hooks` maps each emit to
   `turn_end` with `outcome: iteration_budget`, so a capture host sees 2-3
   `turn_end` events for ONE turn — the first while the conclusion LLM call is
   still pending. CodeAct/MemAct emit once. Fix shape: emit only on the
   completing entry, or a distinct step name for intermediate entries.
   Consumers of the hook taxonomy must be surveyed first (abstractcode's
   capture lane is the known consumer).
2. **`review_skipped` report accumulation (P2-7)**: `reset_react_turn`
   deliberately preserves the scratchpad, so turn N's report re-lists turns
   1..N-1's review skips. Arguably deliberate durability; if it confuses,
   entries should carry the turn/iteration they belong to rather than being
   pruned (never silently drop #FALLBACK history).

## Promotion criteria
Either a hook-consumer report of double `turn_end` handling, or the next
loop_hooks contract revision (whichever comes first). Coordinate with
abstractcode before changing emit multiplicity.

## Execution note (2026-07-14, operator green light — the contract wave)
Executed in batch 4 (CHANGELOG): ONE turn_end per budget-exhausted turn
(`max_iterations_reached` announce step + terminal `max_iterations` at the
completion branch); `review_skipped` per-turn reset at the ask_user boundary
(reports/output reflect the current turn; ledger keeps history); parse
payload COMMON CORE (`has_tool_calls` + `tool_calls` + `content_preview`)
across all three loops with loop extras additive. Remaining 0028 scope: none
— the naming split was resolved at the inventory's founding publication
(RENAMED_STEPS) and the payload/emission contracts above close the rest.
Item promotable to completed at the next backlog pass.

## Completion report
- Date: 2026-07-15 (backlog pass alongside 0026's close)
- Outcome: all substance shipped in the 2026-07-14 batch-4 contract wave (see
  the execution note above — one turn_end per turn, per-turn report resets,
  parse common core, RENAMED_STEPS record). This close is the promotion the
  note announced; no code changed at close time.
- Validation: pins live in `tests/test_gated_remainder_wave_2026_07_14.py`
  (parse common core, turn-boundary resets) and the emit-inventory drift
  test; 0026's same-day wave re-ran the full suite over them (233/2 green).
- Follow-ups: none.
