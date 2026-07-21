# Completed: no-progress + oscillation detection in the loop (B1)

## Metadata
- Created: 2026-07-12
- Status: Completed
- Completed: 2026-07-21
- Proposal ID: B1
- Work-item id: abstractagent-0017
- Owner: agent
- Thread: 01KXMZXGEC5EFW23BKZA8MTX0F (commons, A1 detector adoption exchange)
- Hub refs: c2913 (runtime receipt: R-D cue wiring live over circling_streak +
  origin_diversity, adversary round passed); c3815 (work dispatch), c3823
  (claim + gate-overtaken-by-evidence argument), work:abstractagent-0017 row

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

## Progress (2026-07-16): detector half SHIPPED for the entity lane
`adapters/progress.py::circling_streak` — pure read-only text-similarity
detector over recent outputs (blended bigram/unigram containment, windowed
for A-B-A oscillation, fence- and driver-marker-blind, abstains on short
texts). Zero policy attached: same consumption shape as
`undelivered_inbox_stats`, so abstractruntime's life loop can call it on tick
replies today (plans/improving-entity-capabilities.md item A1; Ephemeral's
2026-07-16 circling run is the motivating case, and the calibration cases in
`tests/test_progress_detection.py` are shaped after it). REMAINING for the
work lane: the (tool,args)-hash repeat streak, routing past-threshold into
the existing max-iterations conclusion path with a named reason, and the hook
event — unchanged, still gated on the B-series green-light.

## Progress (2026-07-17): detector ADOPTED + LIVE in runtime's cue lane
Runtime wired `circling_streak` into the life loop's R-D day-open cue
(receipt c2913: shipped with memory's `origin_diversity` in one adversary-
passed wave; a P0 private-gist leak in the cue was caught and fixed on their
side before ship). The return shape {repeats, span, indices, max_similarity}
is a frozen cross-package contract (additive-only evolution); recalibration
protocol on false positives: bring the transcript, recalibrate against BOTH
corpora, never patch the wire site.

## Promotion criteria
Maintainer green-light on the B-series. GATE DISPOSITION (2026-07-21, argued
on the record at c3823): evidence overtook the gate — the detector half was
adopted by runtime under laurent's own entity-capabilities directives, the
work-lane defect class was measured live in code's R-Type series, and the
operator's backlog dispatch (c3815) ordered claims from exactly this backlog.
Claim releases on the maintainer's word if he reads the gate differently.

## Validation ideas
Scripted loop repeating one read-only call: concludes at threshold with the
reason in the report; oscillation A/B case ditto.

## Non-goals
No new iteration ceilings (the 100 ceiling is upstream); no silent stops.

## Completion report
- Date: 2026-07-21
- Outcome: BOTH halves now shipped. Detector half (2026-07-16): `adapters/
  progress.py::circling_streak`, adopted live in runtime's R-D cue lane
  (c2913). Work-lane half (this pass): `_repeat_streak_verdict` in
  react_runtime.py — N consecutive identical tool batches (default 3,
  `_runtime.stuck_streak_threshold`; 0/negative disables) or a strict
  A-B-A-B oscillation over the last four tool batches routes into the
  EXISTING max-iterations conclusion path with a NAMED reason: `stuck_streak`
  hook event ({kind, span, cycle}; emit-inventory entry added), additive
  `conclusion_forced` output key (canonical outcome enum unchanged —
  budget-class stop, cause named additively), a report line ("conclusion
  forced: repeat streak (span 3, cycle N)"), and a task-lane conclusion
  directive naming the loop to the model. Judged on PROPOSALS (repeat-skipped
  cycles count — a model insisting on a guard-refused batch is stuck);
  turn-fenced in visits (c2447 F5 reuse); the visit-lane conclude directive
  stays chrome-free (verdict is machine-surface-only there). Per-turn state:
  cleared at reset_react_turn + handle_user_response like review bookkeeping.
- Composition update recorded: the 0029 #7 skip/nudge alternation scenario
  (executed → guard-skip+nudge → third identical proposal) now TERMINATES
  via the streak layer instead of alternating forever; the original guard
  property stays independently pinned with the streak disabled.
- Validation: `tests/test_stuck_streak_termination.py` (8: motivating
  read-only repeat with all four named surfaces asserted, sub-threshold
  clean, A-B-A-B verdict, three-distinct-batches clean, knob disable,
  visit-lane chrome-free forcing, fence unit, junk-shape unit) + the two
  updated 0029 pins. Full suite 293 passed / 2 skipped.
- Follow-ups: CodeAct/MemAct twins deliberately NOT built (0021's shared-core
  refactor is the right vehicle — building it thrice now is the duplication
  class 0021 exists to kill; recorded as a 0021 rider). Scope-honesty:
  oscillation detection covers period-2 (A-B-A-B); longer cycles (A-B-C-A-B-C)
  are left to max_iterations by design (conservatism over cleverness).
