# Planned: review/verifier failure degrades to accept-with-#FALLBACK

## Metadata
- Created: 2026-07-12
- Status: Completed
- Completed: 2026-07-13
- Provenance: operator incident (laurent, abstractcode live, commons c1128) +
  own 2026-07-09 review_mode note

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Live incident 2026-07-12: with review_mode enabled, the verifier's structured
LLM call failed pydantic validation (`ReActVerifier_next_tool_callsItem`
rejected the string 'example_item') and the failure surfaced raw in the
operator's CLI while `_temp.final_answer` already held a valid answer.
Root cause of THIS instance is core's schema-example generator
(abstractcore/structured/handler.py:753 exemplifies every array as
`["example_item"]`, type-invalid for object-arrays — their fix). But the
uncontained-verifier class is mine and was already recorded 2026-07-09:
"re-default review to on once review failures degrade to accept-with-#FALLBACK".

## Current code reality
- react_runtime.py maybe_review_node (~1902): review_mode opt-in, default off.
- review_node (~1927): LLM_CALL with response_schema; a validation raise in
  the structured layer fails the EFFECT before review_parse_node ever runs —
  the run dies holding a valid answer.
- review_parse_node (~2006-2035): already tolerant (raw-content JSON fallback,
  non-dict items skipped) — it would have absorbed the incident reply if the
  lower layer had returned instead of raised.
- codeact_runtime.py carries the older verifier fork (~1123+) with the same
  exposure.

## Problem
A verification aid can KILL a run that already succeeded. Verifier failure
must never be worse than no verifier.

## What we want to do
Contain review-effect failure: when the review LLM call fails (validation,
provider error, anything), accept the held final answer and complete, with a
loud #FALLBACK warning in the report/emit ("review skipped: <reason>").
Coordinate the mechanism with runtime/core (thread c1128-c1129): either the
structured path degrades to raw-content + labeled validation_error (my
preference — my parser is already tolerant), or a `strict: false` knob on
response_schema payloads, or effect-failure routing. Whichever lands below, my
adapter must not die on review failure. Then re-evaluate the 2026-07-09
direction: re-default review_mode to on.

## Scope
ReAct review path (and CodeAct's fork if it survives the C5 ruling / 0021
refactor).

## Non-goals
Fixing core's example generator (their lane, reported c1129); abstractcode's
raw-error surfacing (their lane).

## Validation
Kernel test: scripted review effect FAILURE (handler returns failed outcome)
with a held final answer -> run COMPLETES with the answer + #FALLBACK
review-skipped marker in emits/report; review success path unchanged.

## Thread state (c1128)
- c1129 (mine): root cause pinned — core's example generator (handler.py:753)
  taught the type-invalid placeholder; my parser is already tolerant.
- c1130 (runtime): confirms the raise is NOT in llm_client (they forward
  response_model to core; d9ed2e7 forces non-stream on this shape, so the
  streaming class is excluded) and ENDORSES the clean split: agent degrades
  the verifier, core hardens the coercion, code contains the CLI surface.
  Note: runtime's RetryPolicy (llm_max_attempts=3) re-issues a
  deterministically-failing structured call — the adapter degrade also stops
  that waste.
- c1139 (code): ask 1 closed — abstractcode's own `set_defaults(review=True)`
  enabled the verifier on every react run; now DEFAULT OFF with an in-code
  comment naming the re-default condition: THIS item's degrade belt + core's
  example-generator fix. When both ship, code flips review back on (one line).
  Their CLI also contains failure rendering structurally now (human summary +
  raw error folded). MY SHIP IS THE BLOCKING HALF of review returning to
  default-on downstream — raises this item's priority.

## Progress checklist
- [x] Watch c1128 thread for where the degrade lands (core/runtime/adapter)
- [x] Adapter containment + #FALLBACK emit
- [x] Tests + changelog
- [x] Re-evaluate review_mode default (2026-07-09 note)

## Completion report
- Date: 2026-07-13
- Outcome: containment shipped via the runtime's EXISTING opt-in failure
  absorption (`payload._absorb_failure`, the fdf01e0 rule class — no new
  runtime mechanism needed; the c1129 "effect-failure routing if runtime
  prefers that shape" option turned out already built and precedented by the
  visit workflow's staged appliers). Both verifier forks opt in at the review
  LLM_CALL payload; both `review_parse` nodes contain the absorbed record:
  accept the held answer, complete, loudly — `scratchpad.review_skipped`
  (#FALLBACK + reason), a `review: #FALLBACK skipped (...)` report line, and a
  dedicated `review_skipped` emit (`accepted_held_answer: true`). CodeAct's
  fork additionally stops the absorbed record from falling through into the
  unactionable-retry path (re-issuing a deterministically failing call, then
  re-entering `reason` — worse than no verifier on both counts). Runtimes
  without absorption (e.g. released 0.4.29) ignore the payload key: behavior
  there unchanged, verified by grep against the installed release.
- review_mode default re-evaluation: the ADAPTER default stays opt-in
  (absent -> False). The 2026-07-09 re-default note is about the consuming
  CLI's default — abstractcode's c1139 in-code comment names exactly this
  belt + core's example-generator fix (shipped c1201) as its flip condition,
  and that flip is code's one-line lane. Flipping the adapter default would
  add a verifier LLM call to every react run for ALL consumers (gateway,
  flows, entities) — a cost/behavior decision that is not this item's to take.
- Validation: `tests/test_review_failure_containment.py` — kernel run with a
  scripted terminal verifier failure and a held answer completes WITH the
  answer + all three loud surfaces (payload flag pinned on the wire); success
  path guarded byte-unchanged; CodeAct node-level containment incl. the
  retry-counter-untouched pin. Full suite 144 passed / 2 skipped (sibling
  trees).
- Follow-ups: none new; abstractcode owns the review-on flip (c1139), and
  proposed 0021 (shared adapter core) will fold the two containment blocks
  into one when it lands.
