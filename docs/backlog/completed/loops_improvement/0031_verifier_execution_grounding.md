# Completed: verifier execution grounding (R-Type wave — executor tag + budget bound)

## Metadata
- Created: 2026-07-17
- Status: Completed
- Completed: 2026-07-17
- Work-item id: abstractagent-0031
- Owner: agent
- Thread: 01KXPB6E5TYRJESMRZ8Y0H0YY0 (commons c2725, R-Type experiment)
- Hub refs: c2780, c2783, c2784 (ship + boundary + takes); c2856 (A/B results);
  c2872 (budget-bound fix); c2881 (code's adoption); c2945 (series close)
- Provenance: code's operator-directed R-Type experiment (n=2/arm, ornith-35b):
  LLM-read verification blessed three runtime-dead artifacts (crash on first
  input, every-frame ReferenceError, corner-ninth draw) that only EXECUTION
  caught. Filed retroactively at the Option A adoption pass (2026-07-18) —
  the work pre-dated the ruling; receipts were hub-first.

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The verifier lane (ReAct + CodeAct review nodes) judged completion by READING
tool outputs. The R-Type series showed the signal ceiling: an unexecuted
artifact can pass any read. The composition principle that won the series:
execute where an executor exists; LLM-read only where none does — and the
signal source (world-side ground truth), not loop sophistication, is what
catches the failure classes.

## What shipped (two waves, one arc)

### Wave 1 — executor-tag seam (2026-07-17 afternoon)
`generation_params.py`: `EXECUTOR_TAGS` (frozenset {"executor"}),
`executor_tool_names()` (allowlist-ordered, tag-driven, junk-safe),
`verifier_execution_preference()` (prompt block; empty without executors —
verifier prompts stay BYTE-IDENTICAL for deployments without executor tools).
Both review nodes name executor-tagged tools with the rule "an artifact that
was never executed is not verified — return next_tool_calls invoking the
matching executor"; the existing forced-tool-call seam (next_tool_calls →
act, transcript-threaded) executes them. The probe binary lives TOOL-SIDE
(abstractcode's browser_probe declares tags=["executor"] citing this
contract, their c2769) — abstractagent never carries an executor dependency.

### Wave 2 — review-budget re-arm fix (2026-07-17 evening, from A/B data)
The per-answer review-budget reset (`review_count = 0` after tool execution)
fired for VERIFIER-FORCED batches too — the verifier re-armed its own budget
through the calls it forced itself, unbounding consecutive review rounds on
one unchanged answer (live cost: A/B run2 burned a 40-minute wall cap
re-reviewing a good artifact). Fix: `_temp.review_forced_batch` marks batches
synthesized by review_parse; observe skips the reset for them (forced rounds
CONSUME review_max_rounds, which now genuinely bounds them); the marker dies
with its batch and at user-response/turn boundaries. Deliberate asymmetry
pinned: model-issued activity is a new claim and still re-arms. Chosen over
code's suggested "N green probes" counter because run2's pathology was
probe-once-then-READ — the counter would never have fired; the budget bounds
the class.

## Validation
- `tests/test_verifier_execution_preference.py` (12): tag classification,
  prompt byte-identity without the tag, both adapters' prompts, end-to-end
  forced-probe execution, never-complete verifier bounded at exactly
  max_rounds, model re-arm preserved, CodeAct observe symmetry.
- Full suite 284 passed / 2 skipped (2026-07-17).
- External: code's A/B (c2856) — 4/4 probe-green with the executor present;
  seam fires and catches the crash class; blowup class re-tested fixed-at-
  budget (their adoption c2881: report language "no correctness regression,
  convergence speed at best, worst case bounded at review_max_rounds").

## Follow-ups
- Backlog 0019 status note updated same day (execution preference is the
  shipped in-loop half; the hook-based verdict stays unbuilt, higher bar).
- Standing: review_mode default stays ON (0027 rationale); harnesses with
  external verification keep --no-review as their knob.
- The A2 format-repair nudge (visit-1 finding) was SPEC-ONLY from this seat
  (driver-lane; runtime built it to spec same hour, their c3006, adopted with
  clause 2a folded at c3007) — no abstractagent artifact, deliberately no
  item file.
