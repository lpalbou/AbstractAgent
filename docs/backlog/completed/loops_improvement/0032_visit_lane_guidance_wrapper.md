# Completed: visit-lane wrapper for drained guidance (c2447 residue closed)

## Metadata
- Created: 2026-07-17
- Status: Completed
- Completed: 2026-07-17
- Work-item id: abstractagent-0032
- Owner: agent
- Thread: 01KXR83E7CJBP7V13N36PT587S (commons c2792, proposal)
- Hub refs: c2796 (semantics adoption + 2 pins), c2798 (runtime voice-owner
  sign-off), c2806 (resolved + ship receipt), c2800/c2809 (metadata-key
  misnomer residual); decision record `decision:visit-guidance-wrapper`
- Provenance: last open sub-item of the c2447 incident slice
  (claim:agent-c2447-loop-tail-leak, closed done at row v8). Filed
  retroactively at the Option A adoption pass (2026-07-18).

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Everything drained from the durable inbox at the reason boundary — gateway
inject_guidance (genuinely the operator), hook steering, verifier
next_prompt lines, and the loops' OWN retry nudges — landed in the durable
transcript under one wrapper: "[Operator guidance — this amends the task;
the final answer must satisfy it]". In a composed entity visit that is
dishonest twice: a host-authored retry nudge wears operator words
(fabricated authority), and task/final-answer vocabulary is the c2447
chrome class the suppress_loop_tail knob exists to keep out.

## What shipped
`generation_params.py`: `GUIDANCE_WRAPPER_TASK` / `GUIDANCE_WRAPPER_VISIT` +
`guidance_wrapper(runtime_ns)`. Under `_runtime.suppress_loop_tail` the
wrapper on all three loops is "[A note arrived during this conversation —
not from your visitor]" — the only attribution honestly claimable while
inbox items carry no source field. Task lane byte-unchanged.

Ruled pins encoded (semantics c2796):
1. VISITOR-COUPLED SCOPE — the spelling is true exactly where the visit
   bridge sets the knob; a visitor-less lane adopting suppress_loop_tail
   needs a SECOND spelling ruled through the same path (comment beside the
   strings).
2. NEVER A PARSE ANCHOR — machine detection of drained guidance keys on
   message metadata kind="operator_guidance" (which deliberately does NOT
   rename), never on bracket prose (a visitor can type the same bytes).

Acknowledged misnomer (semantics c2800, documented at the key's definition
site + the claim row): host nudges ride under kind="operator_guidance" too —
the key means "drained from the inbox", never "the operator said this"; no
audit surface may key on it for operator attribution. The deferred source
split (additive inbox `source` key) is the repair when a live incident
demands it — runtime pre-approved directionally (c2798).

## Validation
- 7 pins in `tests/test_c2447_visit_lane_honesty.py`: task-lane byte-identity
  (3 loops), visit-lane wrapper + guidance words preserved (3 loops),
  metadata-anchor pin. File total 20 green.
- Full suite 281 passed / 2 skipped at ship (284 after the same-day 0031
  wave 2).

## Follow-ups
- WATCHED (recorded in decision:visit-guidance-wrapper): the source-split,
  gated on a live incident; and the frozen-set sync rule if new election
  conventions are added driver-side (A2's clause 2a shares the class).
