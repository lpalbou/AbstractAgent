# AbstractAgent backlog overview

AbstractAgent is the agent behavior layer of AbstractFramework: the default
agentic loops (ReAct, CodeAct, MemAct) as durable workflow adapters over
AbstractRuntime, plus their shared logic (tools contract, generation params,
loop hooks, allowlist handling). This backlog is the package's durable
planning memory.

## Counts (2026-07-13)

| State | Count |
|---|---|
| Planned | 4 |
| Proposed | 14 |
| Completed | 1 |
| Deprecated | 0 |
| Recurrent | 2 |

## Next recommended work

1. `planned/loops_improvement/0010` + `0012` + `0013` — trivial defect fixes,
   same-day batch (fenced-block intent bug, delegate-budget honesty, MemAct
   dead knobs).
2. `planned/loops_improvement/0011` — sibling transcript repair; check
   proposed `0021` (shared-core refactor) first: if 0021 is green-lit, 0011
   folds into it.
3. Everything in `proposed/loops_improvement/` waits on its stated promotion
   criteria (mostly maintainer green-light per proposal ID; `0022` C5 is an
   explicit maintainer ruling; `0015` A2 is gated on runtime's executor lane).

## Planned items

| ID | Item | Area |
|---|---|---|
| 0010 | `planned/loops_improvement/0010_codeact_fenced_block_intent_bug.md` (A4) | codeact adapter |
| 0011 | `planned/loops_improvement/0011_sibling_transcript_repair_strict_providers.md` (C2) | codeact+memact adapters |
| 0012 | `planned/loops_improvement/0012_sibling_delegate_budget_honesty.md` (C3) | codeact+memact adapters |
| 0013 | `planned/loops_improvement/0013_memact_dead_knobs.md` (C4) | memact agent surface |

## Proposed items

| ID | Item | Promotion gate |
|---|---|---|
| 0001 | `proposed/0001_agent_gateway_install_boundary.md` | install-boundary evidence (legacy item, renamed from date-form 2026-07-12) |
| 0014 | `proposed/loops_improvement/0014_native_code_action_primary_path.md` (A1) | maintainer green-light; pairs with 0010 |
| 0015 | `proposed/loops_improvement/0015_persistent_interpreter_session.md` (A2) | green-light + runtime executor lane (seam agreed c1110–c1112) |
| 0016 | `proposed/loops_improvement/0016_execution_trust_tiers.md` (A3) | maintainer ruling on tier model; needs decision record |
| 0017 | `proposed/loops_improvement/0017_no_progress_oscillation_detection.md` (B1) | green-light |
| 0018 | `proposed/loops_improvement/0018_structured_handoff_long_runs.md` (B2) | green-light + a real long-run consumer |
| 0019 | `proposed/loops_improvement/0019_deterministic_turn_end_verification.md` (B3) | green-light + one real consumer |
| 0020 | `proposed/loops_improvement/0020_provider_stop_semantics_over_heuristics.md` (B4) | green-light |
| 0021 | `proposed/loops_improvement/0021_shared_adapter_core_refactor.md` (C1) | green-light; ideally after 0022 ruling |
| 0022 | `proposed/loops_improvement/0022_codeact_fold_decision.md` (C5) | MAINTAINER RULING (user-facing selector) |
| 0023 | `proposed/loops_improvement/0023_memact_status_honesty.md` (D1) | maintainer preference |
| 0024 | `proposed/loops_improvement/0024_dead_public_surface_pruning.md` (D2) | green-light (semver-relevant) |
| 0025 | `proposed/loops_improvement/0025_cross_seat_integration_flags.md` (D3+D4) | other seats' lanes; close when superseded |
| 0026 | `proposed/loops_improvement/0026_loop_hooks_follow_ups.md` | next hooks build wave |

## Topic tracks

- **loops_improvement** (mixed): the 2026-07-12 loops meta-audit proposal set
  (maintainer-endorsed IDs A1–A4, B1–B4, C1–C5, D1–D4 + hooks follow-ups).
  Defects in `planned/loops_improvement/` (README there), features/decisions
  in `proposed/loops_improvement/` (README there, includes the suggested
  execution order as proposed to the maintainer).

## Completed ledger

| Date | ID | Item | Outcome |
|---|---|---|---|
| 2026-07-13 | 0027 | `completed/loops_improvement/0027_review_failure_degrades_to_accept.md` | Verifier failure contained in both forks via runtime `_absorb_failure`: held answer accepted, run completes, loud #FALLBACK (scratchpad marker + report line + `review_skipped` emit). Agent half of the c1128 review re-default condition; adapter default stays opt-in (flip is abstractcode's, c1139). Suite 144/2. |

## Deprecated

(none yet)

## Process

- New item: scan all lifecycle dirs for the next unused global `NNNN`, name
  `NNNN_slug.md` (no dates in filenames), use the planned/proposed templates
  (codex backlog skill), update this overview in the same pass.
- Complete: append `## Completion report` (date, outcome, validation
  evidence), move to `completed/`, add a ledger row here, run
  `recurrent/post_completion_follow_up_triage.md`.
- Deprecate: append `## Deprecation report` with the reason, move to
  `deprecated/`, add a row here.
- Hygiene: `recurrent/backlog_and_adr_hygiene.md` after any add/move/close.

## Planning notes

- 2026-07-12: backlog system normalized (overview, lifecycle dirs, recurrent
  tasks); legacy date-named item renamed to `0001_...`; loops_improvement
  track created from the maintainer-endorsed meta-audit proposal set. IDs
  0002–0009 left as gap for urgent insertions. The A/B/C/D proposal IDs are
  the maintainer's reference vocabulary — keep them in item headers and
  reports.
- Commit policy: nothing here implies a commit; workspace rule stands (no
  commits without the maintainer's word).
