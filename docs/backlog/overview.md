# AbstractAgent backlog overview

AbstractAgent is the agent behavior layer of AbstractFramework: the default
agentic loops (ReAct, CodeAct, MemAct) as durable workflow adapters over
AbstractRuntime, plus their shared logic (tools contract, generation params,
loop hooks, allowlist handling). This backlog is the package's durable
planning memory.

## Counts (2026-07-15)

| State | Count |
|---|---|
| Planned | 0 |
| Proposed | 15 |
| Completed | 7 |
| Deprecated | 0 |
| Recurrent | 2 |

(0028 added 2026-07-13: hook taxonomy multi-emit + report accumulation —
audit residue from the production-readiness wave, promotion gated on a hook
consumer report or the next loop_hooks contract revision. 0029 added
2026-07-13: whole-package audit residue — the unfixed findings from the
completed fable5 pass whose P0 [stale tool-result replay] and four other
findings were fixed same-day; 0011 gained the audit's fresh evidence for the
sibling orphan-tool-message class. 0030 added 2026-07-13: capability-stack +
API-honesty residue from the operator-directed twin audits — eight defects
fixed same-day [changelog]; the residue holds the skills-attachment contract
doc, MCP side-effect guard + the wire-name P1 flagged to core, delegate
substrate palette, work-door task reset + unattended recipe, emit inventory,
default-model and max_tokens decisions.)

## Next recommended work

1. Everything in `proposed/loops_improvement/` waits on its stated promotion
   criteria (mostly maintainer green-light per proposal ID; `0022` C5 is an
   explicit maintainer ruling; `0015` A2 is gated on runtime's executor lane).
   The hooks lane is fully closed (0026 + 0028 completed 2026-07-15).

## Planned items

| ID | Item | Area |
|---|---|---|
| (none) | | |

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
| 0029 | `proposed/loops_improvement/0029_audit_residue_2026_07_13.md` | residue: only #9 (0024 green-light) + #10's conclusion-call half (0021) remain |
| 0030 | `proposed/loops_improvement/0030_capability_and_honesty_residue.md` | per-finding; surgical parts next defect batch, decisions need green-light |

## Topic tracks

- **loops_improvement** (mixed): the 2026-07-12 loops meta-audit proposal set
  (maintainer-endorsed IDs A1–A4, B1–B4, C1–C5, D1–D4 + hooks follow-ups).
  Defects in `planned/loops_improvement/` (README there), features/decisions
  in `proposed/loops_improvement/` (README there, includes the suggested
  execution order as proposed to the maintainer).

## Completed ledger

| Date | ID | Item | Outcome |
|---|---|---|---|
| 2026-07-15 | 0028 | `completed/loops_improvement/0028_hook_taxonomy_multi_emit.md` | Substance shipped in the 2026-07-14 batch-4 contract wave (one turn_end per turn, per-turn report resets, parse common core, RENAMED_STEPS); promoted at this pass per its own execution note. |
| 2026-07-15 | 0026 | `completed/loops_improvement/0026_loop_hooks_follow_ups.md` | Hooks parity + terminal honesty: `init` and `parse_tool_calls` (→ canonical `tool_proposed`) now fire on all three loops; the vacuous mutation pin rewritten as a vandal-handler pin (act payload's live `args` covered); true terminals emit `inbox_undelivered` for durable-inbox guidance that landed after the last drain (entries preserved, never consumed; composition handoffs exempt). Inventory + docs/hooks.md updated. Suite 233/2. |
| 2026-07-14 | 0011 | `completed/loops_improvement/0011_sibling_transcript_repair_strict_providers.md` | Sibling transcripts satisfy strict providers via EXTRACTION: ReAct's proven pipeline moved to shared `adapters/transcripts.py` (durable `tool_calls` preservation, orphan repair both directions), all three adapters delegate; ReAct byte-identical (prefix pins as harness); latent foreign-id class fixed. Pins in `test_sibling_strict_transcripts_0011.py`. Suite 221. |
| 2026-07-13 | 0027 | `completed/loops_improvement/0027_review_failure_degrades_to_accept.md` | Verifier failure contained in both forks via runtime `_absorb_failure`: held answer accepted, run completes, loud #FALLBACK (scratchpad marker + report line + `review_skipped` emit). Agent half of the c1128 review re-default condition; adapter default stays opt-in (flip is abstractcode's, c1139). Suite 144/2. |
| 2026-07-13 | 0010 | `completed/loops_improvement/0010_codeact_fenced_block_intent_bug.md` | FINAL check precedes fenced extraction (a final answer never executes); fallback flag-gated (`codeact_fenced_fallback`, default ON — deviation reasoned in report: default flip belongs to A1's wave). 3 pins. Suite 156/2. |
| 2026-07-13 | 0012 | `completed/loops_improvement/0012_sibling_delegate_budget_honesty.md` | CodeAct+MemAct delegate children now honor the shared schema (explicit arg wins; else parent budget, floor 20 — ReAct's ruled resolution mirrored). 8 parametrized pins. |
| 2026-07-13 | 0013 | `completed/loops_improvement/0013_memact_dead_knobs.md` | Dead knobs REMOVED from MemActAgent (TypeError on use); docs/agents.md aligned (stale ReAct claim corrected). ReactAgent's unread `plan_mode` named for 0024. |

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
