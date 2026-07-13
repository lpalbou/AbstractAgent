# Loops improvement track (proposed half)

## Status
Proposed (maintainer endorsed the point set 2026-07-12 — "i think they are all
good points, for the moment, make sure they live in your dedicated backlog" —
which is endorsement of the ideas, NOT a build order; each item promotes on
its own criteria).

## Purpose
Durable memory for the loops meta-audit proposal set (IDs A1-A4, B1-B4, C1-C5,
D1-D4) minus the defect items already in `planned/loops_improvement/`
(0010=A4, 0011=C2, 0012=C3, 0013=C4). Provenance: two adversarial audits
(integration map + capability/parity) plus a best-practices research pass
(OpenHands, smolagents, LangGraph/AutoGen, Anthropic/Amp context-management
findings), 2026-07-12.

## Items
- `0014_native_code_action_primary_path.md` (A1): native execution calls as
  the documented primary path.
- `0015_persistent_interpreter_session.md` (A2): Jupyter-style kernel per run
  — runtime builds (seam agreed c1110-c1112), we co-design + consume.
- `0016_execution_trust_tiers.md` (A3): owned-host / workspace-contained /
  isolated, operator-configured.
- `0017_no_progress_oscillation_detection.md` (B1): repeat/oscillation streak
  forces the loud conclusion path.
- `0018_structured_handoff_long_runs.md` (B2): explicit handoff artifact,
  never compaction.
- `0019_deterministic_turn_end_verification.md` (B3): turn-end verdict hook as
  a contract.
- `0020_provider_stop_semantics_over_heuristics.md` (B4): gate the deferred-
  action nudge off under native stop.
- `0021_shared_adapter_core_refactor.md` (C1): ~1,200 duplicated lines out;
  every wave lands once.
- `0022_codeact_fold_decision.md` (C5): fold CodeAct into ReAct — MAINTAINER
  RULING REQUIRED.
- `0023_memact_status_honesty.md` (D1): experimental label or root export.
- `0024_dead_public_surface_pruning.md` (D2): dead factories/stubs/re-exports.
- `0025_cross_seat_integration_flags.md` (D3+D4): other seats' lanes, tracked.
- `0026_loop_hooks_follow_ups.md`: named P2/P3 leftovers from the hooks wave.

## Reading order / suggested execution order (as proposed to the maintainer)
planned 0010+0012+0013 (trivial, same day) -> 0017+0019+0020 (small, high
leverage) -> 0021 with planned-0011 inside it -> 0015 behind runtime's
executor lane -> 0016+0018 as designed follow-ups -> 0022/0023 on ruling.

## Governing ADRs
None identified after review. 0016 and 0018 likely NEED decision records when
promoted (trust policy / context policy).

## Scope
abstractagent's three loop adapters, shared logic, public surface, and the
seams they consume.

## Non-goals
Executor/isolation implementation (runtime's lane per the agreed seam);
entity chat-lane work (entity-agency plan's territory); anything before the
respective promotion criteria are met.

## Notes for future agents
Keep the A/B/C/D IDs visible in reports — they are how the maintainer
references these items. The full audit evidence lives in the 2026-07-12
conversation record and the two adversary reports; the code-reality sections
in each item carry the load-bearing line references.
