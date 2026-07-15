# Loops improvement track (planned half)

## Status
COMPLETE (2026-07-15 hygiene note): every defect item in this half shipped —
0010/0012/0013/0027 (2026-07-13), 0011 (2026-07-14, via the transcripts.py
extraction). The files live in `completed/loops_improvement/`; this README
stays as the track's provenance record. The feature half of the same track
lives in `proposed/loops_improvement/`.

## Purpose
Fold the 2026-07-12 loops meta-audit into durable planning memory. Two adversarial
audits (integration map + capability/parity audit) and a best-practices research
pass produced a maintainer-endorsed proposal set referenced by IDs A1–A4, B1–B4,
C1–C5, D1–D4. This planned half carries the items that are DEFECTS in shipped
behavior — no maintainer ruling is needed to fix a bug in code this package owns.

## Items
- `0010_codeact_fenced_block_intent_bug.md` (A4): fenced code executes even in
  final/illustrative replies — order + intent fix.
- `0011_sibling_transcript_repair_strict_providers.md` (C2): CodeAct/MemAct
  orphan tool messages 400 on native OpenAI multi-turn tool use.
- `0012_sibling_delegate_budget_honesty.md` (C3): CodeAct/MemAct hardcode child
  budget 10 while the shared schema promises parent-inheritance.
- `0013_memact_dead_knobs.md` (C4): `review_mode`/`plan_mode` accepted and
  silently ignored.
- `0027_review_failure_degrades_to_accept.md`: verifier failure must degrade
  to accept-with-#FALLBACK, never kill a run holding a valid answer (live
  operator incident c1128, 2026-07-12).

## Reading order
0010 → 0012 → 0013 (independent, trivial) → 0011 (larger; may fold into the
proposed shared-core refactor 0021 instead — check its state first).

## Governing ADRs
None identified after review (this repo has no ADR system). ADR-0026 (workspace
convention: no silent truncation) is respected by all items.

## Scope
Bug fixes inside abstractagent's three loop adapters and shared logic.

## Non-goals
No new capabilities; no CodeAct fold/retirement (proposal C5, maintainer's
ruling); no executor/isolation work (runtime's lane per the c1110–c1112 seam).

## Notes for future agents
Provenance: adversarial audits 2026-07-12 (integration + parity), maintainer
endorsement same day ("i think they are all good points"). The full proposal
text with the A/B/C/D IDs is mirrored in the track READMEs; keep those IDs in
completion reports so the maintainer's references stay resolvable.
