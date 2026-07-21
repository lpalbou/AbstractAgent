# Adversary 3 — synthesis attack + final ranked recommendations (2026-07-16)

(Attack pass over adversary 1's credibility audit + adversary 2's gap analysis; every
load-bearing claim re-verified against code + backlog. Full text preserved from the pass.)

## 1. NIH audit corrections to adversary 2
- Reflexion (a): verdict holds; wrong exhibit credited — the within-run analog is the REVIEW
  verdict path (not parse retries); model memory is the TRANSCRIPT (inbox drains append
  durable user messages, react_runtime.py:1116-1123), scratchpad.cycles is host-only except
  the one conclusion render. Sketch B UNDERSOLD: the distilled lesson text already exists
  (conclusion call output: progress/answer/uncertainties/next steps, :2354-2366) + the
  machine-readable outcome field (:2525) + reset_react_task(carry_messages=False,
  handoff_note=…) IS the Reflexion episode boundary. Only door wiring + containment missing.
  B is the one residue where Reflexion's hard precondition (reliable external failure signal
  — the WebShop lesson) is genuinely satisfied (door gates + outcome field).
- Meta-Reasoner (b): rejection right in substance, understated inventory — a HOST-side
  controller is already buildable with zero loop changes (hooks = the progress summary;
  Runtime.steer() sidecar = the strategy channel, runtime.py:1295-1320). Genuine gap:
  mid-run generation-param change (sidecar carries text only; _runtime.thinking host-editable
  only at task boundaries via reset_react_task). Deterministic degenerate case (detect
  no-progress → conclude) IS backlog 0017, filed before the report existed. Caveat kept:
  steers do not wake parked runs (:1304-1306).
- ReWOO (c): verdict survives, RATIONALE WRONG — caching kills re-prefill only; the real
  capture is batching (logic/react.py:128-132) + runtime 0214 CONCURRENT read-only batch
  execution (tool_executor.py:28-58, ThreadPool max 8, fail-safe serial) + VisualFlow wired
  pins as #E1..#En. Honest residue: model-authored dependent chains (LLMCompiler proper) —
  real decode/latency savings on slow local substrates, well under the paper's 5x once
  caching is in, zero consumer incidents. Keep dormant.
- Other four verdicts confirmed on spot-check; no glossed dimension.

## 2. Missing-topics fold: NO new build candidate
- LLMCompiler parallel tool DAGs: cheap win ALREADY SHIPPED (0214). Weakens sketch C further.
- Context compaction: house answered with 0018 (handoff artifact, never compaction — Amp
  retired compaction for cause); new observation: 0018 is a structural dependency of any
  immortal resident.
- Thinking-budget scheduling: exists at zero loop cost (_runtime.thinking per-call,
  delegate-inherited; door edits at task boundaries) — needs a RECIPE DOCUMENT, not a build.
- LATS/MCTS: dies with ToT (issuance idempotency; unforkable side effects). PRMs: none exist
  for our domains; graph-layer gates are the house line (0019 resolution). Debate: app-layer
  votes exist; reasoning-only = flow. Plan-and-execute: covered (update_plan + delegate +
  coding-agent). Computer-use: different lane.
- META-FINDING: the pre-existing loops_improvement backlog already contained every actionable
  idea; the report added zero net-new build items; its one live citation (2607.01641)
  validates 0017, filed before the report existed.

## 3. Sketch attack
- A (parked resident): unlisted failure mode = UNBOUNDED TRANSCRIPT in an immortal resident
  (reset_react_turn preserves the life; a generic resident has no entity-memory context
  system; ADR forbids trimming). Survives with lifecycle rule (reset_react_task per task
  family, or gate on 0018). Second: steers don't wake parked runs — document or wake-after-
  steer at the door. NOT a double-build (flow's event-inbox agent = raw rebuild with none of
  the hardening: no repeat guard/review/act_seq/hooks/conclusion). RIGHT SHAPE: packaged
  abstractagent composition. Effort M. Consumer-gated.
- B (failure lessons): containment DESIGNED (answer, not kill): (1) family scoping
  (workflow_id + task-template hash, door-assigned); (2) SUCCESS-CLEARS retirement (kills the
  rich-get-richer attractor); (3) ≤3 lessons/family FIFO, bounded + truncation marker;
  (4) provenance labels via handoff_note (metadata.kind=task_handoff); (5) explicit door
  seeding, NEVER similarity recall (the decoy-channel lesson — reject "run the door on
  MemAct" as the transport). Effort S-M. Kill: zero repeat-failure delta or token cost >
  failure cost saved.
- C (parallel delegate join): unlisted failure modes: (1) the child-completion→event bridge
  must be DURABLE and the existing hook is not (Runtime._terminal_hooks process-local,
  best-effort, runtime.py:962-966 — crash parks the parent forever); (2) orphan spend +
  cascading cancel (async children of a dead parent; failed join + fresh act_seq can double
  the fleet). Needs children registry, idempotent join keyed on sub_run_id set, cascading
  cancel. Honest effort L. Build only on a recorded bottleneck incident.
- D (dormant flows): survives; record they'd live in flow's example library.

## 4. FINAL RANKED LIST
BUILD NOW:
1. 0017 no-progress/oscillation detection (lane a, S) — read-only repeat loops spin uncounted
   today (duplicate guard covers side-effect tools only, :1332-1338); validated by the
   report's one live source; the deterministic Meta-Reasoner core. Sharpening: streak keys on
   (tool, args, RESULT FINGERPRINT) — post-edit re-reads are legitimate. Kill: false
   positives on legitimate post-edit re-reads.
2. Sketch B hardened (lane b, S-M) — with the 5-point containment. Proof: recurring-family
   bench, lessons on/off. Kill: zero improvement or cost > savings.
FILE GATED:
3. Sketch A packaged parked resident (lane b, M) — promotion: one named consumer beyond
   entity visits; ships with transcript-lifecycle rule + steer-doesn't-wake caveat; shape
   ruling: abstractagent composition, not flow template.
4. Thinking-per-task-boundary recipe (lane a, S, docs-only) — immediate on green light;
   param-lane steer extension promotes only when text steering proves insufficient.
5. 0018 annotations — house answer to "context compaction"; structural dependency of
   immortal residents. Criterion unchanged.
6. Sketch C parallel delegate join (lane a, honest L) — promotion: a recorded sequential-
   delegation bottleneck; constraints on file: durable bridge, cascading cancel, idempotent
   join registry.
REJECT (one-liners): ReWOO in-adapter (banked via batching+0214+flow wiring); ToT/LATS/debate
in-loop (issuance idempotency; unforkable effects); model-controlled set_thinking/bandit
(self-granted spend); in-run auto-compaction (ADR + 0018 non-goal; Amp retired it); Toolformer
(training-time; below ReAct in the report's own table); AutoGPT 2.0 (does not exist as
described); standing Self-Refine loop (the +103% case); PRM-guided search in-loop (no PRM for
our domains); computer-use loops (different lane).
RECORD AS RIGHT: step-cap/loop-detection discipline (external validation of
max_iterations+guards+act_seq AND 0017's gap); Reflexion's core insight (shipped as
review→steer; fresh-episode lands at run boundaries); ReWOO's economics (implemented as
0212/0213 cache stability without the brittleness); trace/termination checklist (exceeded);
META: three passes over a fabricated report converged on ~90% already-filed backlog — the
loop-audit process is validated, generated surveys are keyword lists only.

TWO HONEST DEBTS SURFACED (corrections to OUR record, not code): adversary 2's ReWOO
rationale (caching is not the capture mechanism), and the undocumented fact that read-only
parallelism was already banked by 0214.
