# Adversary 2 — gap analysis: report techniques vs shipped loops (2026-07-16)

HEADLINE: zero techniques justify a loop change. 7/10 already shipped, usually stronger
(durable, crash-replay-safe, cost-audited, honest about failure). Three genuine residues are
all graph-layer compositions (the coding-agent precedent). Checklist scores 6/6 satisfied —
two of its literal recommendations (live prompt sections, KV step-cache) are what 0212/0213
exist to prevent.

## Verdict table
1. Self-Verification — ALREADY-SHIPPED twice. In-loop review_mode default ON at facade
   (agents/react.py:75-84; containment via _absorb_failure at react_runtime.py:2175-2194;
   per-answer review budget :1872-1878; evidence-grounded critique :2109-2112; can force
   next_tool_calls with transcript repair :2227-2250). Graph layer: coding-verify-gates
   independent fresh-context verifier agent + failure-specific reprompting (coding-agent).
   House division: review catches reasoning drift, gates catch broken artifacts. Cost known:
   +103% tokens for zero delta beside an external check; --no-review exists.
2. Self-Refine — ALREADY-SHIPPED (review loop = generate→critique→revise, review_max_rounds=3,
   per-answer reset; MemAct mandatory structured finalize with draft-preserving containment
   memact_runtime.py:1148-1161, 1186-1203). Refining without new evidence is the exact +103%
   warning case. Pure prose-refinement loop = trivial two-node flow if a consumer appears.
3. Event-driven — ALREADY-SHIPPED at the durable layer (WAIT_EVENT/WAIT_UNTIL, events_inbox
   cursor-drain contract, event-inbox-react-agent 48-node flow; loop_hooks.py:37-41
   message_drained seam; final_next_node composition knob react_runtime.py:805-816;
   ReactMiddle visit merge). RESIDUE: no GENERIC (non-entity) packaged resident → sketch A.
4. Reflexion — within-run half SHIPPED (bounded parse retries :1429-1458 fence-blind heuristic
   :422-460; review steering through durable inbox :2257-2259; coding-agent failure feedback
   across rounds in durable cg.loop_state). RESIDUE: cross-run episodic failure memory for
   task agents → sketch B (primitives exist: remember_note :1616-1625, recall_memory
   :1583-1592, task_reset handoff_note task_reset.py:82-95; nothing writes a lesson on failed
   gates; reset_react_task wipes scratchpad deliberately).
5. AutoGPT hierarchical planning — SHIPPED as delegate + flows (durable children, recursive
   delegation + ask_user excluded :1646, budget inheritance floor-20 :1668-1694, host-gated
   substrate palette :1734-1792 — model cannot self-escalate; update_plan :1556-1581; CodeAct
   plan_mode codeact_runtime.py:422-486; plan rides volatile tail cache-safely :1186-1198).
   RESIDUE: delegation is SEQUENTIAL (async:False :1794-1803) while runtime supports
   fire-and-forget children (runtime.py:3460-3469 returns sub_run_id) — missing the JOIN →
   sketch C (the only loop-change candidate).
6. MEGALoop — memory-augmented planning SHIPPED (MemAct compose memact_runtime.py:336-344,
   effect :512, failed-compose loudness :414-432; finalize envelope :1122-1161; entity stack
   beyond the article). ≥10k-step single-loop REJECT: violates max_iterations=20 ruling +
   unbounded-growth class; long horizons decompose at the graph layer.
7. ReWOO — SHIP-AS-WORKFLOW (low priority). Token play mostly pre-captured: byte-stable system
   prompts (logic/react.py:111-116), volatile tails excluded from cache fingerprint
   :1215-1222, explicit multi-call batching (logic/react.py:128-130). Remaining: zero LLM
   calls between tool steps — only pays on STATIC pipelines; literally a VisualFlow
   (structured llm_call plan → wired tool_calls → folding llm_call). In-adapter
   observation-decoupling would forfeit the loop's reason to exist.
8. ToT — REJECT in-loop (branching actions incoherent with linear issuance-counter idempotency
   :1839-1850 act_seq; can't fork a filesystem). Reasoning-only branching = 6-node flow
   (N parallel llm_call + judge). Cost N×depth multiplicative, worse than the measured +103%.
   Build only when a concrete reasoning-only decision point shows branch-worthy failure rates.
9. Meta-Reasoner — REJECT dynamic control; static knob SHIPPED (_runtime.thinking normalized
   generation_params.py:44-85, applied :478-485, inherited by delegates :1726, rides params
   not prompt bytes). Objections: controller cost unvalidated; model-controlled set_thinking =
   self-granted spend (the class the substrate palette structurally forbids). If evidence
   lands: host-side hook policy setting thinking at task boundaries via task_reset — not loop
   code.
10. Toolformer — REJECT (training-time technique; we consume substrates; tool_support
    capability bit + grant-derived native declaration is our correct exposure; report's own
    table shows it below ReAct on 5/6 benchmarks).

## Checklist 1-6 vs stack: all satisfied (details: byte-stable prompt contract beats "live
sections"; toolset_id staleness refresh :172-176, 1075-1089; MemAct compose + deliberate
ReAct memorylessness; ruled max_iterations + deterministic conclusion :2309-2537 (STOP-token
hacks rejected :10); review default-on + turn-scoped repeat guard :1338-1408 + warn pct
:1138-1140 + issuance idempotency (three layers vs src_006); ledger + scratchpad.cycles
:1312-1317, extend-never-assign :2005-2016 — execution-exact replay, not just a log.)

## Design sketches (actionable residues)
A. GENERIC PARKED RESIDENT — SHIP-AS-WORKFLOW. Package the visit composition minus identity
   seams: PARK (wait_event + durable-inbox drain) → seed messages → reset_react_turn → full
   ReAct cycle via final_next_node → deliver seam → re-PARK. Zero adapter changes. Risk:
   fleet-scale leans on unlanded runner fairness (GW-D); one resident fine. Proof: kill -9
   mid-burst resume with history intact vs bridge seat losing in-flight state; idle-cost vs
   the agora listen poll loop.
B. CROSS-RUN FAILURE LESSONS AT THE WORK DOOR — SHIP-AS-WORKFLOW. On outcome=iteration_budget
   or failed gates, the DOOR writes one bounded lesson (remember_note/channel store) and seeds
   the next related task's handoff_note or recall query. Machine-readable outcome field
   shipped for exactly this consumer :2289-2297. Risks: stale lessons poisoning unrelated
   tasks (rich-get-richer attractor class); redundancy with carry_messages. Proof: repeat-
   failure rate on recurring task families, lessons on/off, with token deltas; kill if zero.
C. PARALLEL DELEGATE FAN-OUT + DURABLE JOIN — the ONLY LOOP-CHANGE-CANDIDATE. N delegate calls
   as async:True subworkflows; parent parks on WAIT_EVENT until all children complete;
   observations fold in arrival order. Must live in the adapter: act-node owns the
   tool-observation contract (call_id pairing, transcript repair) + budget/allowlist/substrate
   inheritance :1646-1694. Runtime half needs child-completion→event bridge. Cost: N× tokens;
   join idempotency care. NO recorded incident where sequential delegation was the bottleneck
   — build on demand, not on the report.
D. ReWOO-as-flow / ToT-as-flow — dormant until a consumer exists; both expressible in today's
   VisualFlow vocabulary.

## Top-3 recommendations (with counters)
1. Package the generic parked resident (A). Strongest consumer story; zero loop-code risk.
   COUNTER: bridge seats live-proven at 3-seat scale; GW-D unlanded; "who needs a third
   resident shape this month?" may be nobody.
2. Failure-lesson notes at the work door (B), gated on measurement with a kill criterion.
   COUNTER: MemAct is the designed home for cross-task memory (zero new machinery
   alternative: run the work door on MemAct); the only hard datum on added-context aids is
   +103% for zero delta. Ship as an experiment with the bench, not a default.
3. Adopt nothing else from this report — and record why (negative result as deliverable:
   7/10 shipped stronger, checklist 6/6, two literal recommendations are our recorded
   anti-patterns, citation hygiene poor). Parallel delegate join (C) stays a named pre-sketched
   candidate awaiting a real bottleneck incident. COUNTER: NIH failure mode; ReWOO +12
   ALFWorld / ToT +9 MATH deltas "are real papers even if packaging is sloppy" — [NOTE FROM
   THE ORCHESTRATING SEAT: adversary 1 subsequently PROVED those two specific numbers
   fabricated — ReWOO never ran ALFWorld, ToT never ran MATH — so this counter-argument's
   examples are dead; the general NIH caution stands.] The defensible hedge: both are one
   flow away whenever a consumer appears — the graph layer is where they should wait.
