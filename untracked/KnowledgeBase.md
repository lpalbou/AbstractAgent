# AbstractAgent seat — KnowledgeBase (untracked, accumulating)

Critical insights, logics, and lessons for the agent seat (abstractagent
package). Append-only in spirit: deprecated insights move to the DEPRECATED
section with reasons, never deleted.

## Verifier lane

- **Signal source beats loop sophistication (R-Type series, 2026-07-17)**:
  LLM-read verification blessed a crash-on-first-input game, an every-frame
  ReferenceError, and a corner-ninth draw; only EXECUTION caught all three.
  The lane principle: execute where an executor exists; LLM-read only where
  none does. In-loop this is DECLARATION-DRIVEN (tools opt in via
  ToolDefinition.tags ["executor"]; the verifier prompt names them), never
  hardcoded tool names — executors are environment capabilities that live
  tool-side (abstractcode's browser_probe), never in this package.
- **Instrument PRESENCE dominates**: with the executor granted, 4/4 arms
  shipped working artifacts and bare react self-probed 9-11x unprompted
  (organic pickup is universal). The tag's job is making the executor exist
  in the verifier's world; availability does most of the work.
- **The review-budget re-arm defect (fixed 2026-07-17)**: the per-answer
  reset (`review_count = 0` after tool execution) fired for VERIFIER-FORCED
  batches — the verifier re-armed its own budget through calls it forced
  itself, unbounding re-review on one unchanged answer (live: 40-min wall
  cap burned). Structural fix (`_temp.review_forced_batch`: forced batches
  consume the budget, model-issued activity still re-arms) beats a
  symptom-shaped counter ("N green probes") — the actual pathology was
  probe-once-then-READ, which the counter would never catch. Bound the
  CLASS, not the symptom.
- **Verifier failure is never artifact failure (0027)**: a terminally failed
  verifier call degrades to accept-with-#FALLBACK via runtime's opt-in
  `_absorb_failure` payload key. This mechanism generalized: flow's
  `continueOnError` node config now maps onto the same key — when another
  seat prices "new machinery", check whether the kernel already ships the
  floor.
- **Verifier prompts must be grounded in gate/tool outputs**: an LLM judging
  from source re-derives executes-shaped opinions; grounded in what actually
  ran, it judges only the gap that is its job. Discipline line: "Be strict:
  only count actions supported by the tool outputs."

## Visit/entity lane honesty

- **Chrome classes (c2447)**: loop-position tails, [plan] renders, conclusion
  scaffolding, and the "[Operator guidance …]" wrapper are TASK-AGENT chrome;
  under `_runtime.suppress_loop_tail` (set by runtime's visit BRIDGE) all of
  it is suppressed or host-voiced. Task lane byte-unchanged is a test-pinned
  invariant every time.
- **Attribution honesty**: a host-authored retry nudge must not wear operator
  words. The visit wrapper claims only what is true for ALL inbox sources
  ("not from your visitor"). The machine key `kind="operator_guidance"` is an
  acknowledged misnomer (means "drained from the inbox") — no audit surface
  may read it as operator attribution; the source-split (additive `source`
  key) is the repair, gated on a live incident.
- **Never a parse anchor**: entity-visible bracket prose can be imitated by a
  visitor; machine detection keys on structured metadata only
  (tools_ran-not-prose law, applied to wrappers).
- **Vocabulary crossing seats**: strings that rest in durable transcripts get
  ruled (runtime = visit voice owner, semantics = spelling chair) BEFORE
  shipping. Spelling is visitor-coupled — a visitor-less lane adopting the
  knob needs a second ruled spelling, never reuse.
- **Frozen-set sync class**: A2's election-fence exclusion
  (`_ELECTION_FENCE_LANGS` runtime-side) must grow in the same change as any
  new election convention, or the nudge starts firing on real acts — the
  diary_type-clamp drift class. Name the sync rule beside the set.

## Cross-seat process

- **Spec-then-build across seats works same-hour**: A2 (format-repair nudge)
  went spec (agent) → build to spec (runtime) → adversary-found spec gap →
  gap folded back as spec text (clause 2a) in one evening. The spec owner
  adopts implementation-found precisions as SPEC TEXT, not as implementation
  trivia.
- **Structural detection clauses need an "our own conventions" exclusion
  audit**: my "any key=value info string is tool-shaped" clause flagged the
  driver's own election fences. Generality rules must be checked against the
  host's own legitimate vocabulary before shipping.
- **Receipts discipline (Option A, ruled 11-0 + operator confirmation
  2026-07-18)**: file = state (docs/backlog/, id `abstractagent-<NNNN>`),
  hub = obligations; claims are pointer rows with no status prose; receipts
  carry machine-checkable evidence; close cites receipts. Join headers on
  next touch, no bulk rewrite.
- **This week's corruption-wave lesson**: repo trees carried truth while hub
  claim rows froze. Receipts are REPO facts (tests, commits, CHANGELOG);
  reconstruction reads repos first, hub records second.

- **Structure without a mandate dies (A ruling, 2026-07-20)**: the act-only
  ref layer (refs in transcripts + send-time dereference) was OUR invention,
  never laurent's law — when he ruled "everything lives in the runtime, diary
  = the AI's experiential notes", both halves deleted same-day (runtime's
  resolver, my minting). The privacy property that SURVIVES is the one with a
  mandate: the write-boundary diary capture (words fly to the book at the
  result boundary). Lesson: when building privacy machinery, distinguish the
  operator's actual law from the team's inferred elaborations — elaborations
  are deletable and should be built cheap.

## Session mechanics (this seat)

- **Session reconstruction**: on respawn, verify the tree (run the suite),
  read the dead session's transcript tail, and reconcile hub claim rows
  before posting anything. State is reconstructed, never assumed.
- **MCP bridge can drop mid-session** (persistent "Not connected" after
  hours of good calls; observer hit the same class the same night). The
  `agora` CLI is the full-fidelity fallback (inbox/read/post/ack/work all
  work); the background listener is CLI-based and unaffected. Don't
  retry-loop the MCP; note it and continue via CLI.
- **Shell quoting for agora post**: bodies with backticks/nested quotes break
  inline `-c` quoting — write the body to a temp file and use `"$(cat f)"`.

## DEPRECATED

(none yet)
