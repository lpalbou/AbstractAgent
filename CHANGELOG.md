# Changelog

## Unreleased (2026-07-12)

### Fixed (2026-07-15 — verifier schema strict-mode expressibility, airelay 422 incident)
- **Review verifier schema was inexpressible under OpenAI strict validators**:
  `next_tool_calls[].arguments` declared a free-form dict
  (`{"type": "object"}` without `properties`) — OpenAI-strict backends
  (subscription relays, responses API) refuse the whole request with a
  deterministic 4xx (`invalid_json_schema` / "Extra required key 'arguments'
  supplied"), killing every ReAct/CodeAct review cycle on those providers
  (operator-reported: airelay gpt-5.4, `LLM_CALL failed ... 422`). Fix: the
  shared `verifier_response_schema()` (new, in `generation_params`) declares
  `arguments` as a JSON-ENCODED STRING — the same convention OpenAI's own
  function-calling wire format uses — and parse sites normalize through
  `coerce_verifier_tool_arguments` (dicts pass through for lenient providers
  and legacy transcripts; JSON-object strings decode; anything else degrades
  to `{}` exactly as before). ReAct and CodeAct review nodes now share one
  schema instead of two divergent literals. Live-verified: the new schema is
  accepted natively with `strict:true` on the previously-failing backend.
  Pins in `tests/test_verifier_schema_strict.py` (strict-rules checker walks
  every object node: properties + required-all + additionalProperties:false).
  Defense-in-depth: abstractcore independently gained a schema-rejection →
  prompted-lane fallback, so even inexpressible schemas survive (see
  abstractcore CHANGELOG).

### Fixed (2026-07-14, batch 8 — backlog 0011: sibling transcripts on strict providers)
- **CodeAct/MemAct 400'd on native OpenAI at iteration 2**: both sibling
  parse nodes appended assistant messages CONTENT-ONLY while their
  sanitizers emitted `role:"tool"` + `tool_call_id` — every tool message was
  an orphan under the native tool-calling contract ("assistant tool_calls
  must be followed by matching tool messages"). Unsurfaced because the
  loops' one consumer runs local prompted providers. Fixed by the
  0011-preferred EXTRACTION: ReAct's proven pipeline now lives in
  `adapters/transcripts.py` (durable `tool_calls` preservation, orphan
  repair both directions, adjacent-user merge) and all three adapters
  delegate to it — a shared implementation, not a third copy. The sibling
  parse nodes append assistant messages WITH `tool_calls` metadata; the
  loops' one real difference (CodeAct/MemAct marked-truncation bounds)
  rides in as a hook. ReAct's behavior is byte-identical (its prefix pins
  were the extraction's regression harness).
- **Latent in ReAct's own pipeline, caught by the extraction pins**: a tool
  message with a FOREIGN id inside an answering run passed through — strict
  providers 400 ("tool_call_id not found in tool_calls"). Foreign-id tool
  messages now fold into inert user notes AFTER the announced block, so the
  answered run keeps provider-required adjacency.
- 4 new pins (`tests/test_sibling_strict_transcripts_0011.py`): the
  backlog's validation shape (two-iteration tool run per sibling satisfying
  the strict contract), orphan repair both directions, honest interactive-
  builtin labeling. Suite 221.

### Docs (2026-07-14, 0023 — MemAct labeled experimental)
- MemAct is now labeled **experimental** everywhere users meet it
  (docs/agents.md section + table, README feature line, docs/api.md note),
  with the rationale stated (distinct Letta-style architecture worth keeping;
  one opt-in consumer; less production mileage) and the deep import framed as
  the label made structural. Also corrected agents.md's stale claim that
  legacy plan/review kwargs "raise TypeError" — the shipped behavior is the
  one-release deprecated-ignored shim (loud DeprecationWarning on non-default
  values). The root-export question stays open as maintainer preference.

### Docs (2026-07-14, coredoc truth-up after batches 4-7)
- `llms.txt` / `llms-full.txt` regenerated against the shipped surface:
  `task_reset.py` and the unattended recipe indexed, `read_skill` named on
  the builtins entry, `docs/skills-attachment.md` added to the full index's
  docs list, and the new test files (gated-remainder + audit-residue pins,
  verifier containment, emit-inventory drift, visit merge, act-only) listed.
- `docs/api.md`: documented the schema-only builtins (incl. `read_skill`
  with the both-halves wiring rule), a new "Task reset (queued work)"
  section for `reset_react_task`, and the outputs contract additions
  (`outcome` vocabulary, always-present skip flags, honest budget-terminal
  answers). `docs/README.md` links the task-reset section.
- **Explicit legacy budget silently capped at 20 (#6)**: `ensure_*_vars`
  called `ensure_limits` first, materializing `_limits.max_iterations=20`,
  and the resolver prefers limits — so a raw
  `runtime.start(vars={"max_iterations": 100})` ran capped at 20 (facades
  were unaffected; they set `_limits`). All three ensures now capture
  whether the CALLER set a limits budget before materialization and seed
  `_limits` from the explicit legacy/flat value when it didn't. The ruled
  20 stays the no-value default; a caller-set `_limits.max_iterations` is
  never clobbered by a stale flat key. Sharpened in MemAct: it never
  migrated the flat `iteration`/`max_iterations` keys at all — an explicit
  raw budget was ignored entirely, not just out-prioritized.
- **Sibling budget terminals answered with raw observations (#10)**:
  CodeAct/MemAct's `max_iterations` output took the LAST transcript message
  as the "answer" — often a tool observation or recovery note, lying about
  what the agent said. Both now answer with the agent's last assistant
  words (MemAct prefers the held `draft_answer` — what finalize would have
  polished), append a final assistant message (`kind=final_answer`,
  `budget_exhausted: true`) so the transcript ends honestly, and state
  plainly when no words exist. Full conclusion-call parity with ReAct stays
  gated on 0021.
- **MemAct compose spun forever on a non-dict KG result (#12)**: a custom
  host handler returning a string/list fell through to "schedule KG query"
  and idempotency replayed the identical effect instantly — an infinite
  spin consuming no iteration budget. Present-but-non-dict now fails the
  compose loudly (`compose` emit with `ok: false` + `#FALLBACK` error) and
  continues to reason; the applied-key latch prevents re-query.
- **Unmarked truncation at tiny bounds (#15a)**: CodeAct's two small-budget
  truncation branches dropped the marker entirely (`suffix = ""`) —
  a silent lossy slice (ADR-0026 violation). Both now keep a short marker
  at any bound (ReAct's `_review_truncate` shape).
- **Delegate explicit-0 budget widened (#15b)**: an explicit
  `max_iterations=0` tool arg silently widened the child to
  `max(parent, 20)` while `resolve_max_iterations` clamps explicit 0 to the
  floor of 1 — inconsistent explicit-narrow handling across the two
  surfaces. All three delegate branches now clamp explicit <=0 to 1;
  absent/unparseable still inherits (with the ruled-20 floor guard).
- **Repeat guard disengaged against skipped cycles (#7)**: a guard-skipped
  cycle records the PROPOSED batch but no observations — the next identical
  batch compared against it, found no successful observations, and executed:
  the protection alternated skip/execute/skip on a persistently repeating
  model. Skips now stamp `repeat_skipped` on the cycle and the prev-cycle
  scan passes over marked cycles to the one that actually executed.
- **Hook steering leaked for failure-terminated runs (#14)**: `discard_run`
  fired only from the done/max_iterations terminal NODES — a run that
  terminates by FAILURE or cancellation never executes one, so its queued
  steering rotted in `LoopHooks._pending` forever on long-lived hosts. Two
  halves: the facades now discard on FAILED/CANCELLED (`run_to_completion` +
  `resume` — the deterministic paths no node covers, with a loud `#FALLBACK`
  log when entries drop), and `LoopHooks` gained a `max_pending_runs` cap
  (64; FIFO eviction with a loud stderr note) as the structural backstop for
  hosts driving the runtime directly. Live runs drain at every reason
  boundary, so evictions under sane concurrency only ever hit dead runs.
- 12 new pins (`tests/test_claim_0029_residue_2026_07_14.py`); suite 217.

### Fixed (2026-07-14, batch 6 — wave-F contract adversary findings)
- **Routed-run capability-bit staleness (P1)**: the A1 fence default trusted
  `supports_native_tools` unconditionally, but the runtime seeds that bit
  from its CONFIG model's capabilities — a run routed to a different model
  via `_runtime.model` (the standard gateway channel) kept the wrong bit. A
  native-default runtime routing to a prompted model got fence OFF and the
  run completed SILENTLY with code-as-prose as the "answer". Fix, two parts:
  the CodeAct facade stamps `_runtime.tool_support_model` (the model the
  seeding describes) at start, and `fenced_fallback_enabled` distrusts the
  bit when the effective model differs from the stamp — failing toward fence
  ON (the loud degradation, not the silent completion). The resolver also
  accepts `tool_support: "native"` (same seeder, descriptive spelling) as a
  capability signal. Cross-lane ask filed: runtime should stamp
  `tool_support_model` beside the bit at seed time.
- **Stale steering crossed task boundaries (P2)**: `reset_react_task`
  carried `_runtime.inbox` — undelivered guidance addressed to task N
  drained into task N+1's transcript as "[Operator guidance — this amends
  the task]" OF THE WRONG TASK. The inbox now clears at the task boundary
  (the same reason hook-layer steering discards at terminal). `_runtime` is
  now deep-copied (the one-level copy still shared inbox/allowlist lists and
  palette profile dicts with the completed run's record).
- **Side-effect classifier false negatives (P2)**: the curated set missed
  shipped-registry mutators — `fetch_url` (never read-only-safe: the
  model-controlled method can POST), `shell_exec`/`shell_write_stdin`/
  `shell_close`, and the agora comms tools (`agora_post_message`/
  `agora_send_dm`/`agora_ack_inbox` — resident agents repeating a hub post
  is the duplicate-send class verbatim). All added.
- **Sibling budget terminals lacked the skip flag (P3)**: "review_skipped is
  an always-present bool" was false at CodeAct/MemAct `max_iterations`
  outputs — a door reading the documented shape got None exactly on
  budget-exhausted runs. Both siblings now carry the flag at both terminals.
- **CodeAct fenced action invisible in the parse core (P3)**: on the fenced
  path the parse payload said `has_tool_calls: false` and then the loop
  executed code. CodeAct's parse payload now carries additive `has_code`
  (computed by the same decision the extraction branch uses — one decision,
  two readers); hooks.md documents the `has_tool_calls || has_code` rule.
- **P4 sweep**: `used_tools` resets at the visit turn boundary (same per-turn
  latch class as `review_skipped`); parse emits copy list-shaped arguments
  too; `content_preview`'s `"(no content)"` sentinel documented; facade
  attachment path for `read_skill` documented (`agent.logic.add_tools`);
  task_reset docstring documents the consecutive-user-message behavior and
  the inbox exception.

### Fixed (2026-07-14, batch 5 — wave-E regression adversary findings on batch 4)
- **Visit-boundary turn resets (P1, live-reproduced)**: the 0028 per-turn
  resets landed only on the ask_user boundary; the VISIT composition boundary
  (`reset_react_turn`) — where turns actually cycle in gateway entity visits —
  was missed. Turn 2 of a composed run carried turn 1's stale
  `review_skipped` #FALLBACK into its report/output, and a second
  budget-exhausted turn announced `max_iterations_reached` ZERO times (the
  once-per-turn latch never re-armed). Both now reset in `reset_react_turn`;
  CodeAct/MemAct ask_user boundaries got the symmetric pops
  (`review_skipped`/`finalize_skipped`).
- **A1 parse pin was vacuous (P1)**: the native-leg assertion checked effect
  shape, but CodeAct's parse routes extraction via `next_node="execute_code"`
  and returns no effect on ANY branch — the pin passed against the pre-A1
  code too. Now pins routing + `pending_code` on both legs.
- **`reset_react_task` aliasing (P2)**: the "input not mutated" promise held
  only for the top-level dict — `_runtime` and message dicts were shared, so
  the next run's live mutations rewrote the completed run's stored record
  (in-memory stores). Now copied one level; the old tautological pin
  (`is old or == old`) replaced with a mutate-and-check pin. Also: handoff
  notes are timestamped (MemAct time-range archival skips timestamp-less
  content), `estimated_tokens_used` resets per task, and a legacy
  scratchpad-only `max_iterations` budget carries instead of silently
  widening.
- **Docs promised the pre-A1 contract (P2)**: README, faq, getting-started,
  and agents.md all stated fenced blocks execute unconditionally; corrected
  to the capability-conditional contract, plus a note that hosts swapping a
  run's model after start should pin `codeact_fenced_fallback` explicitly.
- **Parse emit argument aliasing (P3)**: all three adapters' parse emits
  shared the LIVE arguments dict with the pending tool call — a listener
  mutating its payload corrupted the actual call. Copied per emit.

### Added/Changed (2026-07-14, batch 4 — gated 0030/0028 remainder promoted on operator green light)
- **A1 capability-conditional CodeAct fence (0014 scoped piece)**: the
  `codeact_fenced_fallback` default is now `not supports_native_tools` (the
  bit runtime seeds per run from model capabilities); the system prompt
  teaches the ```python fence ONLY when the parser will extract it — one
  resolver (`CodeActLogic.fenced_fallback_enabled`) drives both surfaces so
  teaching and extraction can never disagree (teaching a disabled channel
  manufactures dead replies; a native-tools model taught the fence competes
  with the trained tool_calls channel — the Mnemosyne zero-tool class).
  Explicit flag wins both ways; absent capability bit fails safe (fence ON).
- **`read_skill` builtin (schema-only)**: progressive disclosure for attached
  skills — the host attaches an index via `skills_block` and maps
  `read_skill` in its tool executor to its shelf (the open_attachment
  precedent; execution is host-owned). Deliberately in NO default tool list
  (either half alone manufactures dead calls); contract in
  docs/skills-attachment.md.
- **Shared side-effect classifier** (`generation_params.is_side_effect_tool`):
  curated names + `mcp::` prefix + `ToolDefinition.tags` — origin-READY (the
  day core tags MCP/mutating tools at registration, the classifier lights up
  with zero changes here). ReAct's repeat guard rewired onto it.
- **`reset_react_task`** (`agents/task_reset.py`): pure vars transform for
  the work-door queued shape — per-task state resets (scratchpad, _temp,
  current_iteration), door-owned state persists (_runtime grants/slots/
  substrate, the budget), messages carry by default with an optional durable
  `task_handoff` note; input dict never mutated. v1 stays one-run-per-task —
  the door applies this BETWEEN runs, never mid-run.
- **0028 contract wave executed**: (1) budget exhaustion emits ONE turn_end —
  ReAct's conclusion node announces `max_iterations_reached` once at first
  entry and fires `max_iterations` once at the completion branch (was 2-3
  turn_ends per exhaustion from node re-entry); (2) `review_skipped` resets
  at the ask_user turn boundary (stale prior-turn #FALLBACKs polluted the
  next turn's report/output; ledger keeps history); (3) parse payload COMMON
  CORE across all three loops — `has_tool_calls` + `tool_calls` +
  `content_preview` (≤200) guaranteed everywhere, loop extras additive
  (CodeAct's preview bound unified 100→200).
- 8 new pins (`tests/test_gated_remainder_wave_2026_07_14.py`); inventory +
  hooks.md updated (new `max_iterations_reached` step declared). Suite 202
  passed / 2 skipped.

### Fixed/Changed (2026-07-13, batch 3 — adversary hardening of batches 1+2; two fresh fable5 reviews)
- **P0 un-break (regression adversary): deprecation shims instead of TypeErrors**
  for the removed dead knobs — the hard removal broke two SHIPPED abstractcode
  call sites (`workflow_agent.py:520` passes `max_tokens=None` to `ReActLogic`;
  `react_shell.py` constructs `MemActAgent` with plan/review kwargs). Legacy
  no-op spellings (None / -1 / defaults) construct silently; meaningful values
  warn `DeprecationWarning` naming the migration; removal next release.
- **P0 doc truth (design adversary): the `num_ctx` remedy was a dead knob** —
  `llm_kwargs={"num_ctx": ...}` is not forwarded by the stack (verified: zero
  matches in core/runtime; the Ollama options builder sends only
  temperature/num_predict/top_p/top_k/seed). All four docs now name the
  server-side remedies and the house rule per the operator's ADR: MAXIMUM
  available context unless explicitly chosen otherwise (a fixed lower number
  is the same hidden ceiling relocated); the per-request passthrough gap is
  flagged to core.
- **Outcome vocabulary unified**: `complete_output.outcome` now uses the
  CANONICAL turn_end vocabulary (`final_answer` | `iteration_budget`) instead
  of node names — the same enum was spelled two ways with zero consumers to
  migrate; raw `done`/`max_iterations` emits mirror it. `review_skipped` is an
  always-present bool in all three loops (CodeAct's detail list moved to
  `review_skipped_details`).
- **CodeAct split-brain fix**: plan and review payloads now carry the per-run
  `_runtime.provider/model` override like reason — a routed run planned and
  verified on the runtime-default model (possibly unloaded → every verifier
  round failed-and-absorbed).
- **ReAct review call honors the explicit output cap** (it was the one
  unbounded call type on the now-default review path).
- **Defaults refuse cross-provider guesses**: provider-without-model on a
  foreign provider now raises naming the fix (was: warned and shipped a
  wrong-shaped model tag — contradicting its own docstring); the packaged
  fallback pair is core's own Ollama literal (`qwen3:4b-instruct-2507-q4_K_M`
  — instruct, not the bare thinking-default tag), so the stack has ONE
  packaged pair. Warnings also log (server hosts swallow stderr warnings).
- **One prompt-slot composer**: slot order/headers moved to
  `generation_params.PROMPT_SLOTS` — the composition had been hand-copied into
  three adapters (the exact divergence class the system_prompt_extra fix paid
  for). Facades now expose the slots: `start(..., skills_block=...,
  system_prompt_extra=...)`.
- **Palette self-description + skew loudness**: granted substrate names (+
  host descriptions) render into the stable prefix (a grant the model cannot
  see manufactures dead guesses; wire ids deliberately not rendered); unknown
  profile keys emit `#FALLBACK` `delegate_agent_substrate_skew`;
  malformed-vs-unknown errors are distinct. Delegate inheritance widened to
  `thinking`/`max_output_tokens`/`tool_prompt_examples`.
- **Founding-publication rename**: CodeAct's `parse_retry_empty_response` →
  `parse_retry_empty` (one semantic, one name, zero consumers existed);
  `emit_inventory.RENAMED_STEPS` is the migration record.
- **`context_warning` second latch + ADR framing**: a second one-shot fires if
  usage crosses the ceiling itself; both payloads state explicitly that this
  is observability only — nothing trims or gates (no-silent-truncation ADR);
  honest Ollama scope documented (server-reported usage plateaus below the
  threshold on an already-truncating server). Factories now accept
  `max_output_tokens` (docs had documented it there; it raised TypeError).
- `lmstudio_tool_eval.py` pins `review_mode=False` (measurement
  comparability); hooks.md mermaid shows the review→act loop-back; 7 new pins
  in `tests/test_adversary_hardening_2026_07_13.py`. Suite 194 passed / 2
  skipped.
- **Facade ceiling literal DELETED** (operator ruling, "overengineering"):
  CodeAct/MemAct carried a last-resort `32768` accounting ceiling that could
  only fire when the runtime lookup failed — fabricating a warning threshold
  unrelated to any real window exactly when nothing was known (the runtime
  already defaults the ceiling twice: config → model registry →
  DEFAULT_MAX_TOKENS, plus read-site defaults). Missing ceiling now means the
  `context_warning` stays silent — absence of signal, never an invented one.

### Added/Changed (2026-07-13, batch 2 — 0030 residue promoted on operator green light)
- **`_runtime.skills_block` named slot**: skills attachments compose at a fixed
  position BEFORE `system_prompt_extra` in all three loops — a delegate child's
  sub-agent directive (which overwrites the extra) can no longer erase a
  host-attached skills block. Contract documented in
  `docs/skills-attachment.md` (slots must be byte-stable per run — they live in
  the provider cache prefix).
- **Delegate substrate palette**: `delegate_agent` gains an optional
  `substrate` arg resolved against host-granted `_runtime.delegate_substrates`
  (`{name: {provider, model}}`). Unknown names fail as loud TOOL errors naming
  what IS available; raw provider/model strings are structurally impossible
  (model-chosen substrates would be self-escalation); the palette never
  propagates to grandchildren. New `delegate_agent_substrate` emit.
- **MCP repeat-guard coverage**: `mcp::`-prefixed tools are treated as
  side-effectful by ReAct's duplicate-batch guard (deny-safe: the guard only
  skips re-executing an identical already-succeeded batch), and both batching
  prompt lines now name mcp:: tools.
- **Factory default resolution (B-F8)**: `create_react_agent` /
  `create_codeact_agent` / `create_memact_agent` now take
  `provider=None/model=None` and resolve from AbstractCore config global
  defaults (`abstractcore --config`); the packaged fallback pair (now
  `ollama`/`qwen3:4b`, was the indefensible 1.7B) applies only with a loud
  `#FALLBACK` UserWarning. Explicit args win untouched; config halves never
  cross providers.
- **Output-cap honesty (B-F9)**: all three facades accept `max_output_tokens`
  (the honest name). ReAct's explicit cap now SURVIVES init_node's
  full-context nulling (rides `_runtime`; `ReactAgent(max_tokens=4096)` was
  silently dropped — 0029 #9 fixed). Sibling facades wire it to
  `_limits.max_output_tokens`; sibling `max_tokens` stays the accounting
  ceiling and is documented as such. The dead accept-and-ignore
  `max_history_messages`/`max_tokens` params were REMOVED from
  `ReActLogic`/`CodeActLogic`/`MemActLogic` constructors (TypeError on use;
  0013 precedent).
- **Conclusion cache honesty (B-F7)**: ReAct's max-iterations conclusion
  directive + rendered scratchpad moved from the SYSTEM prompt (one guaranteed
  full re-prefill per budget-exhausted run) to the volatile trailing message.
- **Emit-step inventory (B-F12)**: `adapters/emit_inventory.py` declares the
  full per-adapter raw step-name sets; a drift test diffs them against actual
  `emit(` call sites both directions. `docs/hooks.md` now carries the
  stability contract (canonical names frozen; raw names
  stable-with-announced-renames; error prose is NOT a contract).
- **Unattended recipe**: `abstractagent.agents.unattended` packages the two
  moves (exclude `ask_user` + the no-questions directive) hosts kept
  rediscovering from adapter source.
- **`update_limits` docstring honesty (B-F11)**: all nine runtime-accepted
  keys listed; ReAct's no-op on mid-run `max_history_messages` stated.
- 9 more pins (`tests/test_optimal_wave_batch2_2026_07_13.py`,
  `tests/test_emit_inventory_drift.py`); conclusion-move contract re-pinned in
  the max-iterations test double.

### Changed/Fixed (2026-07-13, operator-directed improvement wave — two fable5 adversaries)
- **Sibling prompt-cache fix (0212 propagated)**: CodeAct and MemAct carried
  `Iteration: N/M` as the FIRST LINE of the system prompt — busting the
  provider prefix cache on every cycle (full re-prefill per call on local
  servers; ReAct was fixed in 0212, the siblings never were). Loop position
  (and CodeAct's live plan text in plan_mode) now rides a trailing volatile
  message with ReAct's adjacency guard; sibling system prompts are pinned
  byte-stable across iterations.
- **`system_prompt_extra` composed in CodeAct/MemAct**: both adapters WROTE
  the "you are a delegated sub-agent" directive into their delegate children
  but never READ the key — the directive was silently dropped on every
  sibling delegation, and hosts had no system-prompt append channel (skills
  blocks, unattended directives). Both now compose it at a fixed post-base
  position (reason + plan/review + finalize), mirroring ReAct.
- **Delegate children inherit the parent's effective substrate + sampling**:
  runtime seeds child `_runtime.provider/model` from CONFIG via setdefault,
  so a per-run substrate override silently reverted in delegated children
  (and temperature/seed reset to defaults mid-tree). All three delegate
  branches now copy provider/model/temperature/seed from the parent's
  `_runtime` explicitly.
- **CodeAct honors `_runtime.provider/model`**: the per-run routing channel
  (gateway pattern) reached ReAct and MemAct payloads but was silently
  ignored by CodeAct.
- **Machine-readable terminal `outcome`**: `complete_output` now carries
  `outcome: "done" | "max_iterations"` in all three loops (plus
  `review_skipped`/`finalize_skipped` booleans) — a work door deciding
  complete-vs-reschedule no longer parses prose. Sibling max-iterations
  terminals also stop raising KeyError on content-less messages.
- **`call_id` on observe emits (all three adapters)**: fleet controllers
  (abstractcode serve) correlate tool_call → approval → tool_result by
  call_id; the observe payload now carries it (additive; empty string when
  absent). Enables code-seat's shipped forwarder (c1613).
- **ReactAgent `review_mode` defaults ON**: the 2026-07-09 opt-in flip named
  its own re-flip condition ("once review failures degrade to
  accept-with-#FALLBACK") — 0027 shipped exactly that, CodeAct already
  defaulted True, and abstractcode re-flipped its CLI on the same condition.
  Three stale/contradictory comments corrected (agents/react.py ×2,
  react_runtime.py) and getting-started.md's false "review not applied"
  claim fixed. Price: one verifier call per candidate final answer.
- **One-shot `context_warning` emit**: the runtime accounts real token usage
  into `_limits.estimated_tokens_used` and the facades seed
  `warn_tokens_pct` — but nothing ever checked it; runs sailed silently past
  the model window (provider 400, or Ollama's silent head-truncation). The
  reason nodes now emit a single `#FALLBACK` `context_warning` when usage
  crosses the threshold; faq.md documents the overflow failure mode and the
  `num_ctx` remedy.
- 9 new pins in `tests/test_improvement_wave_2026_07_13.py`. Residue that
  needs design or green-light recorded in backlog 0030 (skills-block seam
  contract, MCP side-effect guard + wire-name flag to core, delegate
  substrate palette, work-door task reset + unattended recipe, emit
  inventory + drift test, default-model resolution, max_tokens naming).

### Fixed (2026-07-13, whole-package fable5 audit — P0 + second fix wave)
- **P0 — repeated identical tool batches silently replayed stale results**:
  the runtime keys effects on (run_id, node_id, payload) with call_ids
  STRIPPED from the idempotency hash and scans the WHOLE ledger for prior
  completed results — so re-reading a file after editing it, re-running the
  test suite after a fix, or re-running the same fenced code block (the point
  of CodeAct) served the STALE pre-change result from the ledger without
  executing. All four TOOL_CALLS issue sites (ReAct/CodeAct/MemAct `act`,
  CodeAct `execute_code`) now stamp a persisted monotonic `act_seq` into the
  payload: each genuine issuance gets a fresh idempotency key while
  crash-replay dedup still holds (the counter persists in the same save as
  the effect's ledger record). Kernel pin: identical batch across two cycles
  executes TWICE with distinct observations. The runtime half (whether the
  policy itself should re-include call ids) is flagged to the runtime seat.
- **MemAct finalize failure destroyed the held answer (P1, the c1128 class)**:
  the mandatory finalize call carried no failure containment and
  `finalize_parse` discarded `_temp.draft_answer` on any parse failure —
  a failed/unparseable finalize turned a real answer into "No answer
  provided" (or a dead run). Finalize now opts into `_absorb_failure`; the
  parse falls back to the draft with a loud `#FALLBACK` marker
  (`finalize_skipped` / `finalize_used_draft`), envelope application skipped.
- **CodeAct verifier "re-ask reviewer" branch deleted (P1)**: the re-review
  payload was byte-identical, so the idempotency layer REPLAYED the first
  verdict (guaranteed no-op) — and the reviewer-directed nudge ("Return JSON
  only") then leaked into the MAIN model's durable guidance at the next
  reason drain. Mirrors ReAct's earlier resolution: unactionable verdicts
  route straight to reason with the verifier's next_prompt as guidance.
- **ReAct cycle observations were overwritten per observe pass (P1)**:
  `observe` runs multiple times per iteration when the queue splits; the
  durable cycle record kept only the LAST batch's observations, thinning the
  report/conclusion channels and the repeat-guard's evidence. Now extends.
- **Dead structured-answer guard (P2)**: a double-escaped `\\S` in a raw
  string made the "don't retry heading-shaped final answers" branch
  unmatchable since birth; heading-formatted answers with intent verbs burned
  bounded retries. Regex fixed.

### Fixed (2026-07-13, production-readiness wave — fable5 correctness audit)
- **Delegate budget arg-coercion (P1)**: `delegate_agent`'s `max_iterations`
  tool argument arriving as a string-float (`"8.5"` — the known tool-call
  arg-coercion class) raised inside `int()` and silently WIDENED an explicit
  narrow child budget to the >=20 inheritance. New `coerce_iterations()`
  (generation_params) tolerates int/float/string-numeral shapes, refuses
  booleans (True would have made a 1-iteration child), and unparseable values
  fall back to inheritance with a loud `delegate_agent_budget_fallback`
  `#FALLBACK` emit. Applied in all three adapters + the shared
  `resolve_max_iterations` (where a string-float `_limits` budget made the
  loop's own bound fall open to the default). 5 new matrix rows per sibling.
- **CodeAct empty-retry streak never reset on success (P1)**: the
  `empty_response_retry_count` counts CONSECUTIVE empty replies but only
  reset on exhaustion/default-final — two recovered empties early in a run
  made every LATER single empty reply give up immediately ("can't proceed")
  despite remaining budget. Now resets on every successful parse exit
  (tool_calls / FINAL / fenced / default). Pinned.
- **CodeAct review round retry-allowance staleness (P2)**: `review_retry_count`
  was only reset on the give-up path, silently starving later review rounds of
  their one unactionable-retry; now reset when each round starts.
- **CodeAct verifier-skip output loudness (P2)**: the `#FALLBACK`
  review-skipped marker lived only in the emit lane and stored vars (CodeAct's
  output has no report); `complete_output` now carries `review_skipped` —
  loudness parity with ReAct's report line.
- **ReAct conclusion-boundary drain now emits `inbox_drained` (P2)**: durable
  guidance consumed by the max-iterations conclusion fired no `message_drained`
  listen point — capture hosts waiting for consumption confirmation never saw
  it. Recorded, not yet acted (see backlog): the `turn_end` multi-emit on the
  self-re-entering max_iterations node (hook taxonomy), and `review_skipped`
  report accumulation across visit turns (arguably deliberate durability;
  needs per-turn tagging if it confuses).

### Changed (2026-07-13, phase-machine lane audit)
- **Phase vocabulary alignment in test prose**: the grant-boundary test's
  docstrings and test names said `tasked` (runtime's dies-before-release
  migration alias) where the ruled phase key is `work`. Code was already
  canonical (`_PHASE_WORK` derived from runtime's `PHASES` tuple); only the
  words drifted. Renamed (`test_narrow_work_policy_file_...`). Found by the
  operator-directed fable5 lane audit against the ruled entity-phase state
  machine (decision v3: visit/work/personal/sleep, one active, restore-previous
  at visit close); full audit verdict: the adapter is phase-blind by
  construction and matches the ruled machine — no P0 findings.

### Fixed (2026-07-13, defect batch 0010/0012/0013)
- **CodeAct fenced-block execution no longer fires on non-action replies
  (backlog 0010 / A4)**: the `FINAL:` final-answer check now PRECEDES fenced
  code extraction — a reply like `FINAL: here's an example: ```python ...`
  is a final answer whose illustrative block used to EXECUTE (unintended
  execution on a turn with no action intent). Fenced extraction is now also
  flag-gated: `_runtime.codeact_fenced_fallback` (default **true** — the
  shipped CodeAct prompt actively teaches the fence as the prompted-model
  action lane; flipping the default belongs to proposal A1's native-primary
  wave, not a defect fix). Explicit `false` disables extraction entirely for
  native-tool-call setups.
- **delegate_agent child budget honesty in CodeAct/MemAct (backlog 0012 /
  C3)**: both siblings hardcoded the child `_limits.max_iterations` to 10 and
  ignored the `max_iterations` tool argument, while the SHARED tool schema
  documents ReAct's ruled resolution (explicit arg wins; otherwise inherit
  the parent's budget with floor 20 — the agency-caps ruling). The schema
  lied for two of three loops; both now apply ReAct's resolution verbatim.
- **MemAct dead knobs removed (backlog 0013 / C4)**: `MemActAgent` accepted
  `plan_mode`/`review_mode`/`review_max_rounds` and read NONE of them (no
  plan or review nodes exist in the MemAct workflow — the silent fall-open
  class). The parameters are removed from the constructor and `start()`;
  passing them now raises `TypeError`. A MemAct verifier stays a deliberate
  non-goal (keep-experimental). `docs/agents.md` aligned: the stale "ReAct
  does not apply review_mode" claim corrected (honored since 0217, with 0027
  failure containment), MemAct section documents the removal.
- All three pinned in `tests/test_sibling_loop_defects_batch.py` (FINAL+fence
  never executes / default fence path unchanged / flag-off disables; the
  4-case budget matrix per sibling; TypeError on removed knobs).

### Fixed (2026-07-13)
- **Verifier failure degrades to accept-with-#FALLBACK (backlog 0027, c1128
  incident class)**: with `review_mode` on, a terminally failed verifier
  LLM call (structured validation, provider error, anything) used to FAIL a
  run whose `_temp.final_answer` already held a valid answer — a verification
  aid killing a succeeded run, plus wasted retries on a deterministic failure.
  Both verifier forks (ReAct `review_node` + CodeAct's) now opt into the
  runtime's failure absorption (`payload._absorb_failure`, the fdf01e0 rule
  class): the terminal failure lands as `{"ok": false, "absorbed_failure":
  <error>}` at the result key, and `review_parse` contains it — the held
  answer is ACCEPTED and the run completes, loudly: `scratchpad.review_skipped`
  marker (`#FALLBACK` + reason), a `review: #FALLBACK skipped (...)` line in
  the final report, and a dedicated `review_skipped` emit
  (`accepted_held_answer: true`). CodeAct specifically no longer falls through
  into the unactionable-retry path (which re-issued the failing call and then
  re-entered `reason`). Runtimes without the absorption mechanism ignore the
  payload key — behavior there is exactly as before. The review SUCCESS path
  is byte-unchanged. This is the agent half of the c1128 review re-default
  condition (core's example-generator fix is the other half, shipped c1201);
  the `review_mode` default stays opt-in in this adapter — the re-default
  decision is the consuming CLI's (abstractcode c1139). Pinned in
  `tests/test_review_failure_containment.py` (kernel failure run, success
  guard, CodeAct containment).

### Docs
- **Backlog system normalized (codex backlog skill)**: `docs/backlog/` now has
  the full lifecycle layout (overview, planned/proposed/completed/deprecated,
  recurrent hygiene tasks); the legacy date-named item renamed to
  `0001_agent_gateway_install_boundary.md`. The maintainer-endorsed loops
  meta-audit proposal set (IDs A1–A4, B1–B4, C1–C5, D1–D4 + hooks-wave
  follow-ups) recorded as the `loops_improvement` track: defects as planned
  items 0010–0013 (fenced-block intent bug, sibling transcript repair,
  delegate-budget honesty, MemAct dead knobs), features/decisions as proposed
  items 0014–0026 with explicit promotion criteria (C5 fold = maintainer
  ruling; A2 kernel = runtime executor lane per the agreed seam).

### Added (cont.)
- **Streaming passthrough (`_runtime.stream`)**: `runtime_llm_params` passes
  `stream: true` into LLM_CALL params when `_runtime.stream is True` — the
  last link for end-to-end CLI token streaming (runtime's `on_token`
  callback, code seat c1007). STRICT bool only: truthy strings ("true", "1")
  are the tool-args coercion class and never enable streaming; absent/False
  emits no key. Pinned in `test_generation_params_media_policies.py`.

### Fixed
- **Volatile loop tail no longer defeats local prompt caches (B1 pair, agent
  half)**: the ReAct per-cycle trailing message (`[loop] iteration N of M.` +
  plan render) was fingerprinted by runtime's llm_client, forcing a full
  local-cache re-prefill EVERY cycle (code seat's adversary c971; my
  prefix-reuse metric had pinned the symptom as the adjacency-guard trade).
  The separate trailing tail now carries the structural marker
  `volatile: true` (top-level, runtime-confirmed spelling c986) — runtime
  excludes flagged messages from the fingerprint sequence and strips the key
  before any provider SDK sees it. The merged first-turn leg cannot carry the
  flag (it holds the real task) and dies with runtime's B3 boundary-merge.
  Pinned in `test_react_prompt_prefix_stability.py` + a cross-package smoke
  against runtime's real `_strip_volatile_markers`.

### Added
- **First-class loop hooks (listen + steer + capture; maintainer directive
  2026-07-12)**: new shared `adapters/loop_hooks.py` (`LoopHooks`, `HookEvent`,
  `DEFAULT_EVENT_MAP`, root-exported) wired through all three adapters and
  agent constructors (`hooks=` beside `on_step`). LISTEN: every emit point
  dispatches structured `HookEvent`s under canonical names (cycle_start /
  tool_proposed / tool_executed / turn_end+outcome / message_drained; unmapped
  steps pass through raw — total coverage); handlers receive a COPY of the
  payload so mutation cannot corrupt the loop. STEER: handler returns
  (`"text"` or `{"inject": ...}`) queue PER RUN and fold into the durable
  `_runtime.inbox` at the reason boundary — one drain point shared with
  inject_guidance; at-most-once from host memory (documented); undelivered
  steering is discarded LOUDLY at terminals (`hook_steer_discarded`), never
  leaked into a later run (pinned: two sequential runs on one workflow
  product). CAPTURE: handler exceptions contained + surfaced as `hook_error`
  to BOTH the flat `on_step` stream and the handlers themselves (hooks-only
  hosts observe their own failures — pinned); unsupported action shapes are
  loud. Events carry run_id + agent + iteration; `done` emits carry
  `handed_off` so visit-composition turn_ends are distinguishable from run
  completion. Fleet seam: `message_drained` fires when the reason boundary
  consumes delivered guidance — the in-loop "message received" hook.
  Adversary-reviewed (2 P1s fixed pre-ship: cross-run queue leakage via the
  shared workflow product; follow-up invisibility without on_step). Handler
  TIME BUDGET (plan H-row, sync-on-tick DoS): calls over `slow_budget_s`
  (default 1s) earn loud `hook_slow` strikes; `max_slow_strikes` (default 3)
  DISABLES the handler for the instance — main dispatches and notifications
  both stop; fast handlers unaffected. Tests: `tests/test_loop_hooks.py` (12).

## Unreleased (2026-07-11)

### Fixed
- **Act-only rerun refs no longer downgrade to record text (R3 P0, agent
  half)**: runtime's tool+args generalization (64398ff) added a second
  addressable ref shape — `{tool, args}` re-run refs (diary_list; the listing
  re-executes fresh at send time) next to `{tool, entry_id}` entry refs.
  `_act_only_frame_is_dereferenceable` accepted only `entry_id`, so a
  well-authored diary_list frame rendered as inert record text and never
  reached runtime's resolver. Now shape-based: non-empty `entry_id` OR an
  `args` dict qualifies for the ref form; reference-free frames (failures,
  suppressions) keep rendering as inert record text. No tool-name list is
  copied into this adapter (runtime's dispatch owns that; unknown refs
  tombstone loudly since the wedge amendment). Interop test runs the rendered
  bytes through runtime's shipped dereference.

### Changed (maintainer rulings c726/c786)
- **`max_iterations` default 25 → 20 everywhere**: the maintainer ruled 20 as
  THE framework default ("i never said 50", c726; runtime's agent-node pin and
  the published basic-agent bundle both carry 20). This package's 25 was a
  third value in the ecosystem — the copied-default drift class the room
  keeps closing. Flipped: the three agent constructors, the three adapters'
  scratchpad seeds, and `resolve_max_iterations(default=)`. Explicit
  workflow/host values stay authoritative (the "workflow decides" amendment);
  the ruled 100 calls/turn failsafe ceiling is enforced upstream
  (runtime compiler / gateway config), never silently in the loop.
- **Phase-grant boundary tests are spelling-independent**: the ruled
  visit/work/personal/sleep rename (c786) must not redden this suite — the
  boundary tests now derive phase keys from runtime's `PHASES` tuple
  (semantic-order unpack) instead of hardcoding "tasked"/"own_time" era
  spellings. The adapter itself is phase-blind (consumes resolved grants,
  never phase words).

### Added
- **Phase-grant boundary conformance pins (config-object build phase)**:
  `tests/test_react_phase_grant_boundary.py` runs runtime's REAL
  `resolve_tool_grant`/`write_policy_file` against a temp home and pins the
  adapter's consumption contract on BOTH channels (run vars = the plan's
  destination; factory param = the shipped door's channel): default grants
  offer exactly the resolver's set per ruled phase (registry imported from
  runtime's constants, never copied); sleep's default arrives strictly
  narrower than visit and diary-free; a narrow `tasked` policy file arrives
  narrow (N8 consumer half) incl. the maximal-narrow `tasked: []` -> deny-all;
  the TOOL_CALLS execution payload carries the same allowlist as the offer;
  an unknown-name grant prunes loudly. Adversary-reviewed (channel-honesty
  docstring rewrite, imported-constants fixture, execution-payload pin added).
- **Visible allowlist pruning (`_runtime.allowlist_pruned`)**: grant entries
  that do not resolve to a registered tool were dropped silently (deny-safe but
  invisible). All three adapters (ReAct/CodeAct/MemAct, via the shared
  `adapters/tool_allowlist.py`) now record a durable note
  `{"dropped", "requested"[, "invalid"]}` and emit one `allowlist_pruned` step
  event per prune event — including the worst case where a grant of non-name
  garbage (e.g. `[None]`) normalizes to a FULL deny. Deny-safety is unchanged;
  the refusal is now visible to doors/operators (works-or-loud, the
  config-object conformance boundary). Tests:
  `test_react_entity_dress_conformance.py` (2 new pins incl. emit-once).

### Fixed (adversarial audit findings, config-object task)
- **`thinking` freeform strings no longer hard-fail at the provider boundary
  (adversary P2-1)**: `normalize_thinking` forwarded any non-empty string, but
  core RAISES ValueError for non-enum values — the same collision class as the
  prompt_cache_binding incident. Unknown strings are now DROPPED (None = do not
  send, matching `normalize_seed`'s policy); known values emit in canonical
  form. A drift-pin test runs every emitted value through core's real
  normalizer so enum drift is named, never silent.
- **Explicit `max_iterations=0` no longer falls open to 25 (adversary P2-2)**:
  the inline `limits.get("max_iterations", 0) or scratchpad... or 25` pattern
  (8 sites across react/codeact/memact adapters) treated an explicit 0 as
  unspecified. New shared `resolve_max_iterations` resolves presence-first and
  clamps explicit sub-floor values to 1 — an explicit narrow budget never
  silently widens (the agency-caps invariant).

### Changed
- **delegate_agent child iteration budget (agency-caps ruling 2026-07-11)**: the
  delegated child's `_limits.max_iterations` was hardcoded to 10 — a fear-shaped
  default under the ruling ("default caps are 20; below-20 is operator choice").
  The child now inherits the PARENT's budget with 20 as the floor guard; an
  explicit `max_iterations` tool argument wins (added to the delegate_agent
  schema). The no-recursion allowlist strip (delegate_agent/ask_user removed
  from child allowlists) is unchanged — that is wait-safety, not a cap.
  Tests: `test_react_delegate_agent_tool.py` (4 new parametrized pins).

### Fixed
- **Prompt-cache vocabulary collision (live-proven, agency c509)**: `runtime_llm_params`
  now emits bare-string `_runtime.prompt_cache_binding` values as `prompt_cache_key`
  (core's best-effort per-session cache identity) and reserves `prompt_cache_binding`
  for dict shapes (core's strict durable-bloc artifact binding, whose validation
  unconditionally raises for bare strings). The string-under-strict-name shape failed
  100% of live entity-visit turns at the provider boundary the moment the door began
  forwarding params (turn-1 terminal failure). The guard applies at the output
  boundary, so explicit string overrides convert too. `_runtime.prompt_cache_key`
  is now also forwarded directly. Tests: `test_generation_params_media_policies.py`
  (4 new pins, incl. the exact door-stamped string shape).

All notable changes to `abstractagent` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (2026-07-10 — THE MERGE PROOF: this cycle inside runtime's real visit workflow)
- **`tests/test_react_visit_merge.py`**: consumes runtime's SHIPPED merge parameter
  (`build_visit_workflow(react_middle=...)`, a78185c — one owner for the visit graph
  in runtime's package; the middle arrives as DATA to keep the dependency arrow
  one-way). This suite is the CALLER the `ReactMiddle` contract names: it builds the
  middle from this package's public API (`create_react_workflow(final_next_node=
  HARVEST_NODE)` + `reset_react_turn`) and pins the composed graph over a REAL home +
  per-entity runtime. Proven live: mid-loop diary election (iteration 1, beside a tool
  call) captured at the result boundary — elected words rest ONLY in the book (run
  vars + ledger grepped clean); episode formed with `visit_id` + stamped participants
  through runtime's nodes byte-unchanged; identity access counts stay 0 (D2); prelude
  head byte-identical across iterations; restart-mid-visit (criterion 5) holds under
  the merge. The seam is tested from BOTH directions: runtime pins it with a
  contract-faithful stub middle (no adapter import); this suite pins it with the real
  cycle. (Initial version carried local BRIDGE/HARVEST seam nodes as the proposal
  proof; deleted same-day when runtime shipped the parameter.)

### Added (2026-07-10 — visit-workflow composition knob, frozen spec a2a 0013 §A / 0014 draft)
- **`create_react_workflow(final_next_node=...)`**: when set, `done`/`max_iterations`
  finish the TURN instead of the run — they persist the final answer to the durable
  transcript exactly as today (never-strip obligation intact), stash the output dict at
  `_temp.react_output`, and hand off to the named seam node. This is how the entity
  visit TURN chain (RECALL → this adapter's reason/act/observe → ELECTIONS → COMMIT →
  FORM → ANSWER → PARK) embeds the loop: the embedding workflow merges the adapter's
  nodes with its seam nodes. Default None: behavior unchanged.
- **`reset_react_turn(run_vars)`**: per-turn budget semantics for re-entry — resets the
  iteration counter, review count, and per-turn `_temp` carriers while leaving the
  durable life (transcript, `scratchpad.cycles`, plan) untouched. Re-entry goes through
  this helper then `reason`, never `init` (init seeds the task message; visit messages
  arrive from the PARK resume). Pinned through the real runtime with a miniature visit
  workflow (`tests/test_react_visit_composition.py`): handoff instead of completion,
  two-turn visit with fresh budget + durable transcript continuity + append-only prefix
  across the turn boundary, budget exhaustion handing off to the seam.
- **`_runtime.turn_id` + `_runtime.llm_payload_extras` → LLM_CALL payload pass-through**
  (G1 write direction, runtime b8b8c78/56e55c3): the entity runtime's result-boundary
  diary-election capture keys book writes on the payload's `turn_id` (elections without
  it fail loud) and threads word-free `anchor_record_ids`/`anchor_graph_ids` into the
  DIARY_WRITE it performs. The embedding workflow sets both per turn; `reason_node`
  passes them through (extras never overwrite existing payload keys). Absent = absent —
  non-visit runs byte-unchanged (pinned both ways).
- **`_temp.turn_captures` accumulation**: the entity runtime's wrapper returns
  `diary_entries` (word-free metadata) + `act_only_warnings` on EACH LLM result; with a
  multi-iteration loop a mid-loop election's capture would be lost when
  `_temp.llm_response` is overwritten next iteration. `parse_node` accumulates both
  lists across the turn at `_temp.turn_captures` for the embedding ELECT/FORM nodes;
  `reset_react_turn` clears it at turn boundaries. Keys are the entity runtime's
  declared result contract; absent = untouched (pinned incl. the mid-loop election case).

### Added (2026-07-10 — prefix-reuse measurement, A/B criterion 3 evidence tooling)
- **`abstractagent.metrics.prefix_reuse`** (`measure_prefix_reuse` / `prefix_reuse_report`):
  byte-level reuse measurement over captured LLM_CALL payloads, mirroring how provider
  prompt caches key requests (head = system_prompt + tools, reusable only when
  byte-identical; message lane = longest common prefix of byte-identical messages;
  a changed head invalidates everything after it). Deterministic JSON canonicalization;
  pure measurement, no policy. Feeds the fixture-home A/B's criterion 3 ("prefix reuse
  measured and reported — target is evidence, not a pass bar"). Pinned at unit level
  (known reuse shapes, head-mutation invalidation) AND over the real adapter loop,
  where it MEASURES the documented adjacency-guard trade precisely: iteration 2 reuses
  the head but zero messages (iteration 1's task message merged the volatile tail);
  from iteration 3 the transcript prefix reuses and grows
  (`tests/test_prefix_reuse_metric.py`).

### Added (2026-07-10 — entity-dress conformance pins, frozen spec a2a 0013 §4)
- **Entity-dress knobs pinned by test** (`tests/test_react_entity_dress_conformance.py`):
  `_runtime.system_prompt` (the prelude) fully replaces the ReAct persona and stays
  byte-identical across iterations (the visit's cached prefix holds);
  `_runtime.allowed_tools = []` is deny-by-default (zero tool specs reach the provider)
  and is distinct from the key being absent (full registry default) — the load-bearing
  distinction for tier-1 grants; `_limits.max_iterations` is the per-turn budget and
  the tool-free conclusion path runs on exhaustion. No adapter changes needed — the
  dress is configuration on the existing loop, now proven rather than asserted.

### Added (2026-07-10 — never-strip election conformance, frozen spec a2a 0013 §4 line 6)
- **Election fences survive the loop, pinned by test** (`tests/test_react_election_fence_conformance.py`):
  final answers carry ```fenced blocks byte-identical and in order; a fences-only reply
  is a VALID final answer (entity elections are reply content); the max-iterations
  conclusion strip removes tool-call markup spans only (election fences pass through).
- **Deferred-action followthrough heuristic is now fence-blind** (`react_runtime.py`):
  `_looks_like_deferred_action` evaluates the PROSE view with paired ```fenced blocks
  removed. First-person text inside a fence (e.g. a diary election "I will keep
  reading…") is quoted content, not an action commitment — previously it could trigger
  a followthrough retry that DISCARDED the reply and consumed its elections (the exact
  failure the seam spec's never-strip obligation forbids). The heuristic is unchanged
  for genuine prose action claims (pinned both ways). Conservative fence matching:
  only PAIRED fences are excluded; an unterminated trailing fence stays in the prose view.

### Added (2026-07-10 — entity visit seam, frozen spec a2a 0013 v2 §2)
- **Act-only tool observations render as `$act_only` references** (`react_runtime.py`):
  for tools carrying the `act_only` attribute (core's first-class ToolDefinition field;
  getattr-based and fail-closed by absence until core ships it), `observe_node` appends
  the ACT-FRAME REFERENCE — one exact JSON object with a lone `$act_only` top-level key —
  as the durable tool message content instead of rendered output. The runtime LLM_CALL
  handler dereferences the ref at the provider boundary (send time) into a wire copy;
  the words never rest in the durable transcript, scratchpad cycles, LLM payloads, or
  the emit/on_step lane (G1: "the book's words never rest outside the book").
  Handler-authored refs (`{"$act_only": {...}}`-shaped outputs) are honored regardless
  of local declarations (the effect handler is the enforcement authority); a declared
  act-only tool whose handler misbehaves gets loud suppression (`#FALLBACK` frame),
  never silent rendering, and failure diagnostics ride the error channel only.
  Serialization is deterministic (sorted keys) so durable bytes stay prefix-cache
  stable. Honest boundary found while building: the runtime kernel copies raw effect
  results into `_runtime.node_traces` before observe runs — keeping words out of the
  result channel entirely is the handler's contract (runtime seat's surface); the
  adapter pins every surface it owns (`tests/test_react_act_only_observation.py`).
- **Wedge guard: the lone-key ref shape is reserved for dereferenceable frames**
  (cross-package finding against runtime's shipped send-time dereference,
  `abstractruntime/identity/act_only.py`): runtime's LLM wrapper loudly FAILS a call
  on any unresolvable `$act_only` ref, and refs are durable — so an act-only frame
  that references nothing (suppression/failure records, list-style results without an
  `entry_id`) must NOT take the ref shape or it would fail every subsequent LLM call
  in the run. Such frames render as labeled non-ref record text
  (`[tool]: act-only record (no content at rest): {…}`) — still no words at rest,
  inert to the dereference pass. Interop pinned by running runtime's own
  `parse_act_only_ref`/`dereference_act_only_messages` over both shapes (7 tests total).

### Changed (2026-07-09 adversarial-audit wave — 5 critics over the agency-parity changes)
- **Loop-tail adjacency guard**: when the LLM payload already ends with a user message (first
  turn; post-ask_user turns), the volatile `[loop]`/plan/guidance tail now MERGES into that
  message instead of appending a second consecutive user message. Alternation-strict chat
  templates (Mistral/Gemma-class) reject `user,user` with a 400, and a separate trailing banner
  also displaced the runtime grounding envelope from the real task message. In tool-loop shape
  (payload ends with tool results) the tail stays a separate trailing message exactly as before.
  Trade-off (documented in the prefix-stability test): the iteration-1 task message carries the
  merged tail, so it is not prefix-reusable into iteration 2 — one small message, once per run;
  from iteration 2 onward the task message is pure and byte-stable.
- **Recovered-reasoning fallback stays UNBOUNDED (ADR-0026)**: a same-night attempt to cap the
  empty-content → `reasoning` transcript fallback at 1200 chars was REVERTED as a
  maintainer-caught ADR-0026 violation — the recovered reasoning is the model's own thought for
  the cycle and is model-facing context, never ours to slice (a marker does not make a lossy
  slice compliant; truncation is reserved for UI surfaces or explicit user opt-in). The
  pathological shape the cap pretended to solve (generation cut mid-reasoning by an output
  token limit) never reaches this path: `parse_node` retries `finish_reason in
  {"length","max_tokens"}` turns WITHOUT appending partial content, and ReAct's `init_node`
  disables output caps by default. Token growth from faithful reasoning retention is the
  designed fidelity trade, mitigated by prompt-prefix caching (0212) and explicit user-opted
  compaction — never by silent truncation.
- **`review_mode` default flipped to opt-in (False)**: the verifier works (live-proven catching
  real instruction violations) but its failure mode is uncontained — a verifier-side
  deterministic 400 fails a run that already holds a valid final answer, and verifier-forced tool
  calls bypass the parse-node duplicate-side-effect guard. Re-default to on once review failures
  degrade to accept-with-`#FALLBACK`. (Note: `review_mode=True` was dead config at HEAD; the
  0217 wiring made it live, which is what made the default matter.)
- **Orphan-repair honesty guard**: the `ask_user` payload repair no longer claims "handled
  interactively" for arbitrary unanswered tool calls — only interactive builtins get that label;
  any other unanswered id gets `[tool result missing (host error): <name>]` so genuine tool-result
  loss (the known truncation class) stays observable instead of being papered over. Orphan TOOL
  messages (no preceding assistant tool-calls run, e.g. after a history cut) are folded into an
  inert `[unpaired tool result]` user note, mirroring the native-OpenAI fold, instead of 400ing
  strict servers.

### Fixed
- **Multi-turn ask_user runs failed on native OpenAI (pre-existing at HEAD, found in live
  multi-turn verification 2026-07-09)**: `ask_user` resolves through an ASK_USER wait + a user
  message — never a tool message — so the assistant `tool_calls` turn stays unanswered in durable
  history and OpenAI 400s the NEXT request ("must be followed by tool messages responding to each
  tool_call_id"), failing the run after retries. `_sanitize_llm_messages` now repairs orphans at
  the payload boundary: any unanswered tool_call id gets a deterministic adjacent synthetic tool
  result ("[handled interactively; …]"); durable history is untouched and the synthetic text is
  stable so the 0212 cached prefix stays byte-identical. Live-verified (native OpenAI,
  gpt-5-mini): the ask_user probe that 400s on git HEAD completes on current code — artifact
  `docs/backlog/planned/agency-parity/evidence/ab_askuser_native_openai.json` (the earlier
  citation pointed at the OVH multi-turn artifact, which is a different provider's run —
  corrected per audit). OVH multi-turn evidence (session recall + wait/resume + hygiene) is in
  `evidence/ab_multiturn_result.json`; single-run existence demonstrations, not success-rate
  claims.

### Added
- ReAct verifier + planning (agency-parity 0217): the CodeAct-style verifier is now wired into the
  ReAct loop behind `_runtime.review_mode` (previously dead config on ReAct) — a final answer is
  re-checked and, if incomplete, the loop re-enters `act` with the verifier's `next_tool_calls`
  (synthesizing the preceding assistant tool-calls message so the tool results are never orphaned).
  New `update_plan` builtin tool persists a checklist to the scratchpad and renders it on the
  cache-stable message tail. Verification budget resets per-answer after tools run (mirrors CodeAct).

### Changed
- Prompt-prefix cache stability (agency-parity 0212): the per-iteration counter, scratchpad, and
  drained guidance are no longer baked into the (cached) system prompt — they ride a trailing
  ephemeral message so the request prefix stays byte-stable across iterations. Measured common
  prefix across a 3-iteration loop went from ~1% to ~93%.
- Context fidelity (agency-parity 0213): assistant tool-call transcript messages now retain the
  model's reasoning `content` instead of `content=""`, and the redundant scratchpad copy is no
  longer injected into the system prompt (observations already live in the transcript). Remaining
  bounded previews carry the ADR-0026 `#[WARNING:TRUNCATION]` tag + marker.

## [0.3.12] - 2026-06-14

### Changed
- Raised Core and Runtime dependency floors to `abstractcore>=2.13.38` and `AbstractRuntime>=0.4.29` across base, Apple, and GPU install profiles.
- Agent generation-param normalization now preserves explicit Core `thinking` controls alongside temperature, seed, media policy, and prompt-cache binding defaults.

## [0.3.11] - 2026-06-03

### Changed
- Raised Core and Runtime dependency floors to `abstractcore>=2.13.32` and `AbstractRuntime>=0.4.27` across base, Apple, and GPU install profiles.

## [0.3.10] - 2026-05-31

### Changed
- Collapsed the hardware profile surface to `abstractagent[apple]` and `abstractagent[gpu]`, matching Runtime's base/Apple/GPU install policy. The Apple/GPU profiles now cascade to AbstractCore's full local-engine aggregates through Runtime's `apple` and `gpu` profiles.
- Raised Core and Runtime dependency floors to `abstractcore>=2.13.31` and `AbstractRuntime>=0.4.26` across the base and hardware profile extras.

## [0.3.9] - 2026-05-29

### Changed

- Raised Core and Runtime dependency floors to `abstractcore>=2.13.30` and `AbstractRuntime>=0.4.25` across the base and hardware profile extras.

## [0.3.8] - 2026-05-26

### Changed

- Raised Core and Runtime dependency floors to `abstractcore>=2.13.28` and `AbstractRuntime>=0.4.23` across the base and hardware profile extras.

### Fixed

- Media generation parameter handling now keeps `prompt_cache_binding` scoped to text generation so generated-media calls do not receive brittle cache-only arguments.

## [0.3.7] - 2026-05-09

### Changed

- Re-release to pick up abstractruntime>=0.4.9 on PyPI (previous release CI ran before CDN propagation).

## [0.3.6] - 2026-05-09

### Changed

- Raised Runtime dependency floors to `AbstractRuntime>=0.4.9` so Agent
  installs inherit Runtime's base AbstractMemory contract for KG-aware
  workflows.

## [0.3.5] - 2026-05-09

### Changed

- Raised base and hardware-profile dependency floors to `abstractcore>=2.13.12`
  and `AbstractRuntime>=0.4.8` after the Core/Runtime install-profile alignment.
- Added packaging regression coverage for the base dependencies and
  `apple`/`gpu`/`all-apple`/`all-gpu` profile cascades.

## [0.3.4] - 2026-05-08

### Fixed

- Set AbstractCore and AbstractRuntime dependency floors to currently published
  PyPI versions so CI, editable installs, and trusted-publishing releases can
  resolve dependencies in clean environments.

## [0.3.3] - 2026-05-08

### Added

- Added GitHub Actions CI for Python 3.10 through 3.12 with pytest and package
  build checks.
- Added a trusted-publishing release workflow for tagged or manually dispatched
  releases, including version/changelog validation, distribution artifacts,
  PyPI publication, and GitHub Release creation.
- Added an AbstractAgent GitHub bug report template.
- Added a `test` optional dependency extra for CI and release validation.

## [0.3.2] - 2026-05-08

### Changed

- Added native install-profile cascade extras:
  `abstractagent[apple]`, `abstractagent[gpu]`,
  `abstractagent[all-apple]`, and `abstractagent[all-gpu]`.
- Raised optional Core/Runtime profile floors to `abstractcore>=2.13.12` and
  `AbstractRuntime>=0.4.8` so Agent aggregates align with Gateway deployment
  profiles.

## [0.3.1] - 2026-02-04

### Added

- New user-facing docs entrypoints and references:
  - `docs/getting-started.md`, `docs/api.md`, `docs/faq.md`, `docs/README.md`
  - `CONTRIBUTING.md`, `SECURITY.md`, `ACKNOWLEDMENTS.md`
- LLM repo maps: `llms.txt`, `llms-full.txt`

### Changed

- Documentation refresh to match current code behavior (including `open_attachment` runtime-owned attachment reading).
- ReAct “plan-only followthrough” retry heuristic is enabled by default (disable with `_runtime.check_plan=false`).

### Fixed

- `create_memact_agent(...)` correctly wires `tool_executor=` into `create_local_runtime(...)`.
- CodeAct fenced-code execution now includes `allowed_tools` in the `TOOL_CALLS` effect payload (consistent allowlist enforcement).
- `manual_agent_demo.py` is self-contained (no missing `abstractagent.ui` dependency).

## [0.3.0] - 2026-01-06

### Added

- MemAct agent pattern (agent + logic + runtime adapter):
  - `src/abstractagent/agents/memact.py`
  - `src/abstractagent/logic/memact.py`
  - `src/abstractagent/adapters/memact_runtime.py`
- Manual LMStudio evaluation harness: `src/abstractagent/scripts/lmstudio_tool_eval.py`

## [0.2.0] - 2025-12-17

### Added

- Initial agent patterns and durable workflow adapters:
  - ReAct: `src/abstractagent/agents/react.py`, `src/abstractagent/adapters/react_runtime.py`
  - CodeAct: `src/abstractagent/agents/codeact.py`, `src/abstractagent/adapters/codeact_runtime.py`
- Common agent API: `src/abstractagent/agents/base.py`
