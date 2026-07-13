# Changelog

## Unreleased (2026-07-12)

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
