# Changelog

## [0.3.15] - 2026-09-26

### Fixed
- **A reply that announces tool use without calling a tool is no longer the final answer** (ReAct, CodeAct, MemAct). With Qwen3.x on MLX the model often ended a step with one sentence ("I have strong material. Let me verify … before writing the digest.") and the run completed without doing the work. A short reply that ENDS on such an announcement, and a reply whose tool calls could not run (unknown tool name, a call cut off mid-parameter, calls left in the thinking block that AbstractCore could not recover), is now re-prompted ONCE: the failed reply, reasoning included, is quoted verbatim in the corrective user message (Qwen3.5/3.6 templates strip `<think>` from assistant history turns), which asks the model to call the tools now or answer directly. The re-prompt call is recorded in the ledger (`_runtime_observability.reprompted` on the `LLM_CALL` payload). If the second reply fails the same way, or no iteration is left, the step ends with a visible error and `stop_reason.code = "no_tool_call"` in all three loops (ReAct concludes from what the run already has). Neither the announcement nor tool markup is ever published as the answer. Genuine short finals ("Done.", refusals, questions, "Let me summarize: …", "… unless you say otherwise") are not re-prompted. Switches: `_runtime.check_plan` (announcements) and `_runtime.check_unrunnable_calls` (markup), independent. English only for announcements (backlog 0033).
- ReAct's older long-reply followthrough check stays a soft nudge and never ends a step with an error.
- ReAct's long-reply soft nudge fires at most once per step: when the model sends a no-call reply again after the nudge, that reply is the answer. A model that repeated a legitimate long answer was nudged on every iteration until the budget ran out.
- The ReAct conclusion path never publishes an announcement or tool markup as the answer. When it falls back to the progress report, tool markup is stripped and the thought of a cycle whose reply was an announcement or unrunnable markup is left out. The CodeAct/MemAct budget terminals no longer pick a re-prompted reply or a turn that carried tool calls (pre-call narration) as the agent's last words.
- On the last iteration, a reply that announces tools or carries tool calls that cannot run stops the turn with `stop_reason.code = "no_tool_call"` and `budget_exhausted: true` in ReAct, CodeAct and MemAct alike.
- Announcement detection also recognises "Okay, fetching …", "Time to check …" and "I just need to read …".
- **Crash fix for 0.3.13 and 0.3.14 users:** CodeAct and MemAct crashed with a `NameError` when a host-granted `delegate_agent` substrate profile set `thinking` (`normalize_thinking` was called without being imported).

### Added
- `jinja2>=3.1` in the `test` and `dev` extras: a test renders a vendored copy of the Qwen3.6 chat template to prove the quoted reply reaches the prompt.
- Tool calls AbstractCore recovered from the thinking block (`metadata.tool_calls_from_reasoning`) are counted in the ReAct `parse` / `parse_tool_calls` events, the cycle entry, the report and an `info` notice; CodeAct and MemAct add `from_reasoning` to `parse_tool_calls`.
- New events: `parse_reprompt`, `parse_reprompt_failed` (all loops), `parse_reprompt_skipped`, `conclusion_announcement_dropped` (ReAct). `parse_retry_plan_only` still fires for announcements.

## [0.3.14] - 2026-09-26

### Changed
- Raised the AbstractRuntime floor to `AbstractRuntime>=0.5.0` across the base, `apple` and `gpu` install profiles (the runtime release with live token deltas). AbstractRuntime 0.5.0 brings AbstractCore 2.16.0 or newer with it; the direct `abstractcore[tools]>=2.13.41` floor is unchanged.

### Added
- **Delegated children stream too.** `delegate_agent` children in ReAct, CodeAct and MemAct inherit the parent's `_runtime.stream`, so a run that streams its replies to a live view keeps streaming when the agent delegates. An explicit `False` is inherited as well; an unset value stays unset. Needs AbstractRuntime 0.5.0 (live token deltas).

### Documentation
- `docs/troubleshooting.md` collects symptom-first fixes; `docs/architecture.md` shows the full ReAct graph (review nodes included) and the `_runtime` controls delegated sub-agents inherit; `docs/api.md` documents the `stream` slot.

## [0.3.13] - 2026-09-23

### Changed
- Raised dependency floors to `abstractcore[tools]>=2.13.41` and `AbstractRuntime>=0.4.32` across the base, `apple` and `gpu` install profiles. The agent loops rely on runtime 0.4.32 for turn grounding (`abstractruntime.turn_grounding`), run cancellation that stops in-flight effects, and speculation inheritance.
- **Default `max_iterations` is 20** for `ReactAgent`, `CodeActAgent`, `MemActAgent`, the adapters' scratchpad seeds and `resolve_max_iterations`, matching the runtime Agent node. Explicit workflow or host values still win, and an explicit `max_iterations=0` is honoured instead of falling back to the default.
- **`delegate_agent` children inherit the parent's iteration budget** (20 as the floor) instead of a fixed 10. An explicit `max_iterations` tool argument wins. Children also inherit the parent's effective provider/model, sampling, `thinking`, `speculation`, output cap and approval policy.
- **`ReactAgent` `review_mode` is on by default** (as in CodeAct): a candidate final answer is checked by a verifier, and an incomplete one re-enters the tool loop with the verifier's suggested tool calls. A verifier failure degrades to accepting the answer with a `#FALLBACK` warning. Set `review_mode=False` to skip the extra verifier call.
- **Prompt-prefix stability across all three loops.** The iteration counter, scratchpad, plan and drained guidance no longer live in the cached system prompt. In a tool loop, the loop-position line (`[loop] iteration N of M.`, plus any `[budget]`/`[plan]` lines) is appended once to the durable transcript, marked `_af_synthetic: "loop_tail"`, so iteration N's prompt is an exact prefix of iteration N+1's. On a conversational turn, the position line is dropped and only actionable content (`[budget]`, `[plan]`) is merged into the user message.
- **Byte-stable turn grounding.** ReAct, CodeAct and MemAct stamp the runtime grounding envelope (`<runtime_metadata>…</runtime_metadata>`) into the durable user turn once, at the `reason` boundary, through `abstractruntime.turn_grounding.stamp_user_turn_grounding`. What a turn is sent with is what is stored and replayed, so prefix caches restore the whole previous prompt. Hosts that replay their own transcript can pass the stamped message back unchanged: ReAct compares the turn's text, not its bytes, and does not append the task twice.
- The tool-batching rule in the loop prompts is target-scoped: never batch two calls on the same target; calls on independent targets may share one turn (the runtime runs a batch in order).
- Assistant tool-call messages keep the model's reasoning `content`, and the scratchpad is no longer duplicated into the system prompt. Bounded previews carry a `#[WARNING:TRUNCATION]` marker.
- MemAct is labelled **experimental** in the docs and API.
- The deprecated `react-agent` script no longer reads `ABSTRACTCODE_PROVIDER` / `ABSTRACTCODE_MODEL`. Agent loops read no behaviour environment variables; provider, model and gates come from run vars.

### Added
- **Run-scoped speculation (MTP) controls.** `_runtime.speculation` (`False`, `True`, or a Core dict such as `{"mode": "native_mtp", "num_draft_tokens": 2}`) is forwarded on every LLM call of ReAct, CodeAct and MemAct and inherited by delegated children. Explicit Off and a host-granted delegate substrate profile's `speculation` take precedence; Core owns defaults, capability checks and execution.
- **Native-loop registry for gateway bundles** (`adapters/native_loop_registry.py`), exported at the top level: `audit_native_loop_manifest`, `materialize_native_loop_spec`, `build_native_loop_manifest`, `pack_native_loop_bundle`. A manifest without `metadata.allowed_tools` materializes with the host default tool set. Operator scripts: `scripts/build_native_loop_bundle.py` and `scripts/verify_native_loop_gateway_import.py`.
- **Loop hooks** (`LoopHooks`, `HookEvent`, `DEFAULT_EVENT_MAP`): listen, steer and capture across all three loops, including `init`, `tool_proposed`, `inbox_drained`, `inbox_undelivered` and a single terminal event per turn. See `docs/hooks.md`.
- **ReAct planning:** the `update_plan` builtin keeps a checklist in the scratchpad and renders it on the loop tail.
- **`read_skill` builtin** (schema-only) for progressive disclosure of attached skills, and a fixed `_runtime.skills_block` prompt slot.
- **Delegate substrate palette:** `delegate_agent` accepts an optional `substrate` naming a host-granted profile (provider, model, `thinking`, `speculation`).
- **Composition helpers for hosts:** `create_react_workflow(final_next_node=...)`, `reset_react_turn(run_vars)` and `reset_react_task(run_vars)`, `_runtime.turn_id` / `_runtime.llm_payload_extras` pass-through, `_temp.turn_captures`, and `_runtime.suppress_loop_tail` to keep task-agent chrome out of conversational lanes.
- **Loop safety and telemetry:** stuck-streak termination for repeated identical tool batches (`adapters/progress.py`), `mcp::` tools covered by the repeat guard, a one-shot `context_warning` event when token use nears the model window, `prompt_cache` telemetry and `call_id` on `parse` / `observe` payloads, a machine-readable terminal `outcome` in `complete_output`, and visible allowlist pruning (`_runtime.allowlist_pruned`).
- **Tool-result media (ReAct):** a successful tool result whose output declares a `media` list (paths or `{"$artifact": id}` refs, for example a camera capture) is attached to the next model call, so the agent sees what its tools captured (at most 6 items per turn).
- Streaming pass-through (`_runtime.stream`), `max_output_tokens` on all three facades, `abstractagent.agents.unattended` (`unattended_allowlist`, `unattended_runtime_overrides`) and `abstractagent.metrics.prefix_reuse` (`measure_prefix_reuse`, `prefix_reuse_report`).

### Fixed
- Tool calls announced without ids get one id (`call_1`, `call_2`, …) shared by the assistant message and the executed batch, so tool results pair with their calls instead of being reported as missing and folded into a user message. Synthesized carrier messages are marked `_af_synthetic: "tool_result"` so the runtime never grounds them as the user's turn.
- CodeAct and MemAct transcripts are valid on strict providers (native OpenAI) from the second iteration onward; multi-turn `ask_user` runs no longer fail on native OpenAI.
- The review verifier schema is accepted by OpenAI strict-mode validators.
- `delegate_agent` cannot widen its child's tool grant beyond the parent's, and explicit budgets of 0 are no longer widened.
- Repeated identical tool batches no longer replay stale results; a MemAct finalize failure keeps the held answer; CodeAct resets its empty-retry streak after a success and only runs fenced code on action replies.
- `thinking` free-form strings are normalized instead of failing at the provider boundary, and string `prompt_cache_binding` values are sent as `prompt_cache_key`.
- Steering state no longer crosses task boundaries, and hook steering is discarded for failed runs.

### Removed
- `$act_only` reference rendering of tool observations: every tool result is rendered as plain content. `ToolDefinition.act_only` has no effect in the agent loops.
- MemAct constructor options that had no effect.

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
