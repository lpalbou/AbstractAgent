# Changelog

All notable changes to `abstractagent` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
