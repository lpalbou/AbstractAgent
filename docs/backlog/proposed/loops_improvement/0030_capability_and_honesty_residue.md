# Proposed: capability-stack + API-honesty residue (2026-07-13 twin audits)

## Metadata
- Created: 2026-07-13
- Status: Mostly executed (same-day batch 2, operator green light "make it optimal"); remaining items listed below
- Completed: N/A
- Proposal ID: operator-directed twin fable5 wave (capability A / honesty B)

## Batch-2 execution note (2026-07-13 evening)
The operator green-lit the residue ("continue to work and iterate on this,
make sure it is not only working, but optimal"). Executed same evening (see
CHANGELOG "batch 2"): skills_block named slot + skills-attachment contract doc,
MCP repeat-guard coverage + prompt lines, delegate substrate palette,
factory default resolution from AbstractCore config (+ qwen3:4b fallback),
max_output_tokens honesty (incl. the ReAct init nuke fix and dead
logic-constructor param removal), conclusion-call volatile tail (B-F7),
emit-step inventory + drift test + hooks.md stability contract (B-F12),
unattended recipe module, update_limits docstrings (B-F11).

STILL OPEN after batch 2 (updated 2026-07-14 — batch 4 executed the gated
remainder on operator green light; see CHANGELOG):
- EXECUTED batch 4: A1 capability-conditional fence default (one resolver,
  prompt + parser); `read_skill` schema-only builtin; the shared side-effect
  classifier (origin-READY via ToolDefinition.tags — full origin metadata
  still core's lane); `reset_react_task` vars transform (door applies between
  runs; one-run-per-task stands).
- RESOLVED BY CORE (2026-07-13 same evening, c1683): the `mcp::` wire-name P1
  — `abstractcore.tools.wire_naming` (pure deterministic aliasing at both
  declaration boundaries, stateless reverse-map at response normalization) —
  and the Ollama `num_ctx` per-call forwarding gap (`_effective_num_ctx`,
  absence sends nothing, invalid raises). Docs re-aligned; both flags closed.
- Work-door `reset_react_task` twin (promotes when gateway's work-door spec
  lands).
- `parse_retry_empty` vs `parse_retry_empty_response` rename (0028's
  consumer-surveyed contract wave; documented in the inventory until then).
- The `mcp::` wire-name P1 stays core's lane (flagged c1619).
- Bloc-store composition point (c1734/c1735, shape C ruled): the agent-lane
  bloc case is LARGE stable content (attachment-class documents; skills
  corpora on `skills_block` — byte-stable per run by contract, so bloc-shaped
  if it grows to bloc scale). The small-prefix case is already served by the
  session lane + volatile markers (supersedes my c1670 granularity note).
  When core's `prompt_cache` telemetry struct lands in GenerateResponse
  metadata, surface outcome/degraded_reason on LLM-call captures (additive;
  the emit-inventory drift test forces the declaration).
  - **EXECUTED 2026-07-15 (telemetry surface half)**: core's struct landed
    (mlx_provider writes `metadata["prompt_cache"]` with outcome /
    cached_tokens / fed_tokens / #FALLBACK degraded_reason; runtime's
    llm_client folds metadata into every LLM result). All three loops now
    lift it onto the `parse` payload as an additive `prompt_cache` key
    (`generation_params.prompt_cache_capture` — present exactly when the
    provider reported one, never an empty placeholder; no new step name, so
    the drift test is untouched by design). Pinned in
    `tests/test_parse_prompt_cache_capture.py` (present + absent, all three
    loops); documented in docs/hooks.md beside the 0028 common core. The
    bloc-scale `skills_block` composition question itself stays open (core's
    bloc-store lane).

## Context
The operator asked "do you see anything else to improve our agents… is
everything about skill and mcp properly handled?" and directed two fable5
adversaries. FIXED same-day (changelog): sibling cache-busting iteration
headers, system_prompt_extra composition, delegate substrate+sampling
inheritance, CodeAct per-run routing, terminal outcome field, call_id on
observe emits, review_mode default flip + comment corrections,
context_warning emit + overflow docs. THIS item holds the residue that needs
design, coordination, or a green light.

## Residue (audit numbering: A = capability, B = honesty)

### Skills (A-Q1)
- The seam is now REAL in all three loops (`system_prompt_extra` composes) —
  a host can attach `format_available_skills_xml(...)` output + narrow
  `allowed_tools` via `effective_tools(...)` today with zero imports. What
  remains: (a) DOCUMENT the contract (docs/skills-attachment.md: the var, the
  fixed position, cache-stability requirement — the extra must be byte-stable
  per run); (b) decide the named-slot question (a `skills_block` var beside
  `system_prompt_extra` avoids last-writer-wins between a delegation
  directive and a skills block — today delegate children overwrite the key);
  (c) a `read_skill(name)` schema-only builtin for progressive disclosure
  (open_attachment precedent). Selection/trust stay abstractskill's;
  per-phase resolution stays runtime's (phases.yaml skills sections).

### MCP (A-Q2)
- **Cross-seat P1, FLAGGED to core (c-thread)**: `mcp::server::tool` names go
  onto the wire verbatim; OpenAI/Anthropic reject non-`[a-zA-Z0-9_-]`
  function names — declaring any MCP tool 400s the whole call on strict
  providers. Fix is core's (wire-safe alias + reverse map); my loops are the
  blast radius.
- Side-effect vocabulary is local-name-only: the duplicate-side-effect guard
  and the "don't batch side-effectful tools" prompt line don't cover MCP
  writes. Fix shape: treat `origin.type == "mcp"` / the `mcp` tag as
  side-effecting in the guard.
- Document the MCP wiring recipe for these loops (docs/tools.md section).

### CodeAct native-primary (A-Q3 → proposal 0014/A1)
- The flip is now DE-RISKED as a capability-conditional: default
  `codeact_fenced_fallback = not supports_native_tools` (runtime seeds the
  bit per run) AND make the prompt's fence line conditional in the same wave
  (a prompt teaching a disabled channel manufactures dead replies).
  Promotion: the A1 green light.

### Delegation substrate palette (A-Q4)
- Host-gated named profiles: `_runtime.delegate_substrates = {name: {provider,
  model}}`; tool arg `substrate: string` resolved against it; unknown name =
  loud tool error. NEVER raw provider/model strings in tool args (the model
  choosing its own substrate = self-escalation); palette does not propagate
  to grandchildren by default. Composes with the operator's
  granted-resource principle + the recorded per-task escalation door.

### Work-door hosting (A-Q5)
- `reset_react_task` twin (reseed context.task, optional handoff-seeded
  messages) for queued work — v1 stays one-run-per-task, door-owned queue
  (document as the intended shape).
- Packaged "unattended" recipe (exclude ask_user + the directive) so the
  work door doesn't rediscover the delegate site's two moves.

### Honesty residue (B)
- **Emit inventory + drift test (B-F12)**: per-adapter step-name constants
  (DEFAULT_EVENT_MAP pattern) with a test diffing them against actual
  `emit(` call sites; fold into hooks.md. Also the naming split
  (`parse_retry_empty` vs `parse_retry_empty_response`) — rename only with
  0028's consumer-surveyed contract wave.
- **Error-string stability contract (B-F13)**: declare in hooks.md/api.md
  which strings are contracts (canonical event names, #FALLBACK/#TRUNCATION
  markers) and that error prose is NOT, except an enumerated list.
- **Default model (B-F8)**: factories hardcode `qwen3:1.7b-q4_K_M` — too
  weak for ~19 tool schemas + verifier. Preferred: resolve from AbstractCore
  config defaults when unset (#FALLBACK label), keep a small-model example
  in docs only. Needs a green light (changes first-run behavior).
- **`max_tokens` means three things (B-F9)**: ReAct output-cap-then-nuked;
  siblings warning-ceiling-only; logic constructors accept-and-ignore it.
  Rename/wire decision rides 0024/0021 (semver-visible).
- **Conclusion-call cache bust (B-F7)**: the max-iterations conclusion
  appends to the SYSTEM prompt under the same cache key; move to the
  volatile-tail pattern (small, safe — batch with the next adapter touch).
- **update_limits doc drift (B-F11)**: docstring lists 6 of 9 accepted keys;
  ReAct ignores mid-run max_history_messages updates.

## Promotion criteria
Skills contract doc + MCP guard + conclusion-call fix are surgical — next
defect batch. A1 flip, default-model change, max_tokens rename need explicit
green lights (behavior/semver). Work-door items promote when gateway's work
door spec lands. Emit inventory rides 0028's contract wave.
