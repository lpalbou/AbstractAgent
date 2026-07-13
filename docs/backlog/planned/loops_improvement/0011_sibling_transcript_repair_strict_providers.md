# Planned: CodeAct/MemAct transcripts 400 on strict providers (C2)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Proposal ID: C2

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
ReAct preserves assistant `tool_calls` metadata in the durable transcript and
repairs orphaned tool messages at the payload boundary (react_runtime.py
`_sanitize_llm_messages`, ~992-1005: synthesized deterministic tool results so
native OpenAI's "assistant tool_calls must be followed by matching tool
messages" contract holds). CodeAct and MemAct never received this wave.

## Current code reality
- codeact_runtime.py ~550: assistant messages appended WITHOUT `tool_calls`
  metadata while its sanitizer emits tool messages WITH `tool_call_id`
  (~288-292) — an orphan on any strict provider.
- memact_runtime.py ~256-260/583: same shape.
- Consequence: multi-iteration tool use on native OpenAI 400s at iteration 2.
  Unsurfaced because the loops' one consumer (abstractcode selector) runs local
  prompted providers.

## Problem
A functional bug: both sibling loops are broken on the provider class the
framework treats as first-tier.

## What we want to do
Port ReAct's assistant-tool_calls preservation + orphan repair to both
siblings — preferably by EXTRACTING the sanitizer into a shared module rather
than a third copy (see proposed 0021 shared-core refactor; if 0021 lands first
this item folds into it).

## Scope
Transcript append + sanitizer in both sibling adapters (or the shared module).

## Non-goals
No behavior change for ReAct; no new transcript format.

## Expected outcomes
A CodeAct/MemAct run with 2+ tool iterations against a strict-provider-shaped
handler produces payloads that satisfy the native tool-calling contract.

## Validation
Kernel test per sibling: scripted two-iteration tool run; assert every tool
message's `tool_call_id` is answered by a preceding assistant `tool_calls`
entry in the sanitized payload.

## Progress checklist
- [ ] Check 0021 state first (fold vs standalone)
- [ ] Preserve assistant tool_calls in sibling transcripts
- [ ] Port orphan repair
- [ ] Tests + changelog
