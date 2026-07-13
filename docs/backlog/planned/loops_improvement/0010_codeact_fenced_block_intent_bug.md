# Planned: CodeAct fenced-block execution ignores intent (A4)

## Metadata
- Created: 2026-07-12
- Status: Planned
- Completed: N/A
- Proposal ID: A4

## ADR status
- Governing ADRs: None (repo has no ADR system)
- ADR impact: None

## Context
CodeAct's defining behavior: a reply with no native tool calls but containing a
fenced Python block executes that block as an action. Adversarial parity audit
(2026-07-12) + maintainer review confirmed the check ORDER makes this fire on
replies that are not actions at all.

## Current code reality
- `src/abstractagent/adapters/codeact_runtime.py` parse_node: code extraction
  (`logic.extract_code`, first fenced block via `_CODE_BLOCK_RE` in
  `logic/codeact.py:20`) runs BEFORE the `FINAL:` marker check — a reply
  `FINAL: here's an example: ```python ...` executes the example.
- No intent discrimination: illustrative code in an explanation turn executes.
- Execution itself is allowlist-gated (`execute_python` must be granted) and
  rides the normal TOOL_CALLS effect — the gate is not the bug; the trigger is.

## Problem
A user asking a CodeAct agent to DESCRIBE code gets the description executed.
Security-relevant (unintended execution) and correctness-relevant (the reply
was a final answer).

## What we want to do
Reorder + gate: (1) `FINAL:`/final-answer detection precedes code extraction;
(2) fenced extraction only when an explicit flag enables the prompted-model
fallback (native `execute_python` tool calls are the primary path — proposal
A1); (3) never extract from a turn that carries no action intent.

## Scope
parse_node ordering in codeact_runtime.py + the flag; ~10 lines.

## Non-goals
No change to the executor, sandbox, or allowlist gating; no CodeAct retirement
(C5 is the maintainer's ruling).

## Expected outcomes
Illustrative/final replies never execute; the fallback still works for prompted
local models when enabled.

## Validation
Two kernel tests: (a) `FINAL:` reply containing a fenced block completes with
the block NOT executed; (b) flag-on, non-final reply with a fenced block
executes exactly as today.

## Progress checklist
- [ ] Reorder final-answer check before extraction
- [ ] Flag-gate the fenced fallback
- [ ] Two tests + changelog
