# Planned: delegate_agent child budget lies in CodeAct/MemAct (C3)

## Metadata
- Created: 2026-07-12
- Status: Completed
- Completed: 2026-07-13
- Proposal ID: C3

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The shared `DELEGATE_AGENT_TOOL` schema (logic/builtins.py ~313-320) documents:
child inherits the parent's budget (min 20) and an explicit `max_iterations`
argument wins — the c726/agency-caps ruling, implemented in ReAct
(react_runtime.py ~1630-1706) on 2026-07-11.

## Current code reality
- codeact_runtime.py ~772 and memact_runtime.py ~723: child `_limits.max_iterations`
  HARDCODED to 10; the `max_iterations` tool argument is ignored.
- The schema is shared, so for these two loops the schema LIES to the model.

## Problem
Schema/behavior drift + a fear-shaped cap the maintainer explicitly overruled.

## What we want to do
Apply ReAct's resolution in both siblings: explicit arg wins; otherwise
inherit parent with floor 20.

## Scope
~3 lines per adapter + parametrized pins mirroring
tests/test_react_delegate_agent_tool.py.

## Non-goals
No change to the no-recursion allowlist strip (wait-safety, correct as-is).

## Validation
Parametrized tests per sibling: (25,None)->25, (5,None)->20, (25,40)->40, (25,8)->8.

## Progress checklist
- [x] CodeAct fix + pins
- [x] MemAct fix + pins
- [x] Changelog

## Completion report
- Date: 2026-07-13
- Outcome: ReAct's ruled resolution mirrored verbatim in both siblings'
  delegate_agent branches (explicit `max_iterations` arg wins, even below 20
  — an operator/model's explicit choice; otherwise inherit the parent's
  budget via `resolve_max_iterations` with the ruled 20 floor). The shared
  DELEGATE_AGENT_TOOL schema now tells the truth for all three loops.
- Validation: parametrized 4-case matrix per sibling ((25,None)->25,
  (5,None)->20, (25,40)->40, (25,8)->8) in
  tests/test_sibling_loop_defects_batch.py; full suite 156 passed / 2
  skipped. Harness note: the allowlist gate intersects with the LOGIC's
  declared tools, so node-level delegate tests must declare
  DELEGATE_AGENT_TOOL on the logic (the agent constructors do).
