# Proposed: native execute_python/execute_command as the primary code-action path (A1)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: A1

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Modern CodeAct implementations (OpenHands) run code through NATIVE function
calling (`execute_ipython_cell` as a declared tool) — code runs because the
model CALLED execution, never because a reply contained a code block. ReAct
already ships `execute_python` + `execute_command` as approvable,
allowlist-gated tools in ALL_TOOLS.

## Current code reality
- ReAct: native tool calls execute via the act node today; nothing to build.
- CodeAct: fenced-sniffing is the primary path (legacy prompted-model variant).
- Runtime seam AGREED (commons c1110-c1112): runtime owns any future executor
  as a durable effect; adapters own the loop shape. No native execution lane
  exists in runtime today; my sandbox/local.py is the only prototype
  (dev-only subprocess).

## Proposed direction
Document native-call-first as the execution contract for the loops; fenced
extraction survives only as the explicit-flag fallback (see planned 0010 for
the ordering fix). One conformance test pinning the native path.

## Why it might matter
Removes the intent-ambiguity class entirely on tool-calling models; aligns
with the field.

## Promotion criteria
Maintainer green-light on the A-series; lands naturally with 0010.

## Validation ideas
Conformance test: a native execute_python tool call executes; the same code in
a fence with the flag OFF does not.

## Non-goals
No executor/isolation work (runtime's lane).
