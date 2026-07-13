# Proposed: fold CodeAct into ReAct and retire it (C5 — MAINTAINER RULING REQUIRED)

## Metadata
- Created: 2026-07-12
- Status: Proposed (decision item)
- Completed: N/A
- Proposal ID: C5

## ADR status
- Governing ADRs: None
- ADR impact: Needs decision record when ruled (user-facing selector changes)

## Context
Integration audit: CodeAct's only reachable production surface is
abstractcode's --agent selector (never a default; assistant imports it
unreachably; no flow/bundle/entity route). Its unique value is ~15 lines
(fenced extraction -> execute_python mapping). Everything else restates ReAct
at an older stage, including its own verifier fork, and it carries the
strict-provider bug (planned 0011) and prompt-cache instability (iteration
counter in system prompt, logic/codeact.py:94).

## Proposed direction
(1) ReAct gains a code_act flag: fenced extraction in parse (after 0010's
intent fix) mapping to an execute_python tool call through the existing gated
act path; (2) abstractcode --agent codeact maps to ReactAgent+flag (UX
unchanged); (3) codeact files retire after migration (~1,800 lines).
MemAct is explicitly NOT in this ruling: different architecture (per-turn
memory rewriting), kept experimental (see 0023).

## Why it might matter
One loop to harden instead of three; CodeAct inherits every ReAct capability
instantly; the maintainer's A1/A4 direction works under either outcome.

## Promotion criteria
THE MAINTAINER'S EXPLICIT RULING — this changes a user-facing selector.

## Validation ideas
abstractcode codeact-mode acceptance run behaving identically on a fenced-code
task pre/post fold.

## Non-goals
Deleting anything before abstractcode migrates; touching MemAct.
