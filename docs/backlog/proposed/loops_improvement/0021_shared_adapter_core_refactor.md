# Proposed: shared adapter core refactor (C1)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: C1

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Parity audit (2026-07-12): ~59% of CodeAct's 1,329 lines and ~44% of MemAct's
1,045 restate ReAct logic, mostly frozen at an older stage. Of ~25 recent
capabilities, siblings received 3. The one-source precedent already exists
in-repo: loop_hooks.py / tool_allowlist.py / generation_params.py.

## Current code reality
Triplicated near-verbatim: _new_message, ensure_*_vars, _compute_toolset_id,
hooks plumbing, allowlist helpers, ask_user block, delegate_agent block, the
five memory-builtin dispatch blocks, observe core, handle_user_response,
done/max_iterations plumbing; two divergent forks each of
_sanitize_llm_messages and the verifier. React adapter 2,285 lines; siblings
frozen forks.

## Proposed direction
Extract the shared blocks into an adapters/loop_core.py-style module; each
adapter keeps only its loop shape (react retries/verifier, codeact execute
node, memact compose/finalize). ~1,100-1,300 lines removed; every future wave
lands once. Planned 0011 (transcript repair) folds in naturally.

## Promotion criteria
Maintainer green-light; ideally AFTER the C5 fold ruling (retiring CodeAct
first shrinks the refactor).

## Validation ideas
Full suite green before/after; parity smoke for both siblings through the real
kernel; no behavior change pins.

## Non-goals
Feature parity for siblings beyond what the shared core carries for free.
