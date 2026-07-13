# Proposed: prune dead public surface (D2)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: D2

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Integration audit: zero production consumers import the package ROOT; several
root exports have no consumers anywhere.

## Current code reality
- create_codeact_agent / create_memact_agent: zero consumers outside own tests.
- create_react_agent: evidence scripts + own eval harness only.
- repl.py + the react-agent entry point (pyproject): deprecation stub printing
  "moved to AbstractCode".
- Of 13 root-exported tools, siblings use ALL_TOOLS + execute_python +
  self_improve; the other 11 are abstractcore re-exports consumed from
  abstractcore directly.

## Proposed direction
Remove the repl stub + entry point; deprecate the unused factories (or keep
create_react_agent as the documented quickstart and drop the sibling
factories); stop re-exporting the 11 pass-through tools (keep ALL_TOOLS).
Coordinate with the C5 ruling (fold removes the codeact factory anyway).

## Promotion criteria
Maintainer green-light (public-surface change; semver-relevant at next release).

## Non-goals
Breaking abstractcode/assistant imports — verify consumers before each removal.
