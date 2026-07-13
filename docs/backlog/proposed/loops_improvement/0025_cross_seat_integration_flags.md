# Proposed: cross-seat integration flags (D3 + D4 — other seats' lanes)

## Metadata
- Created: 2026-07-12
- Status: Proposed (tracking item — not this package's build)
- Completed: N/A
- Proposal IDs: D3, D4

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Two findings from the integration audit belong to other seats; recorded here
so they are not lost, to be flagged on the hub at the right moment.

## The flags
- D3 (abstractassistant): agent_host.py imports CodeActAgent/MemActAgent into
  construction branches no caller can select (llm_manager.py hardcodes
  agent_kind="react") — dead imports to prune or a selector to build; their
  call.
- D4 (runtime/entity-agency plan): the entity CHAT lane still runs its own
  hand-rolled tool loop (identity/chat.py bounded tool-rounds while-loop)
  while only VISITS got the ReAct middle — the exact split behind the
  2026-07-11 Mnemosyne incident; already the entity-agency plan's territory
  (bundle-as-middle direction), listed for map completeness.

## Promotion criteria
D3: flag to assistant seat opportunistically. D4: superseded the moment the
entity-agency plan's chat-lane phase lands — then close this item.

## Non-goals
Building either from this package.
