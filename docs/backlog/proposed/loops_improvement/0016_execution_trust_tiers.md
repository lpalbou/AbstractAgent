# Proposed: execution trust tiers — owned-host / workspace-contained / isolated (A3)

## Metadata
- Created: 2026-07-12
- Status: Proposed
- Completed: N/A
- Proposal ID: A3

## ADR status
- Governing ADRs: None
- ADR impact: Needs new ADR when promoted (this IS durable policy)

## Context
Maintainer's deployment vision (2026-07-12, verbatim in substance): "at some
point, some agents should fully own the machine they work on (or a vps)". The
industry guidance (microVM/container isolation) applies to SHARED hosts running
UNTRUSTED tasks — not to a fleet resident that owns its box. Isolation is a
spectrum matched to trust and blast radius, not a blanket rule.

## Current code reality
- Today's only executor is workspace-contained-ish: subprocess + timeout +
  allowlist + approval gates; LocalSandbox docstring honestly says dev-only.
- The framework already has the owned-host safety substrate: immutable run
  ledgers (audit), tool approval gates, workspace containment, per-agent
  identity (H8 aliases).

## Proposed direction
Three OPERATOR-CONFIGURED tiers, honestly labeled: (1) owned-host — full host
execution; safety = ledger audit + workspace + egress policy; the fleet/VPS
vision; (2) workspace-contained — today's subprocess + timeout + approval;
label as containment-not-isolation; (3) isolated — container/microVM-class for
untrusted or multi-tenant work; needed the day gateway runs third-party tasks.
The tier is config, never a code default. Vocabulary feeds the runtime
executor design (0015).

## Promotion criteria
Maintainer ruling on the tier model; runtime executor (0015) scoping.

## Validation ideas
Config knob test: tier selection changes which executor binds; unknown tier
refuses loudly.

## Non-goals
Building the isolated executor now; changing today's defaults.
