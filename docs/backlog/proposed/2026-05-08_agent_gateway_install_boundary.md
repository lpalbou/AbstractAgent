# Proposed: Agent Gateway Install Boundary

## Metadata
- Created: 2026-05-08
- Status: Proposed
- Completed: N/A

## Context

AbstractAgent is the agent behavior layer built on AbstractRuntime and AbstractCore tools. In
Gateway deployments, Agent is pulled in for Visual Agent nodes and durable autonomous workflows.

## Current Code Reality

- Agent package metadata depends on `abstractcore[tools]>=2.13.12` and
  `abstractruntime>=0.4.8`.
- Agent imports/re-exports Core tool helpers from package root.
- Agent exposes `apple`, `gpu`, `all-apple`, and `all-gpu` extras as Core/Runtime profile cascades.

## Problem

Agent is behind Gateway, but it is not a hardware or modality package. Its Apple/GPU extras should
stay pass-through Core/Runtime profile cascades rather than adding modality or local-engine
dependencies directly.

The risk is future drift: if Agent keeps older Core/Runtime floors, Gateway native profiles can
resolve a runnable server while Agent nodes use older Core/Runtime contracts.

## Proposed Direction

Keep Agent focused:

- `abstractagent`: Runtime plus Core tools.
- `abstractagent[apple]` / `abstractagent[gpu]`: pass-through Core and Runtime local-engine
  profile cascades.
- `abstractagent[all-apple]` / `abstractagent[all-gpu]`: pass-through Core and Runtime aggregate
  profile cascades.
- optional `dev` and `test` remain test/dev only.
- Do not add direct Vision, Voice, Music, Memory, or provider SDK dependencies to Agent profile
  extras unless Agent itself starts owning that behavior, which it should avoid.

Gateway server profiles should depend on Agent when workflows can contain agent nodes.

Future cleanup:

- consider lazy package-root exports if importing `abstractagent` pulls tool stacks too eagerly;
- document Agent as a behavior package, not a deployment/profile owner.

## Pending Changes Guidance

Keep the 0.3.5 release focused on dependency-floor/profile alignment. Do not add capability package
dependencies here; Gateway/Core compose capabilities.

Related pending changes:

- Gateway `server` should keep Agent because Gateway-hosted workflows need agent nodes.
- Root meta-package pins should align Agent with current Core/Runtime floors.

## Promotion Criteria

Promote when version-alignment work begins across Gateway, Runtime, Core, and root
`abstractframework`.

## Validation Ideas

- Packaging test for Agent dependency floors and pass-through profile cascades.
- Import test for `abstractagent` in a clean venv.
- Gateway flow test with an agent node under the chosen `abstractgateway[server]` profile.
