# Proposed: Agent Gateway Install Boundary

## Metadata
- Created: 2026-05-08
- Status: Proposed
- Completed: N/A

## Context

AbstractAgent is the agent behavior layer built on AbstractRuntime and AbstractCore tools. In
Gateway deployments, Agent is pulled in for Visual Agent nodes and durable autonomous workflows.

## Current Code Reality

- Agent package metadata depends on `abstractcore[tools]` and `abstractruntime`.
- Agent imports/re-exports Core tool helpers from package root.
- There are no pending changes in the Agent repo.

## Problem

Agent is behind Gateway, but it is not a hardware or modality package. It should not grow Apple/GPU
extras or capability-package dependencies just to match a unified naming scheme.

Current dependency metadata is also too loose for the newer Gateway/Core stack:

- no minimum Core version;
- no minimum Runtime version;
- no explicit relation to Gateway server profiles.

## Proposed Direction

Keep Agent focused:

- `abstractagent`: Runtime plus Core tools.
- optional `dev` remains test/dev only.
- Do not add `apple`, `gpu`, `all-apple`, or `all-gpu` unless Agent later owns local model-engine
  dependencies, which it should avoid.

Gateway server profiles should depend on Agent when workflows can contain agent nodes.

Future cleanup:

- tighten dependency floors to the Gateway-supported baseline, for example
  `abstractcore[tools]>=2.13.10` and `abstractruntime>=0.4.6`;
- consider lazy package-root exports if importing `abstractagent` pulls tool stacks too eagerly;
- document Agent as a behavior package, not a deployment/profile owner.

## Pending Changes Guidance

No Agent pending changes are needed for this strategic pass.

Related pending changes:

- Gateway `server` should keep Agent because Gateway-hosted workflows need agent nodes.
- Root meta-package pins should align Agent with current Core/Runtime floors.

## Promotion Criteria

Promote when version-alignment work begins across Gateway, Runtime, Core, and root
`abstractframework`.

## Validation Ideas

- Packaging test for Agent dependency floors.
- Import test for `abstractagent` in a clean venv.
- Gateway flow test with an agent node under the chosen `abstractgateway[server]` profile.

