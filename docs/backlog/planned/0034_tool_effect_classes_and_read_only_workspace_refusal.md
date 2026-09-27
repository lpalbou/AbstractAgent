# Planned: Tool effect classes; `execute_python` and every executing/writing tool refuse a read-only workspace

## Metadata
- Created: 2026-09-27
- Status: Planned (phase: Automations v1, framework backlog 0928; contracts: framework `untracked/design/automations-CONTRACTS.md` rev 2, C4)
- Completed: N/A

## Context

Automations v1 gives a "discussion" (a durable session forked from an automation's conversation) the automation's workspace mounted
READ-ONLY (operator ruling 2026-09-26). The runtime enforces it through a `workspace_read_only` flag on the workspace scope and a single
classification table of tool effects (`TOOL_EFFECT_CLASSES`: read / write / execute / delegate / comms / memory-write): anything that
writes or executes is refused inside a read-only root, and an UNCLASSIFIED tool is refused too (fail closed). Astra's review of the
contracts (framework `untracked/design/astra/turn6.reply.md`, amendment 4) found that tool-name allowlists alone are bypassed by this
package's code-execution tools: `src/abstractagent/tools/code_execution.py:33` (`execute_python`) accepts code without any path argument
and runs a subprocess, so a path-based guard never sees it.

## Problem

This package owns tools that execute or write without a path: `execute_python` (code_execution.py), any shell/subprocess helper, file
writers reached through the CodeAct action path, delegation (`delegate_agent`), comms (`send_email`, `send_telegram_message`) and
memory writes (`memory_note`, diary). None of them declares an effect class today, so the runtime cannot classify them without
maintaining a name list here and there.

## Scope

- Declare an effect class on every tool this package registers (a `effect_class` attribute/field in the tool spec, or a registry
  mapping shipped next to the tools), matching the runtime's `TOOL_EFFECT_CLASSES` vocabulary; `execute_python` and any subprocess
  helper = `execute`; CodeAct's action executor honours the run's read-only scope (refuses file writes and code execution inside the
  root, with the same refusal text the runtime uses).
- A test that enumerates every tool the package exposes and fails when one has no effect class (mirrors the runtime's
  unclassified-tool test); a test that a discussion-shaped run (read-only scope) gets `execute_python` refused and `read_file` allowed.
- Out of scope: the runtime side (framework 0928 / runtime 0847), the gateway restamping.

## Validation

`tests/test_tool_effect_classes.py`; the framework acceptance script `abstractgateway/scripts/accept_automations_v1.py` step "blocked
mutations" passes with this package installed.

## Related

Framework 0928 (umbrella) and `automations-CONTRACTS.md` C4/B; runtime 0847; 0016 (execution trust tiers) and 0020 (provider stop
semantics over heuristics) in `proposed/loops_improvement/`.

## Status note (2026-09-27, after Automations v1 was built)

Still **planned**: the agent-side declaration was not done. The runtime classified the tools centrally instead, so the
read-only guarantee holds without this item.
- **The central table.** `abstractruntime.integrations.abstractcore.tool_effects.TOOL_EFFECT_CLASSES` (runtime `ab30729`)
  names this package's tools too. It uses the spelling `exec`, not `execute`. For example, `execute_python` is `exec`,
  `delegate_agent` is `delegate` and `send_email` is `comms`.
- **Refusal.** `read_only_refusal(name)` refuses write/exec tools inside a read-only root, and refuses any unclassified
  tool (fail closed).
- **Evidence.** The framework E2E and the gateway acceptance script show discussions refusing `write_file` and
  `execute_command`. Runtime tests cover the `execute_python` refusal. Framework root backlog 0928 is completed and
  unreleased.
- **The same table drives the unattended-tools grant.** Under `policy.tool_approval: auto`, an automation's occurrence
  gets every classified tool (65) unless the target narrows `allowed_tools`. Tools outside the table (third-party MCP)
  still ask.

What remains open here:
- **The package declares its own effects.** Each tool this package registers declares its effect class next to the tool,
  so a new agent tool cannot ship unclassified without a test going red. Today it is refused under read-only, and it is
  NOT in the unattended grant, which is safe but silent for the author.
- **The table is not duplicated by hand.** The runtime table is either generated from those declarations or checked
  against them.
- **CodeAct refuses with the runtime's refusal text.** Its action executor refuses file writes and code execution in a
  read-only scope with the same text the runtime uses. Verify whether the runtime's tool-call refusal already covers the
  CodeAct path before building anything.
