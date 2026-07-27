from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _base_vars(*, task: str) -> Dict[str, Any]:
    return {"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}}


@pytest.mark.basic
def test_react_delegate_agent_runs_subworkflow_and_returns_tool_observation() -> None:
    """delegate_agent should run a fresh sub-agent run and return its answer as a tool observation."""

    call_count: dict[str, int] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        call_count[run.run_id] = int(call_count.get(run.run_id, 0) or 0) + 1

        is_child = bool(getattr(run, "parent_run_id", None))
        if is_child:
            return EffectOutcome.completed({"content": "Child answer", "tool_calls": [], "finish_reason": "stop"})

        idx = call_count[run.run_id]
        if idx == 1:
            return EffectOutcome.completed(
                {
                    "content": "Delegating.",
                    "tool_calls": [
                        {
                            "name": "delegate_agent",
                            "arguments": {"task": "Find X", "context": "Only look at file A", "tools": ["read_file", "search_files"]},
                            "call_id": "call_1",
                        }
                    ],
                    "finish_reason": "tool_calls",
                }
            )

        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )

    # Minimal tool defs: include delegate_agent + a couple of read-only tools so allowlist normalization works.
    tool_defs = [
        DELEGATE_AGENT_TOOL,
        ToolDefinition(name="read_file", description="Read", parameters={}),
        ToolDefinition(name="search_files", description="Search", parameters={}),
    ]

    workflow = create_react_workflow(logic=ReActLogic(tools=tool_defs), workflow_id="react_agent")
    runtime.workflow_registry.register(workflow)

    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="Parent task"), actor_id=None, session_id=None)

    for _ in range(200):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED

    output = final.output if isinstance(final.output, dict) else {}
    msgs = output.get("messages") if isinstance(output, dict) else None
    assert isinstance(msgs, list) and msgs
    tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert any("delegate_agent" in str(m.get("content") or "") and "Child answer" in str(m.get("content") or "") for m in tool_msgs), msgs


@pytest.mark.basic
@pytest.mark.parametrize(
    ("parent_budget", "explicit_child", "expected_child_budget"),
    [
        # Agency-caps ruling (2026-07-11): the child inherits the parent's budget
        # with 20 as the floor guard; the old hardcoded 10 was a fear default.
        (25, None, 25),
        # A parent narrowed below the ruling does not narrow the child by default.
        (5, None, 20),
        # An explicit tool-arg budget wins (operator/model choice, above or below).
        (25, 40, 40),
        (25, 8, 8),
    ],
)
def test_react_delegate_child_iteration_budget(parent_budget: int, explicit_child: Optional[int], expected_child_budget: int) -> None:
    """The delegated child's `_limits.max_iterations` follows the caps ruling:
    inherit parent (min 20) by default; an explicit `max_iterations` arg wins."""

    seen_child_budgets: list[int] = []
    call_count: dict[str, int] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        call_count[run.run_id] = int(call_count.get(run.run_id, 0) or 0) + 1

        if bool(getattr(run, "parent_run_id", None)):
            limits = run.vars.get("_limits") if isinstance(run.vars, dict) else None
            if isinstance(limits, dict):
                seen_child_budgets.append(int(limits.get("max_iterations") or 0))
            return EffectOutcome.completed({"content": "Child answer", "tool_calls": [], "finish_reason": "stop"})

        if call_count[run.run_id] == 1:
            arguments: Dict[str, Any] = {"task": "Find X"}
            if explicit_child is not None:
                arguments["max_iterations"] = explicit_child
            return EffectOutcome.completed(
                {
                    "content": "Delegating.",
                    "tool_calls": [{"name": "delegate_agent", "arguments": arguments, "call_id": "call_1"}],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )
    tool_defs = [DELEGATE_AGENT_TOOL, ToolDefinition(name="read_file", description="Read", parameters={})]
    workflow = create_react_workflow(logic=ReActLogic(tools=tool_defs), workflow_id="react_agent")
    runtime.workflow_registry.register(workflow)

    vars_ = _base_vars(task="Parent task")
    vars_["_limits"] = {"max_iterations": parent_budget}
    run_id = runtime.start(workflow=workflow, vars=vars_, actor_id=None, session_id=None)

    for _ in range(200):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    assert seen_child_budgets and seen_child_budgets[0] == expected_child_budget


def _run_delegate_with(
    *,
    parent_allow: list[str],
    requested_tools: Any,
    parent_tool_policy: Optional[Dict[str, Any]] = None,
) -> tuple[RunStatus, list[list[str]], list[Optional[Dict[str, Any]]], list[Dict[str, Any]]]:
    """Drive a parent that delegates with an explicit `tools` arg; capture the
    child's resolved allowlist + tool_policy and the parent's tool observations."""
    child_allowlists: list[list[str]] = []
    child_policies: list[Optional[Dict[str, Any]]] = []
    call_count: dict[str, int] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        call_count[run.run_id] = int(call_count.get(run.run_id, 0) or 0) + 1
        if bool(getattr(run, "parent_run_id", None)):
            rns = run.vars.get("_runtime") if isinstance(run.vars, dict) else {}
            if isinstance(rns, dict):
                child_allowlists.append(list(rns.get("allowed_tools") or []))
                child_policies.append(rns.get("tool_policy"))
            return EffectOutcome.completed({"content": "Child answer", "tool_calls": [], "finish_reason": "stop"})
        if call_count[run.run_id] == 1:
            return EffectOutcome.completed(
                {
                    "content": "Delegating.",
                    "tool_calls": [
                        {"name": "delegate_agent", "arguments": {"task": "Find X", "tools": requested_tools}, "call_id": "call_1"}
                    ],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )
    tool_defs = [
        DELEGATE_AGENT_TOOL,
        ToolDefinition(name="read_file", description="Read", parameters={}),
        ToolDefinition(name="search_files", description="Search", parameters={}),
        ToolDefinition(name="execute_command", description="Run a shell command", parameters={}),
        ToolDefinition(name="write_file", description="Write", parameters={}),
    ]
    workflow = create_react_workflow(logic=ReActLogic(tools=tool_defs), workflow_id="react_agent")
    runtime.workflow_registry.register(workflow)

    vars_ = _base_vars(task="Parent task")
    vars_["_runtime"]["allowed_tools"] = list(parent_allow)
    if parent_tool_policy is not None:
        vars_["_runtime"]["tool_policy"] = parent_tool_policy
    run_id = runtime.start(workflow=workflow, vars=vars_, actor_id=None, session_id=None)

    for _ in range(200):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    output = final.output if isinstance(final.output, dict) else {}
    msgs = output.get("messages") if isinstance(output, dict) else []
    tool_msgs = [m for m in (msgs or []) if isinstance(m, dict) and m.get("role") == "tool"]
    return final.status, child_allowlists, child_policies, tool_msgs


@pytest.mark.basic
def test_delegate_child_tools_are_subset_of_parent_grant() -> None:
    """Grant containment (tool-tiers adversary P0, 2026-07-22): a model-picked
    `tools` arg cannot hand the child a tool the parent was never granted.
    Parent holds {read_file, delegate_agent}; child requests execute_command
    too — only the granted intersection survives."""
    status, child_allowlists, _pol, _msgs = _run_delegate_with(
        parent_allow=["read_file", "delegate_agent"],
        requested_tools=["read_file", "execute_command", "write_file"],
    )
    assert status == RunStatus.COMPLETED
    assert child_allowlists, "child never ran"
    granted = set(child_allowlists[0])
    assert "read_file" in granted
    assert "execute_command" not in granted
    assert "write_file" not in granted


@pytest.mark.basic
def test_delegate_all_requested_tools_ungranted_is_a_tool_error_not_a_grant() -> None:
    """When NONE of the requested tools are in the parent grant, delegate fails
    as a tool error (parent decides) — never silently spawns a toolless child
    with escalated names, never a run failure."""
    status, child_allowlists, _pol, tool_msgs = _run_delegate_with(
        parent_allow=["read_file", "delegate_agent"],
        requested_tools=["execute_command", "write_file"],
    )
    assert status == RunStatus.COMPLETED
    assert not child_allowlists, "child must not run when the whole request is ungranted"
    assert any(
        "subset of the parent" in str(m.get("content") or "") for m in tool_msgs
    ), tool_msgs


@pytest.mark.basic
def test_delegate_child_inherits_parent_tool_policy() -> None:
    """Approval policy inherits monotonically (adversary P0): a run-scoped
    tool_policy on the parent must reach the child, or a parent's tightening
    silently drops one level down."""
    policy = {"require_approval_tools": ["read_file"]}
    status, child_allowlists, child_policies, _msgs = _run_delegate_with(
        parent_allow=["read_file", "search_files", "delegate_agent"],
        requested_tools=["read_file", "search_files"],
        parent_tool_policy=policy,
    )
    assert status == RunStatus.COMPLETED
    assert child_allowlists, "child never ran"
    assert child_policies and child_policies[0] == policy

