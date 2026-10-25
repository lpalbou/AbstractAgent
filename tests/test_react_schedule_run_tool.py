"""The `schedule_run` builtin (framework backlog 1045): a ReAct call becomes ONE `SCHEDULE_RUN` effect
with at most the agent's tools and never `schedule_run` itself; the host creates the wake."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.builtins import SCHEDULE_RUN_TOOL
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _run(arguments: Dict[str, Any], *, allowed_tools: List[str], handler: Any) -> RunState:
    calls: Dict[str, int] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        calls[run.run_id] = calls.get(run.run_id, 0) + 1
        if calls[run.run_id] == 1:
            return EffectOutcome.completed({"content": "Scheduling.", "finish_reason": "tool_calls",
                                            "tool_calls": [{"name": "schedule_run", "arguments": arguments, "call_id": "call_1"}]})
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
                      effect_handlers={EffectType.LLM_CALL: llm_handler}, workflow_registry=WorkflowRegistry())
    runtime.set_schedule_run_handler(handler)
    tool_defs = [SCHEDULE_RUN_TOOL, ToolDefinition(name="read_file", description="Read", parameters={}),
                 ToolDefinition(name="execute_command", description="Run", parameters={})]
    workflow = create_react_workflow(logic=ReActLogic(tools=tool_defs), workflow_id="react_agent")
    runtime.workflow_registry.register(workflow)
    run_id = runtime.start(workflow=workflow, session_id="chat-S",
                           vars={"context": {"task": "Remind me", "messages": []}, "_runtime": {"inbox": [], "allowed_tools": allowed_tools}})
    for _ in range(200):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    return runtime.get_state(run_id)


def _tool_messages(final: RunState) -> List[str]:
    msgs = (final.output or {}).get("messages") or []
    return [str(m.get("content") or "") for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]


@pytest.mark.basic
def test_schedule_run_passes_at_most_the_agents_tools_never_itself() -> None:
    plans: List[Dict[str, Any]] = []

    def host(run, plan):
        plans.append(plan)
        return {"automation_id": "auto-1"}

    final = _run({"after_s": 300, "task": "check the build", "tools": ["read_file", "execute_command", "schedule_run"]},
                 allowed_tools=["schedule_run", "read_file"], handler=host)
    assert final.status == RunStatus.COMPLETED
    (plan,) = plans
    assert plan["allowed_tools"] == ["read_file"]  # execute_command is not the agent's; schedule_run never passes on
    assert plan["session_id"] == "chat-S" and plan["mode"] == "same_session" and plan["kind"] == "task"
    assert plan["input_data"]["_runtime"]["allowed_tools"] == ["read_file"]
    assert any("auto-1" in m for m in _tool_messages(final))


@pytest.mark.basic
def test_schedule_run_inherits_the_agents_tools_when_omitted_and_a_note_gets_none() -> None:
    plans: List[Dict[str, Any]] = []
    final = _run({"after_s": 60, "note": "stretch"}, allowed_tools=["schedule_run", "read_file", "execute_command"],
                 handler=lambda run, plan: plans.append(plan) or {"automation_id": "auto-2"})
    assert final.status == RunStatus.COMPLETED
    assert plans[0]["allowed_tools"] == [] and plans[0]["kind"] == "note"
    plans.clear()
    _run({"after_s": 60, "task": "t"}, allowed_tools=["schedule_run", "read_file", "execute_command"],
         handler=lambda run, plan: plans.append(plan) or {"automation_id": "auto-3"})
    assert plans[0]["allowed_tools"] == ["execute_command", "read_file"]


@pytest.mark.basic
def test_schedule_run_outside_the_allowlist_is_refused_without_calling_the_host() -> None:
    plans: List[Dict[str, Any]] = []
    final = _run({"after_s": 60, "note": "x"}, allowed_tools=["read_file"], handler=lambda run, plan: plans.append(plan) or {"automation_id": "x"})
    assert final.status == RunStatus.COMPLETED
    assert plans == []
    assert any("schedule_run" in m for m in _tool_messages(final))


@pytest.mark.basic
def test_schedule_run_passes_notify_and_notes_notify_by_default() -> None:
    plans: List[Dict[str, Any]] = []
    host = lambda run, plan: plans.append(plan) or {"automation_id": "a"}  # noqa: E731
    _run({"after_s": 60, "note": "drink water"}, allowed_tools=["schedule_run"], handler=host)
    _run({"after_s": 60, "task": "t", "tools": []}, allowed_tools=["schedule_run"], handler=host)
    _run({"after_s": 60, "task": "t", "tools": [], "notify": True}, allowed_tools=["schedule_run"], handler=host)
    assert [p["notify"] for p in plans] == [True, False, True]
    assert plans[0]["notify_title"] == "drink water"
