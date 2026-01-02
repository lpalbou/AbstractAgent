from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


def test_act_node_blocks_repeated_identical_edit_file_calls() -> None:
    logic = ReActLogic(
        tools=[
            ToolDefinition(
                name="edit_file",
                description="edit",
                parameters={},
            )
        ],
        max_history_messages=-1,
        max_tokens=None,
    )
    workflow = create_react_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="react_agent",
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {
                "pending_tool_calls": [
                    {
                        "name": "edit_file",
                        "arguments": {"file_path": "demo.py", "pattern": "a", "replacement": "b"},
                        "call_id": "1",
                    }
                ]
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    handler = workflow.get_node("act")

    plan1 = handler(run, _Ctx())
    assert plan1.next_node == "observe"
    assert plan1.effect is not None

    # Second identical attempt should be blocked by the loop guard (deterministic mutation tool).
    plan2 = handler(run, _Ctx())
    assert plan2.next_node == "observe"
    assert plan2.effect is None

    tool_results = run.vars["_temp"].get("tool_results")
    assert isinstance(tool_results, dict)
    results = tool_results.get("results")
    assert isinstance(results, list) and results
    assert results[0].get("success") is False
    assert "Duplicate tool call blocked" in str(results[0].get("error") or "")


def test_loop_guard_dedupes_and_returns_cached_output_when_previous_call_succeeded() -> None:
    logic = ReActLogic(
        tools=[
            ToolDefinition(
                name="write_file",
                description="write",
                parameters={},
            )
        ],
        max_history_messages=-1,
        max_tokens=None,
    )
    workflow = create_react_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="react_agent",
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {
                "pending_tool_calls": [
                    {
                        "name": "write_file",
                        "arguments": {"file_path": "demo.txt", "content": "hello"},
                        "call_id": "1",
                    }
                ]
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    act = workflow.get_node("act")
    observe = workflow.get_node("observe")

    plan1 = act(run, _Ctx())
    assert plan1.next_node == "observe"
    assert plan1.effect is not None

    run.vars["_temp"]["tool_results"] = {
        "mode": "executed",
        "results": [
            {
                "call_id": "1",
                "name": "write_file",
                "success": True,
                "output": "✅ Successfully written to '/abs/demo.txt' (5 bytes, 1 lines)",
                "error": None,
            }
        ],
    }

    plan_observe = observe(run, _Ctx())
    assert plan_observe.next_node == "reason"

    run.vars["_temp"]["pending_tool_calls"] = [
        {
            "name": "write_file",
            "arguments": {"file_path": "demo.txt", "content": "hello"},
            "call_id": "1",
        }
    ]

    plan2 = act(run, _Ctx())
    assert plan2.next_node == "observe"
    assert plan2.effect is None

    tool_results = run.vars["_temp"].get("tool_results")
    assert isinstance(tool_results, dict)
    results = tool_results.get("results")
    assert isinstance(results, list) and results
    assert results[0].get("success") is True
    assert "Successfully written to" in str(results[0].get("output") or "")
    assert results[0].get("error") in (None, "")


def test_loop_guard_dedupes_repeated_read_only_calls_with_cached_output() -> None:
    logic = ReActLogic(
        tools=[
            ToolDefinition(
                name="list_files",
                description="list",
                parameters={},
            )
        ],
        max_history_messages=-1,
        max_tokens=None,
    )
    workflow = create_react_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="react_agent",
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {
                "pending_tool_calls": [
                    {
                        "name": "list_files",
                        "arguments": {"directory_path": ".", "pattern": "*", "head_limit": 3},
                        "call_id": "1",
                    }
                ]
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    act = workflow.get_node("act")
    observe = workflow.get_node("observe")

    plan1 = act(run, _Ctx())
    assert plan1.next_node == "observe"
    assert plan1.effect is not None

    run.vars["_temp"]["tool_results"] = {
        "mode": "executed",
        "results": [
            {
                "call_id": "1",
                "name": "list_files",
                "success": True,
                "output": "first",
                "error": None,
            }
        ],
    }
    observe(run, _Ctx())

    run.vars["_temp"]["pending_tool_calls"] = [
        {
            "name": "list_files",
            "arguments": {"directory_path": ".", "pattern": "*", "head_limit": 3},
            "call_id": "1",
        }
    ]
    plan2 = act(run, _Ctx())
    assert plan2.next_node == "observe"
    assert plan2.effect is not None

    run.vars["_temp"]["tool_results"] = {
        "mode": "executed",
        "results": [
            {
                "call_id": "1",
                "name": "list_files",
                "success": True,
                "output": "second",
                "error": None,
            }
        ],
    }
    observe(run, _Ctx())

    run.vars["_temp"]["pending_tool_calls"] = [
        {
            "name": "list_files",
            "arguments": {"directory_path": ".", "pattern": "*", "head_limit": 3},
            "call_id": "1",
        }
    ]
    plan3 = act(run, _Ctx())
    assert plan3.next_node == "observe"
    assert plan3.effect is None

    tool_results = run.vars["_temp"].get("tool_results")
    assert isinstance(tool_results, dict)
    results = tool_results.get("results")
    assert isinstance(results, list) and results
    assert results[0].get("success") is True
    assert results[0].get("output") == "second"
    assert results[0].get("error") in (None, "")
