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

