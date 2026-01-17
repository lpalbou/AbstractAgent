from __future__ import annotations

from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2026-01-17T00:00:00Z"


def test_react_observe_emits_tool_result() -> None:
    events: list[tuple[str, dict]] = []

    def on_step(step: str, data: dict) -> None:
        events.append((step, dict(data)))

    logic = ReActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})])
    workflow = create_react_workflow(logic=logic, on_step=on_step)

    run = RunState(
        run_id="r1",
        workflow_id="react_agent",
        status=RunStatus.RUNNING,
        current_node="observe",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 0, "max_iterations": 3, "cycles": []},
            "_runtime": {},
            "_temp": {
                "tool_results": {
                    "results": [
                        {
                            "call_id": "c1",
                            "name": "read_file",
                            "success": True,
                            "output": "File: demo.txt (1 lines)\n\n1: hello",
                            "error": None,
                        }
                    ]
                }
            },
            "_limits": {"max_iterations": 3, "current_iteration": 0, "max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = workflow.get_node("observe")(run, _Ctx())
    assert plan.node_id == "observe"

    obs = [d for (s, d) in events if s == "observe"]
    assert obs, "Expected on_step('observe', ...) to be emitted"
    assert isinstance(obs[0].get("result"), str)
    assert obs[0]["result"].startswith("[read_file]:")


def test_memact_observe_emits_tool_result() -> None:
    events: list[tuple[str, dict]] = []

    def on_step(step: str, data: dict) -> None:
        events.append((step, dict(data)))

    logic = MemActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})])
    workflow = create_memact_workflow(logic=logic, on_step=on_step)

    run = RunState(
        run_id="r1",
        workflow_id="memact_agent",
        status=RunStatus.RUNNING,
        current_node="observe",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 0, "max_iterations": 3},
            "_runtime": {},
            "_temp": {
                "tool_results": {
                    "results": [
                        {
                            "call_id": "c1",
                            "name": "read_file",
                            "success": True,
                            "output": "File: demo.txt (1 lines)\n\n1: hello",
                            "error": None,
                        }
                    ]
                }
            },
            "_limits": {"max_iterations": 3, "current_iteration": 0, "max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = workflow.get_node("observe")(run, _Ctx())
    assert plan.node_id == "observe"

    obs = [d for (s, d) in events if s == "observe"]
    assert obs, "Expected on_step('observe', ...) to be emitted"
    assert isinstance(obs[0].get("result"), str)
    assert obs[0]["result"].startswith("[read_file]:")

