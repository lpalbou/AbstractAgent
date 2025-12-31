from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


def test_parse_node_strips_fabricated_observation_lines_when_tool_calls_present() -> None:
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
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {
                "llm_response": {
                    "content": (
                        "Some explanation.\n"
                        "observation[read_file] (success): File: demo.py\n"
                        "\n"
                        "Action:\n"
                        '[{"name":"edit_file","arguments":{"file_path":"demo.py","pattern":"a","replacement":"b"}}]\n'
                    ),
                    "tool_calls": [{"name": "edit_file", "arguments": {"file_path": "demo.py", "pattern": "a", "replacement": "b"}}],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    handler = workflow.get_node("parse")
    plan = handler(run, _Ctx())

    assert plan.next_node == "act"

    msgs = run.vars["context"]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"].strip() == "Some explanation."
    assert "observation[" not in msgs[0]["content"].lower()
    assert "action:" not in msgs[0]["content"].lower()

