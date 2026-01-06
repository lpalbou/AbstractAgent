from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


def test_parse_node_does_not_retry_on_observation_echo_after_tools_used() -> None:
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
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"used_tools": True},
            "_runtime": {},
            "_temp": {
                "llm_response": {
                    "content": (
                            "FINAL: Files listed above.\n\n"
                            "observation[list_files] (success): Entries in '/tmp' matching '*':\n"
                        "  a.txt\n"
                        "  b.txt\n\n"
                        "Files listed above."
                    ),
                    "tool_calls": [],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    handler = workflow.get_node("parse")
    plan = handler(run, _Ctx())
    assert plan.next_node == "maybe_review"

    msgs = run.vars["context"]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "Files listed above."


def test_parse_node_retries_on_observation_echo_when_no_tools_used() -> None:
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
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {
                "llm_response": {
                    "content": "observation[list_files] (success): Entries: a.txt",
                    "tool_calls": [],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    handler = workflow.get_node("parse")
    plan = handler(run, _Ctx())
    assert plan.next_node == "reason"
    assert run.vars["context"]["messages"] == []


def test_parse_node_does_not_retry_on_observation_substring_in_json() -> None:
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
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {
                "llm_response": {
                    "content": (
                        "FINAL: I will answer directly.\n\n"
                        "```json\n"
                        "{\n"
                        "  \"ref\": \"observation[list_files] (success): Entries: a.txt\"\n"
                        "}\n"
                        "```\n"
                    ),
                    "tool_calls": [],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    handler = workflow.get_node("parse")
    plan = handler(run, _Ctx())
    assert plan.next_node == "maybe_review"
    msgs = run.vars["context"]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
