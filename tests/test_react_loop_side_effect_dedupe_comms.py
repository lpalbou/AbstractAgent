from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import RunState, RunStatus


class _Ctx:
    def now_iso(self) -> str:
        return "2026-02-04T00:00:00+00:00"


def _run(*, vars: dict, current_node: str = "parse") -> RunState:
    return RunState(
        run_id="run",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node=current_node,
        vars=vars,
        waiting=None,
        output=None,
        error=None,
        created_at="2026-02-04T00:00:00+00:00",
        updated_at="2026-02-04T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )


def test_react_parse_skips_repeated_send_email_after_success() -> None:
    send_email = ToolDefinition(
        name="send_email",
        description="Send email",
        parameters={
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body_text": {"type": "string"},
        },
    )

    wf = create_react_workflow(logic=ReActLogic(tools=[send_email]), workflow_id="wf", allowed_tools=["send_email"])

    args = {"to": "you@example.com", "subject": "Hello", "body_text": "Hi"}

    vars = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {
            "iteration": 2,
            "max_iterations": 4,
            "cycles": [
                {
                    "i": 1,
                    "thought": "",
                    "tool_calls": [{"name": "send_email", "arguments": dict(args)}],
                    "observations": [{"name": "send_email", "success": True, "output": {"success": True}}],
                }
            ],
        },
        "_runtime": {"inbox": [], "allowed_tools": ["send_email"], "tool_specs": [send_email.to_dict()]},
        "_temp": {
            "llm_response": {
                "content": "",
                "tool_calls": [{"name": "send_email", "arguments": dict(args)}],
            }
        },
        "_limits": {"max_iterations": 4, "current_iteration": 2},
    }
    run = _run(vars=vars)

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"

    temp = run.vars.get("_temp")
    assert isinstance(temp, dict)
    assert temp.get("pending_tool_calls") == []
