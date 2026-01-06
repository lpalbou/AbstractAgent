from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


def test_review_parse_routes_to_act_when_verifier_emits_tool_calls() -> None:
    logic = ReActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})])
    wf = create_react_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="react_agent",
        status=RunStatus.RUNNING,
        current_node="review_parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": ["read_file"]},
            "_temp": {
                "final_answer": "I'll fix it.",
                "review_llm_response": {
                    "data": {
                        "complete": False,
                        "missing": ["need to inspect the file"],
                        "next_prompt": "",
                        "next_tool_calls": [{"name": "read_file", "arguments": {"path": "x"}}],
                    }
                },
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("review_parse")(run, _Ctx())
    assert plan.next_node == "act"
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list)
    assert pending and pending[0].get("name") == "read_file"


def test_review_parse_routes_to_reason_when_verifier_emits_next_prompt_only() -> None:
    logic = ReActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})])
    wf = create_react_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="react_agent",
        status=RunStatus.RUNNING,
        current_node="review_parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 2, "allowed_tools": ["read_file"]},
            "_temp": {
                "final_answer": "answer",
                "review_llm_response": {
                    "data": {
                        "complete": False,
                        "missing": ["x"],
                        "next_prompt": "Call read_file.",
                        "next_tool_calls": [],
                    }
                },
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    # New contract: "incomplete + no tool calls" is behaviorally invalid → re-ask reviewer once.
    plan1 = wf.get_node("review_parse")(run, _Ctx())
    assert plan1.next_node == "review"

    # If the reviewer remains unactionable, we fall back to prompting the agent back to reason.
    run.vars["_temp"]["review_llm_response"] = {
        "data": {
            "complete": False,
            "missing": ["x"],
            "next_prompt": "Call read_file.",
            "next_tool_calls": [],
        }
    }
    plan2 = wf.get_node("review_parse")(run, _Ctx())
    assert plan2.next_node == "reason"
    inbox = run.vars["_runtime"].get("inbox")
    assert isinstance(inbox, list)
    assert inbox and "[Review]" in str(inbox[-1].get("content", ""))


