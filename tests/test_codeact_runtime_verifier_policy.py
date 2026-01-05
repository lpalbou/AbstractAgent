from __future__ import annotations

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


def test_review_parse_routes_to_act_when_verifier_emits_tool_calls() -> None:
    logic = CodeActLogic(tools=[ToolDefinition(name="edit_file", description="edit", parameters={})])
    wf = create_codeact_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="review_parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": ["edit_file"]},
            "_temp": {
                "final_answer": "I'll fix it.",
                "review_llm_response": {
                    "data": {
                        "complete": False,
                        "missing": ["need to edit file"],
                        "next_prompt": "",
                        "next_tool_calls": [{"name": "edit_file", "arguments": {"path": "x"}}],
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
    assert pending and pending[0].get("name") == "edit_file"


