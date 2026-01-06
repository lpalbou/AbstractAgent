from __future__ import annotations

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import EffectType, RunState, RunStatus


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


def _run(*, vars: dict, current_node: str) -> RunState:
    return RunState(
        run_id="run",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node=current_node,
        vars=vars,
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )


def test_react_review_prompt_avoids_observation_vocabulary() -> None:
    tool = ToolDefinition(name="list_files", description="list", parameters={})
    wf = create_react_workflow(logic=ReActLogic(tools=[tool]), workflow_id="wf")
    run = _run(
        current_node="review",
        vars={
            "context": {
                "task": "t",
                "messages": [
                    {"role": "tool", "content": "[list_files]: ok", "metadata": {"name": "list_files", "success": True}}
                ],
            },
            "scratchpad": {"iteration": 1, "max_iterations": 2, "plan": "p"},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1},
            "_temp": {"final_answer": "answer"},
            "_limits": {"max_history_messages": -1, "max_tokens": 1024},
        },
    )

    step = wf.get_node("review")(run, _Ctx())
    assert step.effect is not None
    assert step.effect.type == EffectType.LLM_CALL
    payload = step.effect.payload if isinstance(step.effect.payload, dict) else {}
    prompt = str(payload.get("prompt") or "")

    assert "Observations (tool outputs)" not in prompt
    assert "supported by Observations" not in prompt
    assert "Tool outputs:" in prompt


def test_codeact_review_prompt_avoids_observation_vocabulary() -> None:
    tool = ToolDefinition(name="execute_python", description="exec", parameters={})
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[tool]))
    run = _run(
        current_node="review",
        vars={
            "context": {
                "task": "t",
                "messages": [
                    {
                        "role": "tool",
                        "content": "[execute_python]: ok",
                        "metadata": {"name": "execute_python", "success": True},
                    }
                ],
            },
            "scratchpad": {"iteration": 1, "max_iterations": 2, "plan": "p"},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1},
            "_temp": {"final_answer": "answer"},
            "_limits": {"max_history_messages": -1, "max_tokens": 1024},
        },
    )

    step = wf.get_node("review")(run, _Ctx())
    assert step.effect is not None
    assert step.effect.type == EffectType.LLM_CALL
    payload = step.effect.payload if isinstance(step.effect.payload, dict) else {}
    prompt = str(payload.get("prompt") or "")

    assert "Observations (tool outputs)" not in prompt
    assert "supported by Observations" not in prompt
    assert "Tool outputs:" in prompt
