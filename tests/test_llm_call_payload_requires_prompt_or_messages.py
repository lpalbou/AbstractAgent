from __future__ import annotations

from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import EffectType, RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2026-01-01T00:00:00+00:00"


def _has_prompt_or_messages(payload: dict) -> bool:
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return True
    messages = payload.get("messages")
    return isinstance(messages, list) and len(messages) > 0


def test_react_reason_llm_payload_includes_prompt_or_messages() -> None:
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[tool_a]),
        workflow_id="wf",
        allowed_tools=["tool_a"],
    )

    run = RunState(
        run_id="run",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="init",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 0, "max_iterations": 2},
            "_runtime": {"inbox": []},
            "_temp": {},
            "_limits": {
                "max_iterations": 2,
                "current_iteration": 0,
                "max_history_messages": -1,
                "max_tokens": 1024,
            },
        },
    )

    workflow.get_node("init")(run, _Ctx())
    plan = workflow.get_node("reason")(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL

    payload = dict(plan.effect.payload or {})
    assert _has_prompt_or_messages(payload)


def test_memact_reason_llm_payload_includes_prompt_or_messages() -> None:
    tool_a = ToolDefinition(name="list_files", description="list", parameters={})
    workflow = create_memact_workflow(logic=MemActLogic(tools=[tool_a]), on_step=None)

    run = RunState(
        run_id="run",
        workflow_id="memact_agent",
        status=RunStatus.RUNNING,
        current_node="reason",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 0, "max_iterations": 2},
            "_runtime": {"active_memory": {"version": 1, "persona": "p"}},
            "_temp": {},
            "_limits": {"max_history_messages": -1, "max_tokens": 1024, "max_iterations": 2, "current_iteration": 0},
        },
    )

    plan = workflow.get_node("reason")(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL

    payload = dict(plan.effect.payload or {})
    assert _has_prompt_or_messages(payload)
