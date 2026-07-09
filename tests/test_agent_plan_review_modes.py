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


def _run(*, vars: dict, current_node: str = "init") -> RunState:
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


def _base_vars(*, runtime_ns: dict | None = None) -> dict:
    return {
        "context": {"task": "t", "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": 2},
        "_runtime": dict({"inbox": []}, **(runtime_ns or {})),
        "_temp": {},
        "_limits": {
            "max_iterations": 2,
            "current_iteration": 0,
            "max_history_messages": -1,
            "max_tokens": 1024,
        },
    }


def test_react_workflow_has_verifier_gated_on_review_mode() -> None:
    """ReAct now wires the CodeAct-style verifier behind `_runtime.review_mode` (backlog 0217).

    Default (review_mode off): a final answer goes straight to `done`.
    review_mode on: a final answer is routed through the verifier (`maybe_review` → `review`).
    There is still no upfront `plan` node (planning is the model-invoked `update_plan` tool).
    """
    tool = ToolDefinition(name="tool_a", description="A", parameters={})
    wf = create_react_workflow(logic=ReActLogic(tools=[tool]), workflow_id="wf", allowed_tools=["tool_a"])

    # The verifier nodes exist; there is no separate upfront plan node.
    assert "maybe_review" in wf.nodes
    assert "review" in wf.nodes
    assert "review_parse" in wf.nodes
    assert "plan" not in wf.nodes

    # init always routes to reason.
    run = _run(vars=_base_vars(runtime_ns={"review_mode": True, "review_max_rounds": 1}))
    assert wf.get_node("init")(run, _Ctx()).next_node == "reason"

    # review_mode ON: maybe_review routes to the verifier.
    run_on = _run(vars=_base_vars(runtime_ns={"review_mode": True, "review_max_rounds": 1}))
    run_on.vars["_temp"]["final_answer"] = "answer"
    assert wf.get_node("maybe_review")(run_on, _Ctx()).next_node == "review"

    # review_mode OFF (default): maybe_review routes straight to done.
    run_off = _run(vars=_base_vars(runtime_ns={}))
    run_off.vars["_temp"]["final_answer"] = "answer"
    assert wf.get_node("maybe_review")(run_off, _Ctx()).next_node == "done"


def test_codeact_plan_mode_routes_to_plan_node() -> None:
    tool = ToolDefinition(name="execute_python", description="Exec", parameters={})
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[tool]))
    run = _run(vars=_base_vars(runtime_ns={"plan_mode": True}))
    init_plan = wf.get_node("init")(run, _Ctx())
    assert init_plan.next_node == "plan"

    plan_step = wf.get_node("plan")(run, _Ctx())
    assert plan_step.effect is not None
    assert plan_step.effect.type == EffectType.LLM_CALL
    payload = plan_step.effect.payload if isinstance(plan_step.effect.payload, dict) else {}
    assert "tools" not in payload


def test_codeact_review_mode_routes_to_reason_when_incomplete() -> None:
    tool = ToolDefinition(name="execute_python", description="Exec", parameters={})
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[tool]))
    run = _run(vars=_base_vars(runtime_ns={"review_mode": True, "review_max_rounds": 1}))
    run.vars["_temp"]["final_answer"] = "answer"

    maybe = wf.get_node("maybe_review")(run, _Ctx())
    assert maybe.next_node == "review"

    run.vars["_temp"]["review_llm_response"] = {
        "data": {"complete": False, "missing": ["x"], "next_prompt": "Do x next", "next_tool_calls": []},
    }
    # New contract: "incomplete + no tool calls" is behaviorally invalid → re-ask reviewer once.
    review_parse_1 = wf.get_node("review_parse")(run, _Ctx())
    assert review_parse_1.next_node == "review"

    # If the reviewer remains unactionable, we fall back to prompting the agent back to reason.
    run.vars["_temp"]["review_llm_response"] = {
        "data": {"complete": False, "missing": ["x"], "next_prompt": "Do x next", "next_tool_calls": []},
    }
    review_parse_2 = wf.get_node("review_parse")(run, _Ctx())
    assert review_parse_2.next_node == "reason"
    inbox = run.vars["_runtime"].get("inbox")
    assert isinstance(inbox, list)
    assert inbox and "[Review]" in str(inbox[-1].get("content", ""))
