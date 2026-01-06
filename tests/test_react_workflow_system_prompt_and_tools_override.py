from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import EffectType, RunState, RunStatus


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


def _base_vars(*, allowed_tools: list[str] | None = None, system_prompt: str | None = None) -> dict:
    runtime_ns: dict = {"inbox": []}
    if allowed_tools is not None:
        runtime_ns["allowed_tools"] = allowed_tools
    if system_prompt is not None:
        runtime_ns["system_prompt"] = system_prompt

    return {
        "context": {"task": "t", "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": 2},
        "_runtime": runtime_ns,
        "_temp": {},
        "_limits": {
            "max_iterations": 2,
            "current_iteration": 0,
            "max_history_messages": -1,
            "max_tokens": 1024,
        },
    }


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


def test_react_workflow_runtime_system_prompt_and_tools_override() -> None:
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    tool_b = ToolDefinition(name="tool_b", description="B", parameters={})

    # Workflow default allowlist is tool_a only.
    wf = create_react_workflow(
        logic=ReActLogic(tools=[tool_a, tool_b]),
        workflow_id="wf",
        allowed_tools=["tool_a"],
    )

    # Runtime vars override allowlist to tool_b and add a system prompt.
    run = _run(vars=_base_vars(allowed_tools=["tool_b"], system_prompt="SYS"))

    init_node = wf.get_node("init")
    init_plan = init_node(run, _Ctx())
    assert init_plan.next_node == "reason"

    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL

    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    assert payload.get("system_prompt") == "SYS"

    tools_payload = payload.get("tools")
    assert isinstance(tools_payload, list)
    tool_names = [t.get("name") for t in tools_payload if isinstance(t, dict)]
    assert tool_names == ["tool_b"]

    # Act node should pass the effective allowlist through to TOOL_CALLS.
    temp = run.vars.setdefault("_temp", {})
    assert isinstance(temp, dict)
    temp["pending_tool_calls"] = [{"name": "tool_b", "arguments": {}, "call_id": "call_1"}]

    act_node = wf.get_node("act")
    act_plan = act_node(run, _Ctx())
    assert act_plan.effect is not None
    assert act_plan.effect.type == EffectType.TOOL_CALLS
    act_payload = act_plan.effect.payload if isinstance(act_plan.effect.payload, dict) else {}
    assert act_payload.get("allowed_tools") == ["tool_b"]


def test_react_workflow_can_disable_tool_examples_in_payload() -> None:
    tool_a = ToolDefinition(
        name="tool_a",
        description="A",
        parameters={"x": {"type": "string"}},
        examples=[{"description": "Example", "arguments": {"x": "y"}}],
    )

    wf = create_react_workflow(
        logic=ReActLogic(tools=[tool_a]),
        workflow_id="wf",
        allowed_tools=["tool_a"],
    )

    vars = _base_vars(allowed_tools=["tool_a"])
    runtime_ns = vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    runtime_ns["tool_prompt_examples"] = False

    run = _run(vars=vars)
    wf.get_node("init")(run, _Ctx())

    plan = wf.get_node("reason")(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL

    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    tools_payload = payload.get("tools")
    assert isinstance(tools_payload, list)
    assert tools_payload
    assert "examples" not in tools_payload[0]

    runtime_ns2 = run.vars.get("_runtime")
    assert isinstance(runtime_ns2, dict)
    specs = runtime_ns2.get("tool_specs")
    assert isinstance(specs, list)
    assert specs
    assert "examples" not in specs[0]
