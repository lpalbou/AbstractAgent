from __future__ import annotations

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import EffectType, RunState, RunStatus
from abstractruntime.memory.active_memory import upsert_task


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


def _base_vars(*, task: str) -> dict:
    return {
        "context": {"task": task, "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": 2},
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {
            "max_iterations": 2,
            "current_iteration": 0,
            "max_history_messages": -1,
            "max_tokens": 2048,
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


def test_react_reason_prompt_includes_active_memory_and_default_system_prompt() -> None:
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    logic = ReActLogic(tools=[tool_a])
    wf = create_react_workflow(logic=logic, workflow_id="wf", allowed_tools=["tool_a"])

    run = _run(vars=_base_vars(task="t"))
    upsert_task(run.vars, title="do something", now_iso=lambda: "2025-01-01T00:00:01+00:00")
    init_node = wf.get_node("init")
    init_plan = init_node(run, _Ctx())
    assert init_plan.next_node == "reason"

    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}

    prompt = str(payload.get("prompt") or "")
    # User-role prompt must contain only the user request.
    assert prompt == "t"
    assert "Active Memory:" not in prompt
    assert "## Current Tasks (evolving)" not in prompt

    sys = payload.get("system_prompt")
    assert isinstance(sys, str) and "autonomous ReAct agent" in sys
    assert "## Active Memory (internal)" in sys
    assert "## Persona (persistent)" in sys
    assert "## Current Tasks (evolving)" in sys


def test_codeact_reason_prompt_includes_active_memory_and_default_system_prompt() -> None:
    execute_python = ToolDefinition(name="execute_python", description="run python", parameters={})
    logic = CodeActLogic(tools=[execute_python])
    wf = create_codeact_workflow(logic=logic)

    run = _run(vars=_base_vars(task="t"))
    upsert_task(run.vars, title="do something", now_iso=lambda: "2025-01-01T00:00:01+00:00")
    init_node = wf.get_node("init")
    init_plan = init_node(run, _Ctx())
    assert init_plan.next_node == "reason"

    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}

    prompt = str(payload.get("prompt") or "")
    assert prompt == "t"
    assert "Active Memory:" not in prompt
    assert "## Current Tasks (evolving)" not in prompt

    sys = payload.get("system_prompt")
    assert isinstance(sys, str) and "You are CodeAct" in sys
    assert "## Active Memory (internal)" in sys
    assert "## Persona (persistent)" in sys
    assert "## Current Tasks (evolving)" in sys


def test_react_system_prompt_omits_tools_session_when_native_tools_are_enabled() -> None:
    """Native tool calling should not duplicate a visible tools catalog in the system prompt."""
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    logic = ReActLogic(tools=[tool_a])
    wf = create_react_workflow(
        logic=logic,
        workflow_id="wf",
        provider="lmstudio",
        model="qwen/qwen3-next-80b",
        allowed_tools=["tool_a"],
    )

    run = _run(vars=_base_vars(task="t"))
    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}

    sys = payload.get("system_prompt")
    assert isinstance(sys, str) and sys.strip()
    assert "## Persona (persistent)" in sys
    assert "## Tools (session)" not in sys


def test_react_system_prompt_omits_tools_session_when_native_tools_are_enabled_without_provider() -> None:
    """Even if `_runtime.provider` is unset, native-tool models must not get Tools(session) injected."""
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    logic = ReActLogic(tools=[tool_a])
    wf = create_react_workflow(
        logic=logic,
        workflow_id="wf",
        # Provider intentionally omitted (some hosts resolve it outside the workflow).
        model="qwen/qwen3-next-80b",
        allowed_tools=["tool_a"],
    )

    run = _run(vars=_base_vars(task="t"))
    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}

    sys = payload.get("system_prompt")
    assert isinstance(sys, str) and sys.strip()
    assert "## Persona (persistent)" in sys
    assert "## Tools (session)" not in sys


def test_react_system_prompt_omits_tools_session_when_native_tools_are_enabled_via_runtime_model() -> None:
    """ReAct workflows typically don't know the model; they must rely on `_runtime.model`."""
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    logic = ReActLogic(tools=[tool_a])
    wf = create_react_workflow(
        logic=logic,
        workflow_id="wf",
        allowed_tools=["tool_a"],
    )

    run = _run(vars=_base_vars(task="t"))
    run.vars.setdefault("_runtime", {})["model"] = "qwen/qwen3-next-80b"

    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}

    sys = payload.get("system_prompt")
    assert isinstance(sys, str) and sys.strip()
    assert "## Persona (persistent)" in sys
    assert "## Tools (session)" not in sys


def test_codeact_system_prompt_omits_tools_session_when_native_tools_are_enabled_without_provider() -> None:
    """CodeAct should apply the same native-tools policy even when provider is resolved externally."""
    execute_python = ToolDefinition(name="execute_python", description="run python", parameters={})
    logic = CodeActLogic(tools=[execute_python])
    wf = create_codeact_workflow(logic=logic)

    run = _run(vars=_base_vars(task="t"))
    # CodeAct workflow reads model from durable runtime vars (no closure model arg).
    run.vars.setdefault("_runtime", {})["model"] = "qwen/qwen3-next-80b"

    reason_node = wf.get_node("reason")
    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}

    sys = payload.get("system_prompt")
    assert isinstance(sys, str) and sys.strip()
    assert "## Persona (persistent)" in sys
    assert "## Tools (session)" not in sys
