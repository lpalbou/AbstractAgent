from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import EffectType, RunState, RunStatus


class _Ctx:
    def now_iso(self) -> str:
        return "2026-02-04T00:00:00+00:00"


def _run(*, vars: dict, current_node: str = "reason") -> RunState:
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


def test_react_reason_refreshes_tool_specs_on_resume_when_allowlist_changes() -> None:
    tool_a = ToolDefinition(name="tool_a", description="A", parameters={})
    tool_b = ToolDefinition(name="tool_b", description="B", parameters={})

    wf = create_react_workflow(logic=ReActLogic(tools=[tool_a, tool_b]), workflow_id="wf")

    vars = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": 2, "cycles": []},
        "_runtime": {
            "inbox": [],
            "allowed_tools": ["tool_b"],
            # Stale from a previous run/version/config.
            "tool_specs": [tool_a.to_dict()],
            "toolset_id": "ts_stale",
        },
        "_temp": {},
        "_limits": {
            "max_iterations": 2,
            "current_iteration": 0,
            "max_history_messages": -1,
            "max_tokens": 1024,
        },
    }
    run = _run(vars=vars)

    plan = wf.get_node("reason")(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL

    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    tools_payload = payload.get("tools")
    assert isinstance(tools_payload, list)
    tool_names = [t.get("name") for t in tools_payload if isinstance(t, dict)]
    assert tool_names == ["tool_b"]

    runtime_ns = run.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    specs = runtime_ns.get("tool_specs")
    assert isinstance(specs, list)
    spec_names = [s.get("name") for s in specs if isinstance(s, dict)]
    assert spec_names == ["tool_b"]
