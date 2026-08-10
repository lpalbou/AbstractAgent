from __future__ import annotations

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.react_runtime import _render_cycles_for_conclusion_prompt
from abstractagent.logic.codeact import CodeActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import RunState, RunStatus


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


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


def test_codeact_plan_tail_flows_whole_when_cap_unset() -> None:
    wf = create_codeact_workflow(
        logic=CodeActLogic(tools=[ToolDefinition(name="execute_python", description="Exec", parameters={})])
    )
    plan_text = "plan step\n" * 1200
    vars = _base_vars(runtime_ns={"plan_mode": True})
    vars["scratchpad"]["plan"] = plan_text
    run = _run(vars=vars)

    step = wf.get_node("reason")(run, _Ctx())
    payload = step.effect.payload if step.effect is not None and isinstance(step.effect.payload, dict) else {}
    rendered = "\n".join(
        str(msg.get("content") or "")
        for msg in (payload.get("messages") or [])
        if isinstance(msg, dict)
    )

    assert plan_text.strip() in rendered
    assert "_limits.plan_render_max_chars" not in rendered
    assert "#[WARNING:TRUNCATION]" not in rendered


def test_codeact_review_keeps_all_tool_outputs_when_windows_unset() -> None:
    wf = create_codeact_workflow(
        logic=CodeActLogic(tools=[ToolDefinition(name="execute_python", description="Exec", parameters={})])
    )
    vars = _base_vars()
    vars["scratchpad"]["plan"] = "check the evidence"
    vars["context"]["messages"] = [
        {"role": "tool", "content": f"tool-output-{idx}: {'x' * 400}"}
        for idx in range(10)
    ]
    vars["_temp"]["final_answer"] = "answer"
    run = _run(vars=vars, current_node="review")

    step = wf.get_node("review")(run, _Ctx())
    payload = step.effect.payload if step.effect is not None and isinstance(step.effect.payload, dict) else {}
    prompt = str(payload.get("prompt") or "")

    for idx in range(10):
        assert f"tool-output-{idx}:" in prompt
    assert "older tool outputs omitted by the caller-set" not in prompt


def test_react_conclusion_render_shows_full_trace_when_limits_unset() -> None:
    scratchpad = {
        "cycles": [
            {
                "i": idx,
                "thought": f"thought-{idx}: " + ("t" * 500),
                "tool_calls": [{"name": "read_file", "arguments": {"path": f"file-{idx}.txt"}}],
                "observations": [
                    {"name": "read_file", "success": True, "output": f"observation-{idx}: " + ("o" * 400)}
                ],
            }
            for idx in range(30)
        ]
    }

    rendered = _render_cycles_for_conclusion_prompt(scratchpad, limits={"max_iterations": 2})

    assert "[cycle 0]" in rendered
    assert "[cycle 29]" in rendered
    assert "thought-0:" in rendered
    assert "thought-29:" in rendered
    assert "observation-0:" in rendered
    assert "observation-29:" in rendered
    assert "showing last" not in rendered
    assert "_limits.conclusion_max_cycles" not in rendered
