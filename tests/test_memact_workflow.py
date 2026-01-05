from __future__ import annotations

from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.logic.memact import MemActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2026-01-05T06:00:00+00:00"


def test_memact_reason_injects_memory_blocks_into_system_prompt() -> None:
    logic = MemActLogic(
        tools=[
            ToolDefinition(name="list_files", description="list", parameters={}),
        ]
    )
    workflow = create_memact_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="memact_agent",
        status=RunStatus.RUNNING,
        current_node="reason",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"iteration": 0, "max_iterations": 3},
            "_runtime": {"active_memory": {"version": 1, "persona": "p"}},
            "_temp": {},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768, "max_iterations": 3, "current_iteration": 0},
        },
    )

    handler = workflow.get_node("reason")
    plan = handler(run, _Ctx())
    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    system_prompt = str(payload.get("system_prompt") or "")
    assert "## MEMORY BLUEPRINTS" in system_prompt
    assert "## MY PERSONA" in system_prompt


def test_memact_finalize_parse_applies_envelope_and_sets_final_answer() -> None:
    logic = MemActLogic(tools=[])
    workflow = create_memact_workflow(logic=logic, on_step=None)

    run = RunState(
        run_id="r1",
        workflow_id="memact_agent",
        status=RunStatus.RUNNING,
        current_node="finalize_parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1, "max_iterations": 3},
            "_runtime": {"active_memory": {"version": 1, "persona": "p"}},
            "_temp": {
                "finalize_llm_response": {
                    "data": {
                        "content": "All done.",
                        "relationships": {"added": [], "removed": []},
                        "current_tasks": {"added": ["Ship MemAct"], "removed": []},
                        "current_context": {"added": [], "removed": []},
                        "critical_insights": {"added": [], "removed": []},
                        "references": {"added": [], "removed": []},
                        "history": {"added": ["I implemented MemAct finalize parsing."]},
                    }
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768, "max_iterations": 3, "current_iteration": 1},
        },
    )

    handler = workflow.get_node("finalize_parse")
    plan = handler(run, _Ctx())
    assert plan.next_node == "done"

    temp = run.vars.get("_temp")
    assert isinstance(temp, dict)
    assert temp.get("final_answer") == "All done."

    mem = run.vars.get("_runtime", {}).get("active_memory", {})
    assert isinstance(mem, dict)
    assert any(
        isinstance(x, dict) and x.get("text") == "Ship MemAct" for x in (mem.get("current_tasks") or [])
    )
    assert any(
        isinstance(x, dict) and x.get("text") == "I implemented MemAct finalize parsing."
        for x in (mem.get("history") or [])
    )

