"""The loop must tell the model when its budget is nearly gone.

Run 9ea71c55-4405-4b6f-b387-3cc78629880b read `[loop] iteration 19 of 20` in
its prompt tail and still spent iterations 19 and 20 on diagnosis, finishing
with a correct one-character diagnosis it never applied. The position line is
information; it was never a steer. `_limits.warn_iterations_pct` declared where
"nearly out" begins and had no behavioural consequence anywhere.
"""

from __future__ import annotations

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_READ = ToolDefinition(name="read_file", description="read", parameters={})


def _run(*, iteration_done: int, max_iterations: int, warn_pct: int = 80) -> RunState:
    """A run poised to enter `reason` for iteration `iteration_done + 1`.

    `context.messages` is seeded so the payload is message-shaped — the loop
    tail only rides a `messages` payload, which is the shape every real
    tool-calling turn has (verified in the incident ledger).
    """
    return RunState(
        run_id="budget-tail",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="reason",
        vars={
            "context": {
                "task": "fix the parse error",
                "messages": [{"role": "user", "content": "fix the parse error"}],
            },
            "scratchpad": {"iteration": iteration_done, "cycles": []},
            "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
            "_temp": {},
            "_limits": {
                "max_iterations": max_iterations,
                "current_iteration": iteration_done,
                "warn_iterations_pct": warn_pct,
                "max_history_messages": -1,
                "max_tokens": 32768,
            },
        },
    )


def _tail_of(run: RunState) -> str:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["read_file"],
    )
    plan = wf.get_node("reason")(run, _Ctx())
    assert plan.effect is not None, "reason must issue an LLM call"
    messages = plan.effect.payload.get("messages") or []
    return "\n".join(str(m.get("content") or "") for m in messages)


def test_no_budget_line_while_the_budget_is_comfortable() -> None:
    tail = _tail_of(_run(iteration_done=4, max_iterations=20))
    assert "[loop] iteration 5 of 20" in tail
    assert "[budget]" not in tail, "an early iteration must not be nagged"


def test_budget_line_appears_at_the_declared_warn_threshold() -> None:
    # warn_iterations_pct=80 of 20 -> from iteration 16 onwards.
    comfortable = _tail_of(_run(iteration_done=14, max_iterations=20))
    assert "[budget]" not in comfortable

    warned = _tail_of(_run(iteration_done=15, max_iterations=20))
    assert "[loop] iteration 16 of 20" in warned
    assert "[budget] 4 iteration(s) left" in warned
    assert "apply it now" in warned


def test_the_exact_shape_of_the_incident_iteration_19_of_20() -> None:
    """The iteration that had a correct diagnosis and no instruction to land it."""
    tail = _tail_of(_run(iteration_done=18, max_iterations=20))
    assert "[loop] iteration 19 of 20" in tail
    assert "[budget] 1 iteration(s) left" in tail
    assert "the turn ENDS" in tail


def test_last_iteration_is_told_to_stop_investigating() -> None:
    tail = _tail_of(_run(iteration_done=19, max_iterations=20))
    assert "[loop] iteration 20 of 20" in tail
    assert "This is your LAST iteration" in tail
    assert "Do not start new investigation" in tail


def test_threshold_follows_the_declared_percentage() -> None:
    # A caller who declares 50% gets warned from the halfway point.
    half = _tail_of(_run(iteration_done=9, max_iterations=20, warn_pct=50))
    assert "[budget] 10 iteration(s) left" in half

    # ...and the same iteration under the default 80% is not warned.
    default = _tail_of(_run(iteration_done=9, max_iterations=20))
    assert "[budget]" not in default


def test_visit_lane_gets_no_loop_chrome_at_all() -> None:
    """c2447: loop vocabulary must stay out of an entity visit."""
    run = _run(iteration_done=19, max_iterations=20)
    run.vars["_runtime"]["suppress_loop_tail"] = True
    tail = _tail_of(run)
    assert "[budget]" not in tail
    assert "[loop]" not in tail
