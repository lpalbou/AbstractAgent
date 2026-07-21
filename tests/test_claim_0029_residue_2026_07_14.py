"""Pins for the 0029 audit-residue claim (2026-07-14, standing-order slice).

- #6: an EXPLICIT legacy/flat `max_iterations` beats the materialized
  `_limits` default (20) at the resolver — raw create_*_workflow callers'
  budgets were silently capped; the ruled 20 stays the no-value default and
  a caller-set `_limits.max_iterations` stays untouched.
- #10: sibling budget terminals answer with the agent's LAST WORDS (or the
  held draft for MemAct), never a raw tool observation; the transcript ends
  with a final assistant message marked budget_exhausted.
"""
from __future__ import annotations

from typing import Any, Dict

from abstractagent.adapters.codeact_runtime import create_codeact_workflow, ensure_codeact_vars
from abstractagent.adapters.generation_params import resolve_max_iterations
from abstractagent.adapters.memact_runtime import create_memact_workflow, ensure_memact_vars
from abstractagent.adapters.react_runtime import ensure_react_vars
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_EXEC = ToolDefinition(name="execute_python", description="run", parameters={})


def _run(vars: Dict[str, Any], node: str = "init", wf_id: str = "react_agent") -> RunState:
    return RunState(run_id="r", workflow_id=wf_id, status=RunStatus.RUNNING, current_node=node, vars=vars)


def test_explicit_flat_budget_beats_materialized_default_all_loops() -> None:
    """0029 #6: runtime.start(vars={'max_iterations': 100}) must run at 100."""
    for ensure in (ensure_react_vars, ensure_codeact_vars, ensure_memact_vars):
        run = _run({"task": "t", "max_iterations": 100})
        _, scratchpad, _, _, limits = ensure(run)
        assert scratchpad["max_iterations"] == 100, ensure.__name__
        assert limits["max_iterations"] == 100, ensure.__name__
        assert resolve_max_iterations(limits, scratchpad) == 100, ensure.__name__


def test_explicit_scratchpad_budget_beats_materialized_default() -> None:
    for ensure in (ensure_react_vars, ensure_codeact_vars, ensure_memact_vars):
        run = _run({"task": "t", "scratchpad": {"max_iterations": 7}})
        _, scratchpad, _, _, limits = ensure(run)
        assert resolve_max_iterations(limits, scratchpad) == 7, ensure.__name__


def test_caller_set_limits_budget_is_never_overridden_by_legacy() -> None:
    """Facade-style callers own _limits: a stale flat key must not clobber it."""
    for ensure in (ensure_react_vars, ensure_codeact_vars, ensure_memact_vars):
        run = _run({"task": "t", "max_iterations": 100, "_limits": {"max_iterations": 5}})
        _, scratchpad, _, _, limits = ensure(run)
        assert limits["max_iterations"] == 5, ensure.__name__
        assert resolve_max_iterations(limits, scratchpad) == 5, ensure.__name__


def test_no_value_default_stays_ruled_20() -> None:
    for ensure in (ensure_react_vars, ensure_codeact_vars, ensure_memact_vars):
        run = _run({"task": "t"})
        _, scratchpad, _, _, limits = ensure(run)
        assert resolve_max_iterations(limits, scratchpad) == 20, ensure.__name__


def _budget_exhausted_vars(tail_messages: list) -> Dict[str, Any]:
    return {
        "context": {"task": "t", "messages": tail_messages},
        "scratchpad": {"iteration": 3, "max_iterations": 3},
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {"max_iterations": 3, "current_iteration": 3},
    }


def test_codeact_budget_terminal_answers_with_last_assistant_words() -> None:
    """0029 #10: a tool-observation tail must not become the 'answer'."""
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    msgs = [
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "I found the bug in parse()."},
        {"role": "user", "content": "[Code execution result]: {'stdout': 'traceback...'}"},
    ]
    run = _run(_budget_exhausted_vars(msgs), node="max_iterations", wf_id="codeact_agent")
    plan = wf.get_node("max_iterations")(run, _Ctx())
    out = plan.complete_output
    assert out["answer"] == "I found the bug in parse()."
    assert out["outcome"] == "iteration_budget"
    # Transcript ends with the answer as a final assistant message.
    last = out["messages"][-1]
    assert last["role"] == "assistant" and last["content"] == out["answer"]
    assert last["metadata"]["kind"] == "final_answer" and last["metadata"]["budget_exhausted"] is True


def test_memact_budget_terminal_prefers_held_draft() -> None:
    wf = create_memact_workflow(logic=MemActLogic(tools=[_EXEC]))
    msgs = [
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "working on it"},
        {"role": "user", "content": "[Tool results]: ..."},
    ]
    vars = _budget_exhausted_vars(msgs)
    vars["_temp"]["draft_answer"] = "The draft answer held before finalize."
    run = _run(vars, node="max_iterations", wf_id="memact_agent")
    plan = wf.get_node("max_iterations")(run, _Ctx())
    out = plan.complete_output
    assert out["answer"] == "The draft answer held before finalize."
    assert out["messages"][-1]["content"] == out["answer"]


def test_budget_terminal_honest_when_no_words_exist() -> None:
    """No assistant words at all -> an honest statement, not a fabricated one."""
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    msgs = [{"role": "user", "content": "t"}]
    run = _run(_budget_exhausted_vars(msgs), node="max_iterations", wf_id="codeact_agent")
    plan = wf.get_node("max_iterations")(run, _Ctx())
    assert "without producing an answer" in plan.complete_output["answer"]


def test_memact_compose_non_dict_kg_result_fails_forward_not_forever() -> None:
    """0029 #12: a custom host handler returning a non-dict used to make
    compose re-issue the IDENTICAL query forever (idempotency replays it
    instantly, consuming no budget). Present-but-non-dict = failed compose."""
    steps: list = []
    wf = create_memact_workflow(
        logic=MemActLogic(tools=[_EXEC]),
        on_step=lambda s, d: steps.append((s, d)),
    )
    vars: Dict[str, Any] = {
        "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
        "scratchpad": {"iteration": 0, "max_iterations": 5},
        "_runtime": {"inbox": [], "memact_composer": {"enabled": True}},
        "_temp": {"memact_composer": {"kg_result": "NOT A DICT", "pending_key": "k"}},
        "_limits": {"max_iterations": 5, "current_iteration": 0},
    }
    run = _run(vars, node="compose", wf_id="memact_agent")
    plan = wf.get_node("compose")(run, _Ctx())
    # Continues to reason with NO re-issued effect (the spin).
    assert plan.next_node == "reason"
    assert plan.effect is None
    # Failed loudly on the emit lane and cleared the poisoned result.
    compose_emits = [d for s, d in steps if s == "compose"]
    assert compose_emits and compose_emits[-1]["ok"] is False
    assert "#FALLBACK" in compose_emits[-1]["error"]
    assert "kg_result" not in run.vars["_temp"]["memact_composer"]
    # Re-entry does not re-query (applied_key latched).
    plan2 = wf.get_node("compose")(run, _Ctx())
    assert plan2.effect is None and plan2.next_node == "reason"


def test_truncation_always_marked_even_at_tiny_bounds() -> None:
    """0029 #15a: CodeAct's small-budget branch dropped the marker entirely
    (unmarked slice — ADR-0026 violation). Any lossy bound keeps a marker."""
    import inspect

    from abstractagent.adapters import codeact_runtime

    src = inspect.getsource(codeact_runtime)
    assert 'suffix = ""' not in src, "an unmarked-truncation branch is back"


def _repeat_guard_scenario_vars() -> Dict[str, Any]:
    executed = {"role": "assistant", "content": "wrote it"}
    tool_call = {"name": "write_file", "arguments": {"path": "a.txt", "content": "x"}, "call_id": "c1"}
    return {
        "context": {"task": "t", "messages": [{"role": "user", "content": "t"}, executed]},
        "scratchpad": {
            "iteration": 3,
            "max_iterations": 10,
            "cycles": [
                # Cycle 1: actually executed, observations succeeded.
                {"iteration": 1, "tool_calls": [dict(tool_call)], "observations": [{"name": "write_file", "success": True}]},
                # Cycle 2: guard skipped it — proposed batch, NO observations.
                {"iteration": 2, "tool_calls": [dict(tool_call)], "repeat_skipped": True},
                # Cycle 3 (current): the model repeats the identical batch again.
                {"iteration": 3},
            ],
        },
        "_runtime": {"inbox": [], "allowed_tools": ["write_file"]},
        "_temp": {"llm_response": {"content": "", "tool_calls": [dict(tool_call)]}},
        "_limits": {"max_iterations": 10, "current_iteration": 3},
    }


def test_third_identical_proposal_now_concludes_via_stuck_streak() -> None:
    """COMPOSITION UPDATE (0017 work half, 2026-07-21): this scenario — an
    executed batch, a guard-skipped identical proposal (nudge delivered), and
    a THIRD identical proposal — is now decisively terminated by the
    stuck-streak layer (proposals count; the model ignored the nudge). The
    old endless skip/nudge alternation was the 0017 defect in side-effect
    clothing."""
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic

    write_tool = ToolDefinition(name="write_file", description="w", parameters={})
    steps: list = []
    wf = create_react_workflow(logic=ReActLogic(tools=[write_tool]), on_step=lambda s, d: steps.append((s, d)))
    run = _run(_repeat_guard_scenario_vars(), node="parse")
    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "max_iterations"
    assert [s for s, _ in steps if s == "stuck_streak"], "streak verdict must be loud"


def test_repeat_guard_scans_past_skipped_cycles() -> None:
    """0029 #7 (property preserved under the 0017 layer): a skipped cycle
    records tool_calls but no observations — the NEXT identical batch used to
    compare against IT (no observations -> guard disengaged) and the
    protection alternated skip/execute/skip. The skip stamps `repeat_skipped`
    and the scan passes over marked cycles. Driven with the streak DISABLED
    so the original guard behavior stays independently pinned (configs with
    stuck_streak_threshold=0 or >3 still rely on it)."""
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic

    write_tool = ToolDefinition(name="write_file", description="w", parameters={})
    steps: list = []
    wf = create_react_workflow(logic=ReActLogic(tools=[write_tool]), on_step=lambda s, d: steps.append((s, d)))

    vars = _repeat_guard_scenario_vars()
    vars["_runtime"]["stuck_streak_threshold"] = 0
    run = _run(vars, node="parse")
    plan = wf.get_node("parse")(run, _Ctx())
    # The guard must still engage (scan past the skipped cycle to cycle 1).
    assert plan.next_node == "reason"
    assert [s for s, _ in steps if s == "parse_repeat_tool_calls"], "guard disengaged against a skipped cycle"
    # And the current cycle is now itself marked as skipped.
    assert run.vars["scratchpad"]["cycles"][-1].get("repeat_skipped") is True


def test_hook_steering_discarded_on_failed_and_cancelled_runs() -> None:
    """0029 #14: only done/max_iterations terminal NODES discarded steering —
    failure-terminated runs never execute a terminal node, so their queues
    rotted in LoopHooks._pending forever on long-lived hosts. The facade
    discards on FAILED/CANCELLED; a run-count cap backstops direct hosts."""
    from abstractagent.adapters.loop_hooks import LoopHooks
    from abstractagent.agents.base import BaseAgent

    hooks = LoopHooks()
    hooks._pending["run-failed"] = ["steer A", "steer B"]

    class _FakeAgent:
        pass

    agent = _FakeAgent()
    agent.hooks = hooks

    class _State:
        run_id = "run-failed"
        status = RunStatus.FAILED

    BaseAgent._discard_hook_steering_if_dead(agent, _State())
    assert "run-failed" not in hooks._pending

    # COMPLETED runs are the terminal nodes' business — facade leaves them.
    hooks._pending["run-done"] = ["steer C"]

    class _Done:
        run_id = "run-done"
        status = RunStatus.COMPLETED

    BaseAgent._discard_hook_steering_if_dead(agent, _Done())
    assert "run-done" in hooks._pending

    # Structural backstop: the pending-run cap evicts the OLDEST loudly.
    hooks2 = LoopHooks(max_pending_runs=2)
    hooks2.handlers.append(lambda ev: "steer!" if ev.step == "parse" else None)
    for rid in ("r1", "r2", "r3"):
        hooks2.push_run(rid)
        try:
            hooks2.dispatch("parse", {})
        finally:
            hooks2.pop_run()
    assert "r1" not in hooks2._pending  # oldest evicted
    assert set(hooks2._pending) == {"r2", "r3"}


def test_delegate_explicit_zero_budget_clamps_never_widens() -> None:
    """0029 #15b: explicit max_iterations=0 used to silently WIDEN to
    max(parent, 20); the resolver clamps explicit 0 to the floor of 1 —
    one contract at both sites now."""
    import inspect

    from abstractagent.adapters import codeact_runtime, memact_runtime, react_runtime

    for mod in (react_runtime, codeact_runtime, memact_runtime):
        src = inspect.getsource(mod)
        assert "child_iterations = child_val if child_val >= 1 else 1" in src, mod.__name__
        assert "child_iterations = child_val if isinstance(child_val, int) else 0" not in src, mod.__name__
