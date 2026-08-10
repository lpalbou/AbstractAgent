"""Sibling-loop defect batch (backlog 0010 A4, 0012 C3, 0013 C4).

Three maintainer-endorsed defect fixes from the 2026-07-12 loops meta-audit:

- 0010 (A4): CodeAct's fenced-block execution used to fire on replies that
  carried no action intent — the check ORDER ran code extraction BEFORE the
  `FINAL:` marker, so "FINAL: here's an example: ```python ..." executed the
  example. Fixed: intent (FINAL) precedes extraction; the fenced fallback is
  flag-gated (`_runtime.codeact_fenced_fallback`, default ON because the
  shipped prompt teaches the fence — the default flip belongs to proposal A1's
  native-primary wave, not a defect fix).
- 0012 (C3): the shared DELEGATE_AGENT_TOOL schema documents "child inherits
  the parent's budget (min 20); explicit max_iterations wins" — ReAct honors
  it; CodeAct/MemAct hardcoded 10 and ignored the argument (the schema lied
  for two of three loops). Fixed by mirroring ReAct's ruled resolution.
- 0013 (C4): MemActAgent accepted plan_mode/review_mode/review_max_rounds and
  read none of them (silent fall-open). Removed — passing them raises
  TypeError (loud by construction).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_EXEC_TOOL = ToolDefinition(name="execute_python", description="run", parameters={})


def _codeact_wf():
    # DELEGATE_AGENT_TOOL is declared the way the agent constructor declares it
    # (the allowlist gate intersects with the logic's declared tools).
    return create_codeact_workflow(
        logic=CodeActLogic(tools=[DELEGATE_AGENT_TOOL, _EXEC_TOOL]),
        on_step=None,
    )


def _parse_run(*, content: str, runtime_ns: Optional[Dict[str, Any]] = None) -> RunState:
    return RunState(
        run_id="r-parse",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1},
            "_runtime": dict({"inbox": [], "allowed_tools": ["execute_python"]}, **(runtime_ns or {})),
            "_temp": {"llm_response": {"content": content, "tool_calls": []}},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )


FENCED_EXAMPLE = "here's an example:\n```python\nprint('boom')\n```"


def test_codeact_final_reply_with_fenced_block_never_executes() -> None:
    """0010 validation (a): a FINAL-marked reply containing a fenced block is a
    final ANSWER — the illustrative code must NOT execute."""
    wf = _codeact_wf()
    run = _parse_run(content=f"FINAL: {FENCED_EXAMPLE}")

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "maybe_review"
    temp = run.vars["_temp"]
    assert "pending_code" not in temp
    assert temp["final_answer"].startswith("here's an example:")


def test_codeact_non_final_fenced_block_executes_as_today() -> None:
    """0010 validation (b): the prompted-model fallback is unchanged by default —
    a non-final reply with a fenced block routes to execute_code."""
    wf = _codeact_wf()
    run = _parse_run(content=FENCED_EXAMPLE)

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "execute_code"
    assert run.vars["_temp"]["pending_code"] == "print('boom')"


def test_codeact_fenced_fallback_flag_off_disables_extraction() -> None:
    """0010: with `_runtime.codeact_fenced_fallback=false` (native-primary
    setups), fenced text never executes — the reply is treated as an answer."""
    wf = _codeact_wf()
    run = _parse_run(content=FENCED_EXAMPLE, runtime_ns={"codeact_fenced_fallback": False})

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "maybe_review"
    temp = run.vars["_temp"]
    assert "pending_code" not in temp
    assert "print('boom')" in temp["final_answer"]


def test_codeact_fenced_execution_does_not_inject_hidden_timeout() -> None:
    """ADR-0027: fenced-code fallback must not smuggle the old 10s timeout."""
    wf = _codeact_wf()
    run = _parse_run(content=FENCED_EXAMPLE)

    parse_plan = wf.get_node("parse")(run, _Ctx())
    assert parse_plan.next_node == "execute_code"

    exec_plan = wf.get_node("execute_code")(run, _Ctx())
    assert exec_plan.effect is not None and exec_plan.effect.type.value == "tool_calls"
    tool_calls = exec_plan.effect.payload["tool_calls"]
    assert tool_calls == [
        {
            "name": "execute_python",
            "arguments": {"code": "print('boom')"},
            "call_id": "code",
        }
    ]


def _delegate_act_run(*, workflow_id: str, parent_max: int, arg_max: Optional[int]) -> RunState:
    args: Dict[str, Any] = {"task": "sub task"}
    if arg_max is not None:
        args["max_iterations"] = arg_max
    return RunState(
        run_id="r-delegate",
        workflow_id=workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1, "max_iterations": parent_max},
            "_runtime": {"inbox": [], "allowed_tools": ["execute_python", "delegate_agent"]},
            "_temp": {
                "pending_tool_calls": [
                    {"name": "delegate_agent", "arguments": args, "call_id": "d1"}
                ]
            },
            "_limits": {
                "max_iterations": parent_max,
                "max_history_messages": -1,
                "max_tokens": 32768,
            },
        },
    )


def _memact_wf():
    return create_memact_workflow(
        logic=MemActLogic(tools=[DELEGATE_AGENT_TOOL, _EXEC_TOOL]),
        on_step=None,
    )


# 0012 validation matrix from the backlog item: (parent, explicit_arg) -> child.
# Explicit arg WINS (even below 20 — an operator's explicit choice); otherwise
# inherit the parent with the ruled 20 floor. Arg-coercion rows (fable5 P1
# 2026-07-13): tool-call formats deliver budgets as strings/floats — "8" and
# "8.5" must both honor the explicit narrow intent (int("8.5") used to raise
# and silently WIDEN to the >=20 inheritance); unparseable junk and booleans
# fall back to inheritance, never crash, never a 1-iteration child from True.
_BUDGET_CASES = [
    (25, None, 25),
    (5, None, 20),
    (25, 40, 40),
    (25, 8, 8),
    (25, "8", 8),
    (25, "8.5", 8),
    (25, 8.9, 8),
    (25, "not-a-number", 25),
    (25, True, 25),
]


@pytest.mark.parametrize("parent_max,arg_max,expected", _BUDGET_CASES)
def test_codeact_delegate_child_budget_honors_schema(parent_max, arg_max, expected) -> None:
    wf = _codeact_wf()
    run = _delegate_act_run(workflow_id="codeact_agent", parent_max=parent_max, arg_max=arg_max)

    plan = wf.get_node("act")(run, _Ctx())
    assert plan.effect is not None and plan.effect.type.value == "start_subworkflow"
    child_limits = plan.effect.payload["vars"]["_limits"]
    assert child_limits["max_iterations"] == expected


@pytest.mark.parametrize("parent_max,arg_max,expected", _BUDGET_CASES)
def test_memact_delegate_child_budget_honors_schema(parent_max, arg_max, expected) -> None:
    wf = _memact_wf()
    run = _delegate_act_run(workflow_id="memact_agent", parent_max=parent_max, arg_max=arg_max)

    plan = wf.get_node("act")(run, _Ctx())
    assert plan.effect is not None and plan.effect.type.value == "start_subworkflow"
    child_limits = plan.effect.payload["vars"]["_limits"]
    assert child_limits["max_iterations"] == expected


def test_codeact_empty_retry_streak_resets_on_successful_parse() -> None:
    """fable5 P1 (2026-07-13): the empty-response retry counter counts
    CONSECUTIVE empties — a successful parse (FINAL/fenced/tool_calls) must
    reset it, or two recovered empties early in a run make every LATER single
    empty reply give up immediately despite remaining budget."""
    wf = _codeact_wf()

    # A successful FINAL parse with a stale streak of 2 resets the counter.
    run = _parse_run(content="FINAL: done.")
    run.vars["scratchpad"]["empty_response_retry_count"] = 2
    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "maybe_review"
    assert run.vars["scratchpad"]["empty_response_retry_count"] == 0

    # ... so a later empty reply RETRIES (routes to reason), never gives up.
    run.vars["_temp"]["llm_response"] = {"content": "", "tool_calls": []}
    plan2 = wf.get_node("parse")(run, _Ctx())
    assert plan2.next_node == "reason"
    assert run.vars["scratchpad"]["empty_response_retry_count"] == 1


def test_repeated_identical_tool_batch_executes_again_never_replays() -> None:
    """fable5 P0 (2026-07-13): the runtime keys effects on (run_id, node_id,
    payload) with call_ids STRIPPED and scans the whole ledger for prior
    results — so re-reading a file after editing it (a byte-identical
    TOOL_CALLS batch at the same node) silently REPLAYED the stale pre-edit
    result. The adapters now stamp a persisted per-issuance `act_seq` into the
    payload: each genuine issuance gets a fresh idempotency key while
    crash-replay dedup still holds (the counter persists in the same save as
    the effect record). Kernel proof: same tool, same args, two cycles — the
    handler EXECUTES twice and the second observation carries the new value."""
    from abstractruntime.core.models import Effect as REffect
    from abstractruntime.core.models import EffectType as REffectType
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic

    executions: list[str] = []
    llm_turn = {"n": 0}

    def llm_handler(run, effect, default_next_node):
        llm_turn["n"] += 1
        if llm_turn["n"] <= 2:
            # Two identical read requests across two cycles.
            return EffectOutcome.completed(
                {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}, "call_id": f"c{llm_turn['n']}"}]}
            )
        return EffectOutcome.completed({"content": "All done.", "tool_calls": []})

    def tool_handler(run, effect, default_next_node):
        payload = effect.payload or {}
        value = f"content-v{len(executions) + 1}"
        executions.append(value)
        calls = payload.get("tool_calls") or []
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {"call_id": str(c.get("call_id") or ""), "name": c.get("name"), "success": True, "output": value, "error": None}
                    for c in calls
                ],
            }
        )

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={REffectType.LLM_CALL: llm_handler, REffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})]),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["read_file"],
    )
    rid = rt.start(workflow=wf, vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}})
    from abstractruntime.core.models import RunStatus as RRunStatus

    for _ in range(60):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RRunStatus.COMPLETED, RRunStatus.FAILED, RRunStatus.CANCELLED):
            break

    final = rt.get_state(rid)
    assert final.status == RRunStatus.COMPLETED
    # THE pin: two genuine executions — the second identical batch was NOT
    # served from the ledger.
    assert executions == ["content-v1", "content-v2"]
    # And the durable cycle records observed the DISTINCT results.
    cycles = final.vars["scratchpad"]["cycles"]
    obs = [o for c in cycles if isinstance(c, dict) for o in (c.get("observations") or [])]
    outputs = [str(o.get("output") or "") for o in obs if isinstance(o, dict)]
    assert "content-v1" in " ".join(outputs) and "content-v2" in " ".join(outputs)


def test_memact_dead_knobs_are_deprecated_ignored_loudly() -> None:
    """0013 amended 2026-07-13 (regression adversary P0): the hard TypeError
    removal broke shipped abstractcode agent-switch call sites, so the knobs
    are DEPRECATED-IGNORED for one release — loud DeprecationWarning on
    meaningful values, silent on the legacy defaults (False/False/3), and the
    values are never CONSUMED (no plan/review nodes exist in MemAct)."""
    import warnings

    from abstractagent.agents.memact import MemActAgent

    # Meaningful values warn loudly.
    for kwargs in ({"plan_mode": True}, {"review_mode": True}, {"review_max_rounds": 5}):
        with pytest.warns(DeprecationWarning, match="deprecated-ignored"):
            try:
                MemActAgent(runtime=object(), **kwargs)  # type: ignore[arg-type]
            except AttributeError:
                pass  # runtime=object() dies later in __init__; the warning fired first

    # Legacy default spellings (the abstractcode call shape) stay SILENT.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            MemActAgent(runtime=object(), plan_mode=False, review_mode=False, review_max_rounds=3)  # type: ignore[arg-type]
        except AttributeError:
            pass
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
