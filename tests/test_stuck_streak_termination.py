"""Stuck-streak termination (backlog 0017 work half, work:abstractagent-0017).

The defect: read-only repeat loops spin uncounted until max_iterations — the
side-effect repeat guard deliberately skips only re-EXECUTION of succeeded
side-effect batches. This wave: N consecutive identical tool batches (default
3) or a strict A-B-A-B oscillation route into the EXISTING conclusion path
with a NAMED reason (loud synthesis; `stuck_streak` hook event; additive
`conclusion_forced` output key; canonical outcome enum unchanged).

Conservatism pinned: interleaved distinct work never counts; sub-threshold
repeats never count; the knob disables cleanly; visit turns fence at the turn
boundary (reusing the c2447 F5 fence) and the visit-lane conclude directive
stays free of loop vocabulary.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import _repeat_streak_verdict, create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


def _tc(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {"name": name, "arguments": args, "call_id": call_id}


def _drive(
    llm_script: List[Dict[str, Any]],
    *,
    runtime_vars: Optional[Dict[str, Any]] = None,
    max_iterations: int = 12,
    payload_sink: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], RunState]:
    events: List[Tuple[str, Dict[str, Any]]] = []
    call_n = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        if payload_sink is not None:
            payload = effect.payload if isinstance(effect.payload, dict) else {}
            payload_sink.append(json.loads(json.dumps(payload)))
        idx = min(call_n["n"], len(llm_script) - 1)
        call_n["n"] += 1
        return EffectOutcome.completed(dict(llm_script[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={}),
                                ToolDefinition(name="list_files", description="List", parameters={})]),
        on_step=lambda step, data: events.append((step, json.loads(json.dumps(data, default=str)))),
        workflow_id="wf-stuck",
        provider="stub",
        model="stub",
        allowed_tools=["read_file", "list_files"],
    )
    vars0: Dict[str, Any] = {
        "context": {"task": "Investigate", "messages": []},
        "_runtime": dict({"inbox": []}, **(runtime_vars or {})),
        "_limits": {"max_iterations": max_iterations},
    }
    rid = rt.start(workflow=wf, vars=vars0)
    for _ in range(200):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = rt.get_state(rid)
    assert state.status == RunStatus.COMPLETED
    return events, state


_READ_A = {"content": "Reading.", "tool_calls": [_tc("read_file", {"path": "a.txt"}, "c1")]}
_READ_B = {"content": "Listing.", "tool_calls": [_tc("list_files", {"directory_path": "."}, "c2")]}
_FINAL = {"content": "Answer: nothing more to learn.", "tool_calls": []}


def test_three_identical_readonly_batches_force_named_conclusion() -> None:
    """The motivating case: a read-only repeat loop terminates at the third
    identical proposal with the named reason on every machine surface."""
    payloads: List[Dict[str, Any]] = []
    events, state = _drive([_READ_A, _READ_A, _READ_A, _FINAL], payload_sink=payloads)

    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert len(streaks) == 1
    assert streaks[0]["kind"] == "repeat"
    assert streaks[0]["span"] == 3

    out = state.output or {}
    assert out.get("outcome") == "iteration_budget"
    forced = out.get("conclusion_forced")
    assert isinstance(forced, dict) and forced.get("kind") == "repeat"
    # Named in the human report too — never silent.
    assert "conclusion forced: repeat streak" in str(out.get("report") or "")
    # The third batch never executed: exactly 2 act events.
    assert sum(1 for (s, _) in events if s == "act") == 2
    # The conclusion prompt named the loop to the model (task lane): the
    # last LLM payload is the conclusion call.
    assert "repeated the exact same tool calls without making progress" in json.dumps(payloads[-1])


def test_two_identical_batches_do_not_terminate() -> None:
    """Sub-threshold: two identical proposals then a final answer — the loop
    completes normally with no verdict."""
    events, state = _drive([_READ_A, _READ_A, _FINAL])
    assert not [d for (s, d) in events if s == "stuck_streak"]
    assert (state.output or {}).get("outcome") == "final_answer"
    assert "conclusion_forced" not in (state.output or {})


def test_interleaved_distinct_work_never_counts() -> None:
    """A-X-A-X-A: re-reading a file between DIFFERENT calls is legitimate
    progress — no verdict fires (consecutiveness is load-bearing)."""
    events, state = _drive([_READ_A, _READ_B, _READ_A, _READ_B, _READ_A, _FINAL], max_iterations=12)
    # This IS the A-B-A-B oscillation shape by cycle 4 — deliberately chosen:
    # strict alternation between two batches completing twice IS a verdict.
    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert len(streaks) == 1
    assert streaks[0]["kind"] == "oscillation"


def test_true_interleaving_with_three_distinct_batches_is_clean() -> None:
    """A-B-C-A-B: three distinct batches cycling is neither a repeat nor an
    A-B oscillation — no verdict."""
    read_c = {"content": "Reading c.", "tool_calls": [_tc("read_file", {"path": "c.txt"}, "c3")]}
    events, state = _drive([_READ_A, _READ_B, read_c, _READ_A, _READ_B, _FINAL], max_iterations=12)
    assert not [d for (s, d) in events if s == "stuck_streak"]
    assert (state.output or {}).get("outcome") == "final_answer"


def test_knob_disables_the_verdict() -> None:
    events, state = _drive(
        [_READ_A, _READ_A, _READ_A, _READ_A, _FINAL],
        runtime_vars={"stuck_streak_threshold": 0},
    )
    assert not [d for (s, d) in events if s == "stuck_streak"]
    assert (state.output or {}).get("outcome") == "final_answer"


def test_visit_lane_conclude_stays_chrome_free_and_verdict_is_machine_only() -> None:
    """Under suppress_loop_tail the forcing works identically but the
    conclude directive carries NO loop vocabulary (c2447 chrome class) —
    the named reason lives on the machine surfaces."""
    llm_payloads: List[Dict[str, Any]] = []
    call_n = {"n": 0}
    script = [_READ_A, _READ_A, _READ_A, {"content": "Here is what I have.", "tool_calls": []}]

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(json.loads(json.dumps(payload)))
        idx = min(call_n["n"], len(script) - 1)
        call_n["n"] += 1
        return EffectOutcome.completed(dict(script[idx]))

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_handler,
            EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed(
                {
                    "mode": "executed",
                    "results": [
                        {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
                        for tc in (e.payload.get("tool_calls") or [])
                    ],
                }
            ),
        },
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]),
        workflow_id="wf-stuck-visit",
        provider="stub",
        model="stub",
        allowed_tools=["read_file"],
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Visit turn", "messages": []},
            "_runtime": {"inbox": [], "suppress_loop_tail": True},
            "_limits": {"max_iterations": 12},
        },
    )
    for _ in range(200):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = rt.get_state(rid)
    assert state.status == RunStatus.COMPLETED
    out = state.output or {}
    assert isinstance(out.get("conclusion_forced"), dict)
    # The conclusion prompt (last LLM payload) carries the host-voiced
    # wrap-up, not loop vocabulary.
    concl = json.dumps(llm_payloads[-1])
    for banned in ("without making progress", "tool batches", "maximum allowed ReAct iterations", "[loop]"):
        assert banned not in concl
    assert "bring your reply to a close" in concl


# ---------------------------------------------------------------------------
# Unit half: the verdict fold
# ---------------------------------------------------------------------------

def _cycles(*batches: Optional[List[Tuple[str, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, b in enumerate(batches):
        c: Dict[str, Any] = {"i": i + 1}
        if b is not None:
            c["tool_calls"] = [{"name": n, "arguments": a} for (n, a) in b]
        out.append(c)
    return out


def test_verdict_fence_blocks_prior_turn_cycles() -> None:
    a = [("read_file", {"path": "a"})]
    cycles = _cycles(a, a, a)
    assert _repeat_streak_verdict(cycles, turn_fence=0, threshold=3) is not None
    # Fence planted after the first cycle: only two of the three are this
    # turn's — no verdict (repeating yesterday's search is a relationship).
    assert _repeat_streak_verdict(cycles, turn_fence=1, threshold=3) is None


def test_verdict_ignores_toolless_cycles_and_handles_junk() -> None:
    a = [("read_file", {"path": "a"})]
    cycles = _cycles(a, None, a, None, a)  # toolless cycles are transparent
    assert _repeat_streak_verdict(cycles, turn_fence=0, threshold=3) is not None
    assert _repeat_streak_verdict(None, turn_fence=0, threshold=3) is None
    assert _repeat_streak_verdict([], turn_fence=0, threshold=3) is None
    assert _repeat_streak_verdict(cycles, turn_fence=0, threshold=1) is None  # threshold < 2 disabled
