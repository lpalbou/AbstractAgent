"""Stuck-streak NUDGE-then-stop (backlog 0017 + operator directive 2026-08-21).

The defect 0017 fixed: read-only repeat loops spin uncounted until
max_iterations — the side-effect repeat guard deliberately skips only
re-EXECUTION of succeeded side-effect batches. N consecutive identical tool
batches (default 3) or a strict A-B-A-B oscillation are detected.

What 2026-08-21 changed: the detection no longer ENDS the turn on sight. The
first detection pushes a NUDGE into the loop's inbox — the repeated batch, the
count, the observations it keeps getting back, and the consequence — and lets
the model act on it. The turn is ended only if the pattern continues
(`stuck_streak_hard_threshold`, default = nudge span + 2). Rationale: the old
behaviour named the loop to the model only in the conclusion prompt, i.e.
after its last chance to change course, and it burned a whole turn on a
recoverable mistake (live: 12 of 50 iterations, session acode-bc425138014f).

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


def test_three_identical_readonly_batches_nudge_instead_of_terminating() -> None:
    """The motivating case: at the third identical proposal the loop NUDGES —
    it does not end the turn — and the model gets to recover."""
    payloads: List[Dict[str, Any]] = []
    events, state = _drive([_READ_A, _READ_A, _READ_A, _FINAL], payload_sink=payloads)

    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert len(streaks) == 1
    assert streaks[0]["kind"] == "repeat"
    assert streaks[0]["span"] == 3
    assert streaks[0]["action"] == "nudged"

    out = state.output or {}
    # The turn was NOT ended: the model answered on the next cycle.
    assert out.get("outcome") == "final_answer"
    assert "conclusion_forced" not in out
    # The third batch still never executed: exactly 2 act events.
    assert sum(1 for (s, _) in events if s == "act") == 2

    # The nudge reached the MODEL, on the call right after the detection —
    # with the batch, the count, and the consequence.
    nudge_payload = json.dumps(payloads[3])
    assert "[loop guard] You are stuck" in nudge_payload
    assert "EXACT SAME tool batch 3 times in a row" in nudge_payload
    assert "read_file" in nudge_payload
    assert "this turn will be ended" in nudge_payload
    # And the nudge is recorded for the escalation to key on.
    assert isinstance((out.get("scratchpad") or {}).get("stuck_nudged"), dict)


def test_repeating_after_the_nudge_ends_the_turn_at_span_five() -> None:
    """X+2: told at 3, ignored twice, stopped at 5 — with the named reason on
    every machine surface exactly as before."""
    payloads: List[Dict[str, Any]] = []
    events, state = _drive([_READ_A] * 6 + [_FINAL], payload_sink=payloads)

    streaks = [d for (s, d) in events if s == "stuck_streak"]
    actions = [d.get("action") for d in streaks]
    assert actions == ["nudged", "repeat_after_nudge", "stopped"]
    assert streaks[0]["span"] == 3 and streaks[-1]["span"] == 5
    assert streaks[-1]["nudged"] is True

    out = state.output or {}
    assert out.get("outcome") == "iteration_budget"
    forced = out.get("conclusion_forced")
    assert isinstance(forced, dict) and forced.get("kind") == "repeat"
    assert forced.get("nudged") is True
    assert "conclusion forced: repeat streak" in str(out.get("report") or "")
    # Only the first two batches ever executed.
    assert sum(1 for (s, _) in events if s == "act") == 2
    assert "repeated the exact same tool calls without making progress" in json.dumps(payloads[-1])


def test_hard_threshold_zero_never_terminates() -> None:
    """Nudge-only posture: the loop is named, never guillotined; the model
    keeps its turn and max_iterations stays the only backstop."""
    events, state = _drive(
        [_READ_A] * 8 + [_FINAL],
        runtime_vars={"stuck_streak_hard_threshold": 0},
        max_iterations=12,
    )
    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert streaks and all(d.get("action") != "stopped" for d in streaks)
    assert [d.get("action") for d in streaks].count("nudged") == 1
    out = state.output or {}
    assert out.get("outcome") == "final_answer"
    assert "conclusion_forced" not in out


def test_hard_threshold_equal_to_threshold_restores_stop_on_first_detection() -> None:
    """The pre-2026-08-21 behaviour is still reachable by configuration."""
    events, state = _drive(
        [_READ_A, _READ_A, _READ_A, _FINAL],
        runtime_vars={"stuck_streak_hard_threshold": 3},
    )
    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert len(streaks) == 1 and streaks[0]["action"] == "stopped"
    assert streaks[0]["nudged"] is False
    out = state.output or {}
    assert out.get("outcome") == "iteration_budget"
    assert isinstance(out.get("conclusion_forced"), dict)


def test_a_second_distinct_stuck_pattern_earns_its_own_nudge() -> None:
    """Once per PATTERN, not once per turn: a model that changes strategy and
    then gets stuck a different way must be told about the new loop too."""
    events, state = _drive([_READ_A, _READ_A, _READ_A, _READ_B, _READ_B, _READ_B, _FINAL])
    nudges = [d for (s, d) in events if s == "stuck_streak" and d.get("action") == "nudged"]
    assert len(nudges) == 2
    assert all(d["kind"] == "repeat" and d["span"] == 3 for d in nudges)
    assert nudges[0]["key"] != nudges[1]["key"]
    assert (state.output or {}).get("outcome") == "final_answer"


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
    assert streaks and streaks[0]["kind"] == "oscillation"
    # Nudged at span 4, not terminated; the continued alternation keys on the
    # SAME pattern (the unordered pair), so it is not nudged twice.
    assert streaks[0]["action"] == "nudged"
    assert [d.get("action") for d in streaks].count("nudged") == 1
    assert len({d.get("key") for d in streaks}) == 1
    assert (state.output or {}).get("outcome") == "final_answer"


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
    # Five identical proposals: nudged at 3, ended at 5 (default X+2).
    script = [_READ_A] * 6 + [{"content": "Here is what I have.", "tool_calls": []}]

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

    # The NUDGE rides the same visitor-facing conversation, so it obeys the
    # same rule: the correction is made, the loop vocabulary is not.
    nudge = json.dumps(llm_payloads[3])
    for banned in ("[loop guard]", "tool batch", "cycles", "EXACT SAME", "stuck"):
        assert banned not in nudge
    assert "already been made and returned the same result" in nudge


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


def test_turn_reset_clears_the_nudge_ledger_so_a_new_turn_earns_its_own() -> None:
    """Per-TURN state, like `stuck_streak` itself: a visit turn that was
    nudged must not spend the NEXT turn's nudge, or a model repeating
    yesterday's batch would be guillotined with no warning at all."""
    from abstractagent.adapters.react_runtime import reset_react_turn

    vars0 = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {
            "iteration": 4,
            "cycles": [{"i": 1, "tool_calls": [{"name": "read_file", "arguments": {"p": "a"}}]}],
            "stuck_streak": {"kind": "repeat", "span": 5, "cycle": 5},
            "stuck_nudged": {"abc123": 3},
        },
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {"max_iterations": 10},
    }
    reset_react_turn(vars0)
    sp = vars0["scratchpad"]
    assert "stuck_streak" not in sp
    assert "stuck_nudged" not in sp, "a nudged turn must not consume the next turn's nudge"


# ---------------------------------------------------------------------------
# Escalation cannot be evaded (adversarial review, 2026-08-21)
# ---------------------------------------------------------------------------

def test_interleaving_one_different_batch_cannot_postpone_the_stop_forever() -> None:
    """The trailing span is resettable: A-A-A-B-A-A-A-B… keeps it pinned at 3.
    The first cut of this policy stopped at `nudge_span + 2` and so nudged
    forever and died on max_iterations with NO `conclusion_forced` — the
    operator got the misleading budget card back. Detections are counted
    instead, and a detection count cannot be reset by interleaving."""
    events, state = _drive([_READ_A, _READ_A, _READ_A, _READ_B] * 6 + [_FINAL], max_iterations=30)
    actions = [d.get("action") for (s, d) in events if s == "stuck_streak"]
    assert "stopped" in actions, f"the interleaved loop must still be stopped: {actions}"
    out = state.output or {}
    assert out.get("outcome") == "iteration_budget"
    forced = out.get("conclusion_forced")
    assert isinstance(forced, dict) and forced.get("nudged") is True
    # And it stopped on the pattern's THIRD sighting, not on the budget.
    assert int(out.get("iterations") or 0) < 30


def test_argument_drift_cannot_mint_unlimited_fresh_nudges() -> None:
    """A retry carrying a new id is a NEW pattern key, so per-pattern counting
    alone never escalates (adversary: 8 nudges, 0 stops, 25 iterations). The
    per-turn cap on DISTINCT nudges closes it."""
    script: List[Dict[str, Any]] = []
    for n in range(8):
        drifted = {"content": "Retrying.", "tool_calls": [_tc("read_file", {"path": f"a{n}.txt"}, f"c{n}")]}
        script.extend([drifted, drifted, drifted])
    events, state = _drive(script + [_FINAL], max_iterations=40)
    streaks = [d for (s, d) in events if s == "stuck_streak"]
    nudges = [d for d in streaks if d.get("action") == "nudged"]
    assert len(nudges) <= 3, f"the per-turn nudge cap must hold: {[d.get('action') for d in streaks]}"
    assert streaks[-1].get("action") == "stopped"
    assert (state.output or {}).get("outcome") == "iteration_budget"


def test_nudge_cap_knob_disables_the_distinct_pattern_stop() -> None:
    script: List[Dict[str, Any]] = []
    for n in range(5):
        drifted = {"content": "Retrying.", "tool_calls": [_tc("read_file", {"path": f"b{n}.txt"}, f"d{n}")]}
        script.extend([drifted, drifted, drifted])
    events, state = _drive(
        script + [_FINAL],
        runtime_vars={"stuck_nudge_max_per_turn": 0},
        max_iterations=40,
    )
    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert [d.get("action") for d in streaks].count("nudged") == 5
    assert not [d for d in streaks if d.get("action") == "stopped"]
    assert (state.output or {}).get("outcome") == "final_answer"


def test_a_hard_threshold_beyond_the_scan_depth_is_still_reachable() -> None:
    """The verdict scan stops collecting at `max_span` (24 by default), so an
    operator-set span threshold above it could never fire — and the nudge
    promised a number that would never arrive."""
    events, state = _drive(
        [_READ_A] * 34 + [_FINAL],
        runtime_vars={"stuck_streak_hard_threshold": 30},
        max_iterations=40,
    )
    streaks = [d for (s, d) in events if s == "stuck_streak"]
    stopped = [d for d in streaks if d.get("action") == "stopped"]
    assert stopped, "a reachable stop must exist for hard_threshold=30"
    assert stopped[0]["span"] >= 30 or stopped[0]["hits"] >= 30
    assert (state.output or {}).get("outcome") == "iteration_budget"


def test_malformed_observations_do_not_kill_the_run() -> None:
    """A guard must not be the thing that crashes the loop: `observations`
    arriving as a non-list used to raise straight through the parse node."""
    from abstractagent.adapters.react_runtime import _repeat_streak_verdict, _stuck_nudge_message

    batch = [{"name": "read_file", "arguments": {"path": "a"}}]
    cycles: List[Dict[str, Any]] = [
        {"i": 1, "tool_calls": batch, "observations": 5},
        {"i": 2, "tool_calls": batch, "observations": None},
        {"i": 3, "tool_calls": batch, "observations": [{"name": "read_file", "success": False, "error": "boom"}]},
    ]
    verdict = _repeat_streak_verdict(cycles, turn_fence=0, threshold=3)
    assert verdict is not None
    msg = _stuck_nudge_message(cycles, verdict=verdict, turn_fence=0, remaining=2)
    assert "read_file" in msg and "boom" in msg


# ---------------------------------------------------------------------------
# Identical CALL is not enough — the ANSWER must be identical too
# (nudge forensics, 2026-08-21)
# ---------------------------------------------------------------------------

def _obs_cycles(*outputs: Optional[str]) -> List[Dict[str, Any]]:
    """Cycles proposing the same batch, each with the given observation text.
    `None` = the cycle was refused and produced no observation."""
    batch = [{"name": "poll_job", "arguments": {"job_id": "j1"}}]
    out: List[Dict[str, Any]] = []
    for i, text in enumerate(outputs):
        c: Dict[str, Any] = {"i": i + 1, "tool_calls": [dict(b) for b in batch]}
        if text is not None:
            c["observations"] = [{"name": "poll_job", "success": True, "output": text}]
        out.append(c)
    return out


def test_a_repeat_whose_answers_change_is_not_stuck() -> None:
    """A poll that reports 10% / 45% / 80% is the same CALL three times and is
    making progress. Fingerprinting the call alone flagged it — live
    forensics caught exactly that on a progressing job."""
    from abstractagent.adapters.react_runtime import _repeat_streak_verdict

    progressing = _obs_cycles("running, 10%", "running, 45%", "running, 80%")
    assert _repeat_streak_verdict(progressing, turn_fence=0, threshold=3) is None


def test_a_repeat_whose_answers_are_identical_is_still_stuck() -> None:
    from abstractagent.adapters.react_runtime import _repeat_streak_verdict

    stuck = _obs_cycles("STATUS: pending", "STATUS: pending", "STATUS: pending")
    verdict = _repeat_streak_verdict(stuck, turn_fence=0, threshold=3)
    assert verdict is not None
    assert verdict["kind"] == "repeat" and verdict["span"] == 3
    assert verdict["answered"] is True


def test_cycles_that_produced_no_observation_stay_transparent() -> None:
    """A refused proposal (the guard skipped it) neither proves nor breaks a
    streak — the 0029 #7 property, restated for the answer check."""
    from abstractagent.adapters.react_runtime import _repeat_streak_verdict

    mixed = _obs_cycles("same", None, "same")
    v = _repeat_streak_verdict(mixed, turn_fence=0, threshold=3)
    assert v is not None and v["span"] == 3

    # No observations anywhere: still a stuck PROPOSAL streak, but the nudge
    # has no answer to echo.
    none_at_all = _obs_cycles(None, None, None)
    v2 = _repeat_streak_verdict(none_at_all, turn_fence=0, threshold=3)
    assert v2 is not None and v2["answered"] is False


def test_the_nudge_states_no_unprovable_claim_and_offers_the_waiting_exit() -> None:
    """The first wording asserted "repeating it will not produce a different
    result" — which this guard cannot know, and which is false for a
    poll-until-ready tool. Live, gpt-5.4-mini weighed that sentence against
    its tool's own "poll again with the SAME job_id" and believed the tool:
    the turn was guillotined in 8 of 13 runs. With the claim removed and the
    waiting exit added, it complied in 5 of 5."""
    from abstractagent.adapters.react_runtime import _repeat_streak_verdict, _stuck_nudge_message

    cycles = _obs_cycles("STATUS: pending", "STATUS: pending", "STATUS: pending")
    verdict = _repeat_streak_verdict(cycles, turn_fence=0, threshold=3)
    msg = _stuck_nudge_message(cycles, verdict=verdict, turn_fence=0, remaining=2)
    assert "will not produce a different result" not in msg
    assert "received that same answer 3 times" in msg
    assert "WAITING on something external" in msg
    assert "does not make the wait shorter" in msg
