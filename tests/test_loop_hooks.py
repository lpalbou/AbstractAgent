"""First-class loop hooks: listen + steer + capture (maintainer directive 2026-07-12).

Pins the loop half of the hook contract through the REAL runtime kernel:

- LISTEN: handlers receive structured HookEvents for the canonical moments
  (cycle_start / tool_proposed / tool_executed / turn_end with outcome), and
  unmapped steps pass through under their raw names — total coverage;
- STEER: a handler's returned injection folds into the DURABLE guidance
  inbox and reaches the next reason payload as an operator-guidance message
  (resume-safe: hooks never mutate loop state mid-node), and the consumption
  fires the `message_drained` listen point (the fleet's in-loop
  "message received" hook);
- CAPTURE/containment: a raising handler never kills the run — the failure
  surfaces as a `hook_error` event on the same stream; unsupported action
  shapes surface likewise (works-or-loud);
- COMPAT: `on_step` continues to receive the flat stream unchanged when both
  surfaces are wired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.loop_hooks import HookEvent, LoopHooks
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


_TOOL_REPLY = {
    "content": "Looking.",
    "tool_calls": [{"name": "probe", "arguments": {"q": "x"}, "call_id": "call_1"}],
}
_FINAL_REPLY = {"content": "Done.", "tool_calls": []}


def _run_loop(
    llm_responses: List[Dict[str, Any]],
    *,
    hooks: Optional[LoopHooks] = None,
    on_step: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], RunState]:
    llm_payloads: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        import json as _json

        llm_payloads.append(_json.loads(_json.dumps(payload)))
        idx = min(calls["n"], len(llm_responses) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(llm_responses[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="probe", description="Probe", parameters={})]),
        workflow_id="react_hooks",
        provider="stub",
        model="stub",
        on_step=on_step,
        hooks=hooks,
    )
    vars: Dict[str, Any] = {"context": {"task": "Hook test task", "messages": []}, "_runtime": {"inbox": []}}
    run_id = runtime.start(workflow=workflow, vars=vars, actor_id=None, session_id="sess-hooks")
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return llm_payloads, state


def test_listen_canonical_events_fire_in_order_with_structured_payloads() -> None:
    events: List[HookEvent] = []
    hooks = LoopHooks().add(events.append)

    _run_loop([_TOOL_REPLY, _FINAL_REPLY], hooks=hooks)

    names = [e.name for e in events]
    # The canonical spine, in loop order.
    assert names.index("cycle_start") < names.index("tool_proposed") < names.index("tool_executed")
    assert names[-1] == "turn_end"
    # turn_end carries WHY (final answer vs budget).
    turn_end = events[-1]
    assert turn_end.data.get("outcome") == "final_answer"
    assert turn_end.step == "done"
    # Structured capture: tool events carry the tool name from the emit data.
    tool_exec = next(e for e in events if e.name == "tool_executed")
    assert tool_exec.data.get("tool") == "probe"
    # Raw steps remain visible on every event (compat vocabulary).
    assert all(isinstance(e.step, str) and e.step for e in events)


def test_unmapped_steps_pass_through_under_raw_names() -> None:
    events: List[HookEvent] = []
    hooks = LoopHooks().add(events.append)
    # `parse_final` fires on every final answer and has no canonical mapping —
    # it must pass through under its raw name (total coverage, no filtering).
    _run_loop([_FINAL_REPLY], hooks=hooks)
    raw_named = {e.name for e in events if e.name == e.step}
    assert "parse_final" in raw_named


def test_steer_from_hook_folds_into_durable_inbox_and_reaches_next_reason() -> None:
    events: List[HookEvent] = []
    steered = {"done": False}

    def steer_once(event: HookEvent) -> Any:
        events.append(event)
        if event.name == "tool_executed" and not steered["done"]:
            steered["done"] = True
            return {"inject": "Change of plan: answer in French."}
        return None

    hooks = LoopHooks().add(steer_once)
    flat: List[str] = []
    payloads, state = _run_loop(
        [_TOOL_REPLY, _FINAL_REPLY], hooks=hooks, on_step=lambda s, d: flat.append(s)
    )

    # The injection reached the SECOND reason payload as a durable user
    # interjection (the operator-guidance shape), not an ephemeral tail.
    second = payloads[1]
    msgs = second.get("messages") or []
    guidance_msgs = [
        m for m in msgs if isinstance(m, dict) and "Change of plan: answer in French." in str(m.get("content") or "")
    ]
    assert guidance_msgs, "hook injection must reach the next reason payload"
    # And it persists in the DURABLE transcript (resume-safe steering).
    durable = ((state.vars or {}).get("context") or {}).get("messages") or []
    assert any("Change of plan" in str(m.get("content") or "") for m in durable if isinstance(m, dict))
    # The consumption fired the in-loop message hook (fleet seam).
    assert any(e.name == "message_drained" for e in events)
    # The steer act itself surfaced on the FLAT stream (observable steering;
    # follow-ups deliberately do not re-enter hook dispatch).
    assert "hook_steer" in flat


def test_raising_handler_is_contained_and_surfaces_as_hook_error() -> None:
    flat_steps: List[str] = []

    def bad_handler(event: HookEvent) -> None:
        if event.name == "cycle_start":
            raise RuntimeError("listener bug")

    hooks = LoopHooks().add(bad_handler)
    # on_step receives the flat stream INCLUDING hook_error follow-ups.
    _, state = _run_loop(
        [_FINAL_REPLY],
        hooks=hooks,
        on_step=lambda step, data: flat_steps.append(step),
    )
    assert state.status == RunStatus.COMPLETED  # the run survived the listener
    assert "hook_error" in flat_steps


def test_unsupported_action_shape_is_loud_never_silent() -> None:
    flat: List[Tuple[str, Dict[str, Any]]] = []

    def weird_handler(event: HookEvent) -> Any:
        if event.name == "cycle_start":
            return {"unknown_action": True}
        return None

    hooks = LoopHooks().add(weird_handler)
    _run_loop([_FINAL_REPLY], hooks=hooks, on_step=lambda s, d: flat.append((s, d)))
    errors = [d for s, d in flat if s == "hook_error"]
    assert errors and "unsupported hook action" in str(errors[0].get("error"))


def test_on_step_compat_unchanged_when_both_surfaces_wired() -> None:
    flat: List[str] = []
    events: List[HookEvent] = []
    hooks = LoopHooks().add(events.append)
    _run_loop([_TOOL_REPLY, _FINAL_REPLY], hooks=hooks, on_step=lambda s, d: flat.append(s))
    # The flat stream still carries the raw vocabulary...
    assert "reason" in flat and "observe" in flat and "done" in flat
    # ...and the hook stream saw the same moments under canonical names.
    assert {"cycle_start", "tool_executed", "turn_end"} <= {e.name for e in events}


def test_steering_is_run_scoped_never_leaks_into_a_later_run() -> None:
    """Adversary P1-1: the pending queue is keyed BY RUN. Steering queued at
    turn_end (no later reason boundary in that run) must NOT contaminate the
    next run served by the same workflow product — it is DISCARDED LOUDLY at
    the terminal (`hook_steer_discarded`), and run 2's transcript stays clean."""
    flat: List[Tuple[str, Dict[str, Any]]] = []
    fired = {"n": 0}

    def steer_at_turn_end(event: HookEvent) -> Any:
        if event.name == "turn_end" and fired["n"] == 0:
            fired["n"] += 1
            return {"inject": "TOO LATE FOR THIS RUN"}
        return None

    hooks = LoopHooks().add(steer_at_turn_end)

    llm_payloads: List[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        import json as _json

        llm_payloads.append(_json.loads(_json.dumps(payload)))
        return EffectOutcome.completed(dict(_FINAL_REPLY))

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="probe", description="Probe", parameters={})]),
        workflow_id="react_hooks_two_runs",
        provider="stub",
        model="stub",
        on_step=lambda s, d: flat.append((s, d)),
        hooks=hooks,
    )
    for session in ("run-one", "run-two"):
        run_id = runtime.start(
            workflow=workflow,
            vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}},
            actor_id=None,
            session_id=session,
        )
        for _ in range(40):
            state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
            if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                break
        assert runtime.get_state(run_id).status == RunStatus.COMPLETED
        if session == "run-two":
            durable = ((runtime.get_state(run_id).vars or {}).get("context") or {}).get("messages") or []
            assert not any(
                "TOO LATE" in str(m.get("content") or "") for m in durable if isinstance(m, dict)
            ), "run 1's late steering leaked into run 2"

    # The undelivered steering was discarded LOUDLY at run 1's terminal.
    assert any(s == "hook_steer_discarded" and d.get("count") == 1 for s, d in flat)


def test_hooks_only_config_observes_hook_error_without_on_step() -> None:
    """Adversary P1-2: with hooks wired and NO on_step, handler failures must
    still be observable — follow-ups are notified to the handlers themselves
    (pure listen), so the programmatic surface sees its own errors."""
    seen: List[HookEvent] = []

    def sometimes_bad(event: HookEvent) -> None:
        seen.append(event)
        if event.name == "cycle_start" and not any(e.step == "hook_error" for e in seen):
            raise RuntimeError("listener bug")

    hooks = LoopHooks().add(sometimes_bad)
    _, state = _run_loop([_FINAL_REPLY], hooks=hooks, on_step=None)
    assert state.status == RunStatus.COMPLETED
    assert any(e.step == "hook_error" and "listener bug" in str(e.data.get("error")) for e in seen)


def test_handler_mutation_of_event_data_cannot_corrupt_the_loop() -> None:
    """Adversary P2-1: handlers receive a COPY of the emit payload — mutating
    event.data must not rewrite the tool arguments the loop executes."""
    tool_payloads: List[Dict[str, Any]] = []

    def mutating_handler(event: HookEvent) -> None:
        if event.name == "tool_proposed":
            calls = event.data.get("tool_calls")
            if isinstance(calls, list):
                for c in calls:
                    if isinstance(c, dict) and isinstance(c.get("arguments"), dict):
                        c["arguments"]["q"] = "CORRUPTED"

    hooks = LoopHooks().add(mutating_handler)

    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node, effect
        idx = min(calls["n"], 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict([_TOOL_REPLY, _FINAL_REPLY][idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        import json as _json

        tool_payloads.append(_json.loads(_json.dumps(payload)))
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="probe", description="Probe", parameters={})]),
        workflow_id="react_hooks_mutation",
        provider="stub",
        model="stub",
        hooks=hooks,
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-mutation",
    )
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    assert tool_payloads, "the tool call must have executed"
    executed_args = (tool_payloads[0].get("tool_calls") or [{}])[0].get("arguments") or {}
    assert executed_args.get("q") == "x", "handler mutation must never reach execution"


def test_events_carry_run_identity() -> None:
    """Adversary P2-3: every in-loop event carries the run id (and the agent
    workflow id), so handlers can scope observation and steering."""
    events: List[HookEvent] = []
    hooks = LoopHooks().add(events.append)
    _, state = _run_loop([_FINAL_REPLY], hooks=hooks)
    assert events, "events must have fired"
    assert all(e.run_id == state.run_id for e in events)
    assert all(e.agent == "react_hooks" for e in events)


def test_slow_handler_strikes_out_and_is_disabled_loudly() -> None:
    """Plan H-row (sync-on-tick DoS): a handler over the time budget earns
    loud `hook_slow` strikes and is DISABLED at the strike cap — the loop
    survives, the containment is observable, fast handlers keep firing."""
    import time as _time

    flat: List[Tuple[str, Dict[str, Any]]] = []
    slow_calls = {"n": 0}
    fast_calls = {"n": 0}

    def slow_handler(event: HookEvent) -> None:
        slow_calls["n"] += 1
        _time.sleep(0.03)

    def fast_handler(event: HookEvent) -> None:
        fast_calls["n"] += 1

    hooks = LoopHooks(slow_budget_s=0.01, max_slow_strikes=2)
    hooks.add(slow_handler).add(fast_handler)

    # Enough scripted cycles that the slow handler strikes out mid-run.
    _run_loop(
        [_TOOL_REPLY, _TOOL_REPLY, _FINAL_REPLY],
        hooks=hooks,
        on_step=lambda s, d: flat.append((s, d)),
    )

    slow_events = [d for s, d in flat if s == "hook_slow"]
    disable_errors = [d for s, d in flat if s == "hook_error" and "DISABLED" in str(d.get("error"))]
    assert slow_events, "over-budget calls must surface as hook_slow"
    assert disable_errors, "the strike cap must disable the handler loudly"
    # Disabled means STOPPED (main dispatches AND notifications): the slow
    # handler's total calls = max_slow_strikes main calls + the notifications
    # it received before benching — strictly fewer than the fast handler,
    # which kept firing for the whole run.
    assert slow_calls["n"] < fast_calls["n"]
    # And the bench happened mid-run: the fast handler saw turn_end, the
    # slow one never did (it was disabled cycles earlier).
    assert slow_calls["n"] <= 4


def test_codeact_and_memact_dispatch_the_same_contract() -> None:
    """The hook layer is one shared source across the three adapters — pin
    that the sibling factories accept hooks and dispatch canonical events."""
    from abstractagent.adapters.codeact_runtime import create_codeact_workflow
    from abstractagent.adapters.memact_runtime import create_memact_workflow
    from abstractagent.logic.codeact import CodeActLogic
    from abstractagent.logic.memact import MemActLogic

    for factory, logic in (
        (create_codeact_workflow, CodeActLogic(tools=[])),
        (create_memact_workflow, MemActLogic(tools=[])),
    ):
        events: List[HookEvent] = []
        hooks = LoopHooks().add(events.append)

        def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
            del run, effect, default_next_node
            return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

        runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: llm_handler},
        )
        workflow = factory(logic=logic, on_step=None, hooks=hooks)
        run_id = runtime.start(
            workflow=workflow,
            vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}},
            actor_id=None,
            session_id="sess-sibling",
        )
        for _ in range(60):
            state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
            if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                break
        assert runtime.get_state(run_id).status == RunStatus.COMPLETED
        assert "cycle_start" in {e.name for e in events}, factory.__name__
