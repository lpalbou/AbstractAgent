"""Backlog 0026 follow-ups (loop-hooks wave, 2026-07-15), through the REAL kernel.

Pins the ungated 0026 remainder:

- `parse_tool_calls` fires on ALL THREE loops at the tool-batch commit point,
  so the canonical `tool_proposed` event is loop-independent;
- every loop emits `init` once at workflow entry (ReAct gained the run-init
  emit; the founding "ReAct emits no init" asymmetry is closed);
- durable-inbox guidance landing AFTER the loop's last drain point (e.g.
  `inject_guidance` while the final/conclusion LLM call is in flight) is
  reported as a loud `inbox_undelivered` terminal event — never a silent
  completion — and the entries stay in the run's durable vars (read-only
  honesty: the record shows what never got delivered);
- on ReAct's forced-conclusion path both halves hold: guidance drained BEFORE
  the conclusion dispatch still influences (it rides the conclusion prompt as
  "Host guidance:"), guidance arriving DURING the conclusion call is loud;
- composition handoffs (`final_next_node`) do NOT fire `inbox_undelivered` —
  the continuing run drains the inbox at its next reason boundary.

(The act-payload mutation pin — 0026's fourth item — lives in
tests/test_loop_hooks.py::test_handler_mutation_of_event_data_cannot_corrupt_the_loop,
rewritten from its vacuous original in the same wave.)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.loop_hooks import HookEvent, LoopHooks
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
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
# MemAct's mandatory finalize call consumes one extra scripted response; a
# JSON envelope keeps that leg on the clean path (no draft fallback noise).
_MEMACT_FINALIZE_REPLY = {"content": json.dumps({"content": "Done."}), "tool_calls": []}

_PROBE_TOOL = ToolDefinition(name="probe", description="Probe", parameters={})


def _make_workflow(loop: str, *, hooks: Optional[LoopHooks], on_step: Optional[Any]):
    if loop == "react":
        return create_react_workflow(
            logic=ReActLogic(tools=[_PROBE_TOOL]), hooks=hooks, on_step=on_step,
            provider="stub", model="stub",
        )
    if loop == "codeact":
        return create_codeact_workflow(logic=CodeActLogic(tools=[_PROBE_TOOL]), hooks=hooks, on_step=on_step)
    if loop == "memact":
        return create_memact_workflow(logic=MemActLogic(tools=[_PROBE_TOOL]), hooks=hooks, on_step=on_step)
    raise AssertionError(f"unknown loop {loop}")


def _drive(
    loop: str,
    llm_responses: List[Dict[str, Any]],
    *,
    hooks: Optional[LoopHooks] = None,
    on_step: Optional[Any] = None,
    vars_extra: Optional[Dict[str, Any]] = None,
    llm_side_effect: Optional[Callable[[RunState], None]] = None,
) -> Tuple[List[Dict[str, Any]], RunState]:
    """Run one scripted loop to a terminal; returns (llm payloads, final state).

    `llm_side_effect(run)` runs inside the LLM handler against the LIVE run —
    the same access an inject_guidance write has while a call is in flight.
    """
    llm_payloads: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(json.loads(json.dumps(payload)))
        idx = min(calls["n"], len(llm_responses) - 1)
        calls["n"] += 1
        if llm_side_effect is not None:
            llm_side_effect(run)
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
    workflow = _make_workflow(loop, hooks=hooks, on_step=on_step)
    vars: Dict[str, Any] = {"context": {"task": "Follow-up wave task", "messages": []}, "_runtime": {"inbox": []}}
    if vars_extra:
        vars.update(vars_extra)
    run_id = runtime.start(workflow=workflow, vars=vars, actor_id=None, session_id=f"sess-0026-{loop}")
    for _ in range(120):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED, f"{loop}: {state.error}"
    return llm_payloads, state


def _scripts_for(loop: str, *, with_tool: bool) -> List[Dict[str, Any]]:
    script: List[Dict[str, Any]] = [_TOOL_REPLY] if with_tool else []
    script.append(_FINAL_REPLY)
    if loop == "memact":
        script.append(_MEMACT_FINALIZE_REPLY)
    return script


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_init_and_tool_proposed_fire_on_all_three_loops(loop: str) -> None:
    """Parity pins: `init` announces the run's task once at entry on every
    loop, and the canonical `tool_proposed` (raw `parse_tool_calls`, payload
    `{"count": N}`) fires at the tool-batch commit point on every loop."""
    events: List[HookEvent] = []
    hooks = LoopHooks().add(events.append)

    _drive(loop, _scripts_for(loop, with_tool=True), hooks=hooks)

    inits = [e for e in events if e.step == "init"]
    assert len(inits) == 1, f"{loop}: init must fire exactly once per run"
    assert inits[0].data.get("task") == "Follow-up wave task"

    proposed = [e for e in events if e.name == "tool_proposed"]
    assert proposed, f"{loop}: tool_proposed must fire when a tool batch commits"
    assert proposed[0].step == "parse_tool_calls"
    assert proposed[0].data.get("count") == 1

    names = [e.name for e in events]
    assert names.index("init") < names.index("cycle_start") < names.index("tool_proposed")


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_terminal_undelivered_inbox_is_loud_on_all_three_loops(loop: str) -> None:
    """Guidance landing in the durable inbox while an LLM call is in flight
    can no longer influence the run — the terminal must say so loudly
    (`inbox_undelivered`) and must NOT consume the entries (the durable vars
    keep the record of what never got delivered)."""
    flat: List[Tuple[str, Dict[str, Any]]] = []

    def inject_late_guidance(run: RunState) -> None:
        runtime_ns = run.vars.get("_runtime")
        assert isinstance(runtime_ns, dict)
        inbox = runtime_ns.setdefault("inbox", [])
        inbox.append({"content": "late guidance"})

    _, state = _drive(
        loop,
        _scripts_for(loop, with_tool=False),
        on_step=lambda s, d: flat.append((s, d)),
        llm_side_effect=inject_late_guidance,
    )

    undelivered = [d for s, d in flat if s == "inbox_undelivered"]
    assert undelivered, f"{loop}: a terminal over undrained guidance must be loud"
    remaining = ((state.vars or {}).get("_runtime") or {}).get("inbox") or []
    remaining_texts = [e.get("content") for e in remaining if isinstance(e, dict)]
    assert "late guidance" in remaining_texts, f"{loop}: entries must stay in the durable record"
    assert undelivered[0].get("count") == len(remaining_texts)
    assert undelivered[0].get("chars", 0) >= len("late guidance")


def test_react_conclusion_surfaces_early_guidance_and_reports_late_guidance() -> None:
    """The 0026 conclude-phase case, both halves on one run: guidance drained
    at the conclusion boundary INFLUENCES (it rides the conclusion prompt as
    "Host guidance:"), guidance arriving DURING the conclusion call is
    reported undelivered at the terminal — neither is silent."""
    flat: List[Tuple[str, Dict[str, Any]]] = []
    entry_counter = {"n": 0}

    def inject_numbered_guidance(run: RunState) -> None:
        entry_counter["n"] += 1
        runtime_ns = run.vars.get("_runtime")
        assert isinstance(runtime_ns, dict)
        runtime_ns.setdefault("inbox", []).append({"content": f"guidance entry {entry_counter['n']}"})

    conclusion_reply = {"content": "Wrapped up: best-effort report.", "tool_calls": []}
    payloads, state = _drive(
        "react",
        [_TOOL_REPLY, conclusion_reply],
        on_step=lambda s, d: flat.append((s, d)),
        vars_extra={"_limits": {"max_iterations": 1}},
        llm_side_effect=inject_numbered_guidance,
    )

    # Call 1 (the tool cycle) deposited entry 1; the conclusion boundary
    # drained it into the conclusion prompt — the influencing half.
    assert len(payloads) == 2
    conclusion_texts = [
        str(m.get("content") or "") for m in (payloads[1].get("messages") or []) if isinstance(m, dict)
    ] + [str(payloads[1].get("prompt") or "")]
    assert any("Host guidance:" in t and "guidance entry 1" in t for t in conclusion_texts), (
        "guidance drained before the conclusion dispatch must ride the conclusion prompt"
    )
    assert any(s == "inbox_drained" for s, _ in flat)

    # Call 2 (the conclusion call itself) deposited entry 2 — too late to
    # influence; the terminal reports it and keeps the durable record.
    undelivered = [d for s, d in flat if s == "inbox_undelivered"]
    assert undelivered and undelivered[0].get("count") == 1
    remaining = ((state.vars or {}).get("_runtime") or {}).get("inbox") or []
    remaining_texts = [e.get("content") for e in remaining if isinstance(e, dict)]
    assert remaining_texts == ["guidance entry 2"]
    assert (state.output or {}).get("outcome") == "iteration_budget"


def test_composition_handoff_does_not_report_undelivered_inbox() -> None:
    """With `final_next_node` the RUN continues past the turn's terminal —
    late guidance is still deliverable at the next reason boundary, so the
    handoff must NOT fire `inbox_undelivered` (and must not consume it)."""
    flat: List[Tuple[str, Dict[str, Any]]] = []

    def inject_late_guidance(run: RunState) -> None:
        runtime_ns = run.vars.get("_runtime")
        assert isinstance(runtime_ns, dict)
        runtime_ns.setdefault("inbox", []).append({"content": "late guidance"})

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        inject_late_guidance(run)
        return EffectOutcome.completed(dict(_FINAL_REPLY))

    def after_node(run: RunState, ctx: Any):
        del ctx
        from abstractruntime import StepPlan

        return StepPlan(node_id="after", complete_output={"handed_off": True})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[_PROBE_TOOL]),
        on_step=lambda s, d: flat.append((s, d)),
        provider="stub",
        model="stub",
        final_next_node="after",
    )
    workflow.nodes["after"] = after_node
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-0026-handoff",
    )
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    assert (state.output or {}).get("handed_off") is True

    assert not any(s == "inbox_undelivered" for s, _ in flat), (
        "a composition handoff must not report deliverable guidance as undelivered"
    )
    remaining = ((state.vars or {}).get("_runtime") or {}).get("inbox") or []
    assert any(
        isinstance(e, dict) and e.get("content") == "late guidance" for e in remaining
    ), "the handoff must leave the guidance for the continuing run's next drain"
