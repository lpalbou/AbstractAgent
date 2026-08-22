"""`/conclude` — ending a long turn WELL, from any client.

Operator ruling 2026-08-21: between `pause` (freeze) and `cancel` (throw the
work away) there was nothing that lets a human say "you have enough, land it".
`POST /commands {type: "conclude"}` rides the existing durable steer lane —
gateway → `Runtime.steer()` → steer sidecar → the run's own tick drains it
into `_runtime.inbox` — carrying `kind: "conclude"` on the message, which the
sidecar preserves verbatim.

The loop must then stop reasoning at its next boundary and run the tool-free
conclusion it already owns, and the turn's verdict must name the OPERATOR as
the cause: not a failure, and not a budget that was never spent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic

_READ = {"content": "Reading.", "tool_calls": [{"name": "read_file", "arguments": {"path": "a"}, "call_id": "c1"}]}
_ANSWER = {"content": "All done.", "tool_calls": []}


def _drive(script: List[Dict[str, Any]], *, conclude_after: Optional[int] = None, note: str = ""):
    """Run the loop, delivering a conclude message into the inbox after N LLM calls
    (which is what the runtime's sidecar drain does at a real boundary)."""
    events: List[tuple] = []
    payloads: List[Dict[str, Any]] = []
    call_n = {"n": 0}
    state_ref: Dict[str, Any] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payloads.append(json.loads(json.dumps(effect.payload, default=str)))
        call_n["n"] += 1
        if conclude_after is not None and call_n["n"] == conclude_after:
            rt_ns = run.vars.setdefault("_runtime", {})
            inbox = rt_ns.setdefault("inbox", [])
            inbox.append(
                {
                    "role": "system",
                    "kind": "conclude",
                    "content": "The operator asked you to conclude now.",
                    "note": note,
                }
            )
            state_ref["delivered_at"] = call_n["n"]
        idx = min(call_n["n"] - 1, len(script) - 1)
        return EffectOutcome.completed(dict(script[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
                    for tc in (payload.get("tool_calls") or [])
                ],
            }
        )

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]),
        on_step=lambda s, d: events.append((s, json.loads(json.dumps(d, default=str)))),
        workflow_id="wf-conclude",
        provider="stub",
        model="stub",
        allowed_tools=["read_file"],
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Do a long thing", "messages": []},
            "_runtime": {"inbox": []},
            "_limits": {"max_iterations": 20},
        },
    )
    for _ in range(200):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    return events, payloads, rt.get_state(rid)


def test_conclude_ends_the_turn_at_the_next_boundary_with_an_operator_verdict() -> None:
    events, payloads, state = _drive([_READ] * 10 + [_ANSWER], conclude_after=2)

    asked = [d for (s, d) in events if s == "conclude_requested"]
    assert len(asked) == 1, "the loop announces that it is concluding on request"
    assert asked[0]["has_note"] is False

    out = state.output or {}
    sr = out.get("stop_reason") or {}
    assert sr.get("code") == "operator_conclude"
    assert sr.get("finished") is False
    # NOT a budget stop: the iterations were never spent, and telling an
    # operator to raise a budget they interrupted would be nonsense.
    assert sr.get("budget_exhausted") is False
    assert "concluded on request" in str(sr.get("label"))
    assert "asked the agent to conclude" in str(sr.get("headline"))
    assert isinstance(out.get("concluded_by_operator"), dict)

    # It stopped EARLY: nowhere near the 20-iteration budget.
    assert int(out.get("iterations") or 0) <= 4, out.get("iterations")

    # The model was told plainly, and told not to fake completion.
    concl = json.dumps(payloads[-1])
    assert "asked you to CONCLUDE NOW" in concl
    assert "Do not pretend the remaining work is done" in concl
    assert "Do NOT call tools" in concl


def test_an_operator_note_reaches_the_model_verbatim() -> None:
    _events, payloads, state = _drive(
        [_READ] * 6 + [_ANSWER], conclude_after=1, note="just the table, skip the analysis"
    )
    concl = json.dumps(payloads[-1])
    assert "just the table, skip the analysis" in concl
    assert (state.output or {}).get("concluded_by_operator", {}).get("note") == "just the table, skip the analysis"


def test_without_a_conclude_the_turn_runs_normally() -> None:
    """The check must be inert when nobody asked: no event, no verdict."""
    events, _payloads, state = _drive([_READ, _ANSWER])
    assert not [d for (s, d) in events if s == "conclude_requested"]
    out = state.output or {}
    assert (out.get("stop_reason") or {}).get("code") == "final_answer"
    assert "concluded_by_operator" not in out


def test_a_conclude_message_is_not_replayed_as_ordinary_guidance() -> None:
    """It is REMOVED from the inbox: leaving it there would re-inject the
    directive into the conversation as if the operator had typed it."""
    from abstractagent.adapters.react_runtime import _take_conclude_request

    ns = {"inbox": [
        {"role": "system", "content": "keep going"},
        {"role": "system", "kind": "conclude", "content": "wrap up", "note": "n"},
    ]}
    assert _take_conclude_request(ns) == "n"
    assert ns["inbox"] == [{"role": "system", "content": "keep going"}]
    # And it is a one-shot: a second look finds nothing.
    assert _take_conclude_request(ns) is None
    assert _take_conclude_request({"inbox": []}) is None
    assert _take_conclude_request({}) is None
