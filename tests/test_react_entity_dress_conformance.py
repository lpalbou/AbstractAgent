"""Entity-dress conformance: the visit loop is configuration on the existing ReAct loop.

Contract source: frozen visit seam spec (a2a/threads/0013-visit-seam-spec, v2 §4):
"ONE loop in entity dress: prelude as system head; tier-1 grant = registry allowlist
deny-by-default; per-turn iteration budget rides `_limits` as a spec constant" and
"stable head = spark/prelude + static visit contract; volatile MEMORIES/presence ride
the message lane."

These tests pin the adapter knobs the entity dress rides, through the real runtime:

- `_runtime.system_prompt` (the prelude) fully REPLACES the ReAct persona and stays
  BYTE-IDENTICAL across iterations — the visit's cached prefix holds;
- `_runtime.allowed_tools = []` is DENY-BY-DEFAULT (zero tool specs reach the
  provider payload) and is distinct from the key being ABSENT (default = full
  registry) — the load-bearing distinction for tier-1 grants;
- `_limits.max_iterations` is honored as the per-turn budget (the conclusion path
  runs when the budget is exhausted).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


_PRELUDE = (
    "You are Castor. You are visiting with a friend.\n"
    "Your values: honesty, repair over perfection, curiosity.\n"
    "MEMORIES and presence arrive in the conversation, never here."
)


def _run_loop(
    llm_responses: List[Dict[str, Any]],
    *,
    runtime_ns: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, Any]] = None,
    allowed_tools: Optional[List[str]] = None,
    steps: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], RunState]:
    captured: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
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
    def on_step(step: str, data: Dict[str, Any]) -> None:
        if steps is not None:
            steps.append((step, data))

    workflow = create_react_workflow(
        logic=ReActLogic(
            tools=[
                ToolDefinition(name="diary_read", description="Read a diary entry", parameters={}, act_only=True),
                ToolDefinition(name="web_search", description="Search the web", parameters={}),
            ]
        ),
        workflow_id="react_entity_dress",
        provider="stub",
        model="stub",
        allowed_tools=allowed_tools,
        on_step=on_step if steps is not None else None,
    )
    ns: Dict[str, Any] = {"inbox": []}
    if runtime_ns:
        ns.update(runtime_ns)
    vars: Dict[str, Any] = {"context": {"task": "A visit turn", "messages": []}, "_runtime": ns}
    if limits:
        vars["_limits"] = dict(limits)
    run_id = runtime.start(workflow=workflow, vars=vars, actor_id=None, session_id="sess-dress")
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, state


_TOOL_LOOP_REPLY = {
    "content": "Looking.",
    "tool_calls": [{"name": "web_search", "arguments": {"query": "voyager"}, "call_id": "call_1"}],
}
_FINAL_REPLY = {"content": "It was good to sit with this.", "tool_calls": []}


def test_prelude_system_head_replaces_persona_and_stays_byte_identical() -> None:
    payloads, _ = _run_loop(
        [_TOOL_LOOP_REPLY, _TOOL_LOOP_REPLY, _FINAL_REPLY],
        runtime_ns={"system_prompt": _PRELUDE},
    )
    assert len(payloads) == 3
    heads = [str(p.get("system_prompt") or "") for p in payloads]
    # Full replacement: the prelude IS the head; no ReAct persona text leaks in.
    assert heads[0] == _PRELUDE
    assert "ReAct agent" not in heads[0]
    assert "MY PERSONA" not in heads[0]
    # Byte-identical across every iteration: the visit's cached prefix holds.
    assert heads[0] == heads[1] == heads[2]


def test_empty_allowlist_is_deny_by_default_and_distinct_from_absent() -> None:
    # Explicit empty allowlist (tier-1 grant not yet given): ZERO tool specs on the wire.
    payloads_deny, _ = _run_loop([_FINAL_REPLY], runtime_ns={"allowed_tools": []})
    tools_deny = payloads_deny[0].get("tools")
    assert not tools_deny  # empty or absent — nothing offered

    # Absent key: the default allowlist (full registry for this workflow) is offered.
    payloads_all, _ = _run_loop([_FINAL_REPLY])
    tools_all = payloads_all[0].get("tools") or []
    names_all = {str(t.get("name")) for t in tools_all if isinstance(t, dict)}
    assert {"diary_read", "web_search"} <= names_all

    # Narrow grant: exactly the granted tool is offered.
    payloads_one, _ = _run_loop([_FINAL_REPLY], runtime_ns={"allowed_tools": ["diary_read"]})
    tools_one = payloads_one[0].get("tools") or []
    names_one = {str(t.get("name")) for t in tools_one if isinstance(t, dict)}
    assert names_one == {"diary_read"}


def test_iteration_budget_rides_limits_and_conclusion_runs_on_exhaustion() -> None:
    # The model never stops calling tools; the budget must stop it.
    conclusion = {"content": "Budget reached — here is where we are.", "tool_calls": []}
    payloads, state = _run_loop(
        [_TOOL_LOOP_REPLY, _TOOL_LOOP_REPLY, conclusion],
        limits={"max_iterations": 2},
    )
    # Two loop calls, then the tool-free conclusion call (the max_iterations path).
    assert len(payloads) == 3
    output = state.output if isinstance(state.output, dict) else {}
    assert "Budget reached" in str(output.get("answer") or "")
    assert int(output.get("iterations") or 0) == 2


def test_unknown_granted_tool_prunes_visibly_never_silently() -> None:
    """Config-object conformance (works-or-loud): a grant naming a tool this
    adapter has NO definition for stays deny-safe (never offered, never
    executable) AND becomes VISIBLE — `_runtime.allowlist_pruned` records the
    dropped names durably so the door/operator can see the grant did not take.
    The registered half of the grant still resolves."""
    steps: List[Tuple[str, Dict[str, Any]]] = []
    payloads, state = _run_loop(
        [_FINAL_REPLY],
        runtime_ns={"allowed_tools": ["diary_read", "not_a_registered_tool"]},
        steps=steps,
    )
    # Offered set = the registered intersection only.
    tools = payloads[0].get("tools") or []
    names = {str(t.get("name")) for t in tools if isinstance(t, dict)}
    assert names == {"diary_read"}
    # The prune is durable, names exactly the dropped grant, and carries the
    # original requested list so a reader can judge the note's freshness.
    ns = (state.vars or {}).get("_runtime") or {}
    note = ns.get("allowlist_pruned")
    assert isinstance(note, dict)
    assert note["dropped"] == ["not_a_registered_tool"]
    assert note["requested"] == ["diary_read", "not_a_registered_tool"]
    # The emit fired exactly once (per prune event, not per cycle).
    prune_events = [d for s, d in steps if s == "allowlist_pruned"]
    assert len(prune_events) == 1
    assert prune_events[0]["dropped"] == ["not_a_registered_tool"]
    assert prune_events[0]["kept"] == ["diary_read"]
    # A fully-registered grant leaves no prune note behind.
    _, state_clean = _run_loop([_FINAL_REPLY], runtime_ns={"allowed_tools": ["diary_read"]})
    ns_clean = (state_clean.vars or {}).get("_runtime") or {}
    assert "allowlist_pruned" not in ns_clean


def test_non_name_grant_entries_are_loud_worst_case_full_deny() -> None:
    """A grant of non-name garbage (e.g. [None]) normalizes to a FULL DENY —
    that must never be silent. The prune note records the rejected entries as
    reprs under `invalid`."""
    payloads, state = _run_loop(
        [_FINAL_REPLY],
        runtime_ns={"allowed_tools": [None, "   "]},
    )
    assert not (payloads[0].get("tools") or [])  # deny-safe held
    ns = (state.vars or {}).get("_runtime") or {}
    note = ns.get("allowlist_pruned")
    assert isinstance(note, dict)
    assert note.get("invalid") == ["None", "'   '"]
    assert note["dropped"] == [] and note["requested"] == []


def test_factory_channel_grant_prunes_visibly_too() -> None:
    """The FACTORY channel (`create_react_workflow(allowed_tools=...)`) is the
    path the shipped entity door rides — gateway now passes the RAW resolver
    grant there (c802) and RELIES on this note firing for names the middle
    carries no definition for. Same works-or-loud contract as the run-var
    channel: the undeclarable name is never offered AND lands durably in
    `_runtime.allowlist_pruned`."""
    payloads, state = _run_loop(
        [_FINAL_REPLY],
        allowed_tools=["diary_read", "undeclarable_tool"],
    )
    tools = payloads[0].get("tools") or []
    names = {str(t.get("name")) for t in tools if isinstance(t, dict)}
    assert names == {"diary_read"}
    ns = (state.vars or {}).get("_runtime") or {}
    note = ns.get("allowlist_pruned")
    assert isinstance(note, dict)
    assert note["dropped"] == ["undeclarable_tool"]
    assert note["requested"] == ["diary_read", "undeclarable_tool"]
