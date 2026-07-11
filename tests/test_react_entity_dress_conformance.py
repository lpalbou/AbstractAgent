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
