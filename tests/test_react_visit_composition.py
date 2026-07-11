"""Visit-workflow composition: the ReAct cycle embeds between seam nodes.

Contract source: frozen visit seam spec (a2a 0013 §A) + runtime's visit workflow
draft v0 (a2a 0014, TURN chain: RECALL → reason/act/observe [this adapter] →
ELECTIONS → COMMIT → FORM → ANSWER → PARK). The adapter's composition knob:

- `create_react_workflow(final_next_node=...)`: `done`/`max_iterations` finish the
  TURN, not the run — they persist the final answer to the durable transcript
  exactly as today, stash the output dict at `_temp.react_output`, and hand off to
  the named seam node;
- `reset_react_turn(run.vars)`: per-turn budget semantics — a re-entered `reason`
  starts with a fresh iteration counter and clean `_temp`, while the durable life
  (transcript, scratchpad.cycles) is untouched;
- multi-turn continuity: turn 2 sees turn 1's transcript (memory of the visit so
  far) and the prefix stays append-only.

These tests build a miniature visit workflow: my nodes + an `elections` seam node
+ a `park` seam node that injects the next visitor message, mirroring the 0014
draft's shape through the real runtime.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus, StepPlan
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.core.spec import WorkflowSpec
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


def _build_visit_workflow(visitor_messages: List[str]) -> Tuple[WorkflowSpec, Dict[str, Any]]:
    """My ReAct nodes + two seam nodes (elections, park) in one workflow spec."""
    seen: Dict[str, Any] = {"turn_answers": [], "turn_outputs": []}

    react = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="entity_visit_turnchain",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
        final_next_node="elections",
    )

    def elections_node(run: RunState, ctx) -> StepPlan:
        del ctx
        temp = run.vars.get("_temp") or {}
        output = temp.get("react_output") or {}
        seen["turn_answers"].append(str(temp.get("final_answer") or ""))
        seen["turn_outputs"].append(json.loads(json.dumps(output)))
        return StepPlan(node_id="elections", next_node="park")

    def park_node(run: RunState, ctx) -> StepPlan:
        del ctx
        # Turn 1's message is the task seed (init); visitor_messages hold turns 2..N.
        completed_turns = len(seen["turn_answers"])
        msg_index = completed_turns - 1
        if msg_index >= len(visitor_messages):
            return StepPlan(node_id="park", complete_output={"turns": completed_turns})
        # Next visitor message arrives (stand-in for the WAIT_EVENT resume):
        # reset per-turn state, append the message, re-enter reason.
        reset_react_turn(run.vars)
        context = run.vars.get("context") or {}
        msgs = context.get("messages")
        if isinstance(msgs, list):
            msgs.append({"role": "user", "content": visitor_messages[msg_index]})
        return StepPlan(node_id="park", next_node="reason")

    nodes = dict(react.nodes)
    nodes["elections"] = elections_node
    nodes["park"] = park_node
    return WorkflowSpec(workflow_id=react.workflow_id, entry_node=react.entry_node, nodes=nodes), seen


def _run_visit(
    llm_script: List[Dict[str, Any]],
    visitor_messages: List[str],
    *,
    max_iterations: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], RunState]:
    captured: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        idx = min(calls["n"], len(llm_script) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(llm_script[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    workflow, seen = _build_visit_workflow(visitor_messages)
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    vars: Dict[str, Any] = {"context": {"task": "First visitor message", "messages": []}, "_runtime": {"inbox": []}}
    if max_iterations is not None:
        vars["_limits"] = {"max_iterations": int(max_iterations)}
    run_id = runtime.start(workflow=workflow, vars=vars, actor_id=None, session_id="sess-visit")
    for _ in range(120):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, seen, state


def test_final_next_node_hands_off_instead_of_completing() -> None:
    captured, seen, state = _run_visit(
        [{"content": "Turn one answer.", "tool_calls": []}],
        visitor_messages=[],
    )
    # The seam node received the turn's answer + full output dict.
    assert seen["turn_answers"] == ["Turn one answer."]
    assert seen["turn_outputs"][0]["answer"] == "Turn one answer."
    assert seen["turn_outputs"][0]["iterations"] == 1
    # The RUN completed at the seam's park node, not at my done node.
    assert (state.output or {}).get("turns") == 1
    assert len(captured) == 1


def test_two_turn_visit_fresh_budget_and_durable_continuity() -> None:
    llm_script = [
        # Turn 1: one tool cycle, then answer.
        {
            "content": "Looking around.",
            "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}],
        },
        {"content": "Turn one answer.", "tool_calls": []},
        # Turn 2: direct answer.
        {"content": "Turn two answer — I remember the start.", "tool_calls": []},
    ]
    captured, seen, state = _run_visit(llm_script, visitor_messages=["Second visitor message"])

    assert seen["turn_answers"] == ["Turn one answer.", "Turn two answer — I remember the start."]
    # Fresh per-turn budget: turn 2's output reports iteration 1, not 3.
    assert seen["turn_outputs"][1]["iterations"] == 1
    assert (state.output or {}).get("turns") == 2

    # Durable continuity: turn 2's LLM payload carries turn 1's transcript
    # (task, assistant tool-call turn, tool result, final answer, new message).
    turn2_payload = captured[2]
    roles = [m.get("role") for m in turn2_payload.get("messages", [])]
    assert roles.count("assistant") >= 2  # turn-1 tool-call + turn-1 final answer
    contents = json.dumps(turn2_payload.get("messages", []))
    assert "Turn one answer." in contents
    assert "Second visitor message" in contents

    # Append-only prefix across the turn boundary: turn 2's message list starts
    # with turn 1's final payload messages minus its volatile trailing merge.
    turn1_final_payload = captured[1]
    stable_prefix = turn1_final_payload.get("messages", [])[:-1]
    turn2_msgs = turn2_payload.get("messages", [])
    assert json.dumps(turn2_msgs[: len(stable_prefix)], sort_keys=True) == json.dumps(stable_prefix, sort_keys=True)


def test_turn_id_rides_llm_payload_when_set() -> None:
    """G1 write direction (runtime b8b8c78): the result-boundary election capture
    requires turn_id ON the LLM_CALL payload. The embedding workflow sets
    `_runtime.turn_id` per turn; my reason node passes it through. Absent = absent
    (non-visit runs unchanged)."""
    captured: List[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        captured.append(json.loads(json.dumps(effect.payload or {})))
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    react = create_react_workflow(
        logic=ReActLogic(tools=[]),
        workflow_id="react_turn_id",
        provider="stub",
        model="stub",
    )
    run_id = runtime.start(
        workflow=react,
        vars={
            "context": {"task": "One turn", "messages": []},
            "_runtime": {
                "inbox": [],
                "turn_id": "t-0007",
                "llm_payload_extras": {"anchor_record_ids": ["r1"], "anchor_graph_ids": ["ex:m1"]},
            },
        },
        actor_id=None,
        session_id="sess-turnid",
    )
    for _ in range(30):
        state = runtime.tick(workflow=react, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    assert captured[0].get("turn_id") == "t-0007"
    # The wrapper-consumed payload extras ride through too (word-free anchors for the
    # G1 election capture); existing payload keys are never overwritten by extras.
    assert captured[0].get("anchor_record_ids") == ["r1"]
    assert captured[0].get("anchor_graph_ids") == ["ex:m1"]

    # Control: without _runtime.turn_id the key is absent from the payload.
    captured.clear()
    run_id2 = runtime.start(
        workflow=react,
        vars={"context": {"task": "One turn", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-turnid-2",
    )
    for _ in range(30):
        state = runtime.tick(workflow=react, run_id=run_id2, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert "turn_id" not in captured[0]


def test_turn_captures_accumulate_across_iterations() -> None:
    """Mid-loop election capture (G1 write direction): the entity runtime's LLM
    wrapper returns word-free `diary_entries` metadata on EACH result; when the model
    elects mid-loop (iteration 1) and answers later (iteration 2), the per-iteration
    result is overwritten — my parse node accumulates captures at `_temp.turn_captures`
    so the embedding ELECT/FORM nodes fold ALL of them."""
    captured: List[Dict[str, Any]] = []
    calls = {"n": 0}
    scripted = [
        {
            "content": "Noted. [kept a diary entry]",
            "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}],
            "diary_entries": [{"entry_id": "diary_01", "kind": "note", "visibility": "private"}],
            "act_only_warnings": [],
        },
        {
            "content": "Turn answer.",
            "tool_calls": [],
            "diary_entries": [{"entry_id": "diary_02", "kind": "note", "visibility": "public", "gist": "second"}],
            "act_only_warnings": ["#FALLBACK one warning"],
        },
    ]

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        captured.append(json.loads(json.dumps(effect.payload or {})))
        idx = min(calls["n"], len(scripted) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(scripted[idx]))

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
    react = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="react_captures",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
        final_next_node="harvest",
    )
    harvested: Dict[str, Any] = {}

    def harvest_node(run: RunState, ctx) -> StepPlan:
        del ctx
        temp = run.vars.get("_temp") or {}
        harvested.update(json.loads(json.dumps(temp.get("turn_captures") or {})))
        return StepPlan(node_id="harvest", complete_output={"ok": True})

    nodes = dict(react.nodes)
    nodes["harvest"] = harvest_node
    wf = WorkflowSpec(workflow_id=react.workflow_id, entry_node=react.entry_node, nodes=nodes)

    run_id = runtime.start(
        workflow=wf,
        vars={"context": {"task": "Visit turn", "messages": []}, "_runtime": {"inbox": [], "turn_id": "t-0001"}},
        actor_id=None,
        session_id="sess-captures",
    )
    for _ in range(60):
        state = runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED

    # Both iterations' captures arrived at the seam — mid-loop election included.
    entry_ids = [e.get("entry_id") for e in harvested.get("diary_entries", [])]
    assert entry_ids == ["diary_01", "diary_02"]
    assert harvested.get("act_only_warnings") == ["#FALLBACK one warning"]


def test_budget_exhaustion_also_hands_off_to_the_seam() -> None:
    llm_script = [
        {
            "content": "Working.",
            "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}],
        },
        {"content": "Out of budget — concluding the turn.", "tool_calls": []},
    ]
    captured, seen, state = _run_visit(llm_script, visitor_messages=[], max_iterations=1)
    # The conclusion (max_iterations path) handed off to elections, not completed the run.
    assert len(seen["turn_answers"]) == 1
    assert "concluding the turn" in seen["turn_answers"][0]
    assert (state.output or {}).get("turns") == 1
    assert len(captured) == 2  # one loop call + one conclusion call
