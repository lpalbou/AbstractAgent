"""Injected guidance must persist in the ReAct transcript (maintainer ruling 2026-07-09).

Before this ruling, drained `_runtime.inbox` guidance rode a one-shot ephemeral tail:
visible to exactly one LLM call, gone the next cycle — so a final answer written any
cycle later re-anchored on the original task and dropped the operator's correction.

Drives a scripted 3-iteration ReAct loop through the REAL runtime and asserts:
- guidance drained at cycle N is visible in cycle N's LLM payload AND every later one;
- it lands in the durable transcript as a user message (metadata.kind=operator_guidance);
- the inbox is emptied by the drain (single delivery, durable visibility);
- the old `[guidance]` ephemeral tail is gone;
- guidance arriving before the first assistant turn does not produce consecutive user
  turns in the payload (alternation-strict templates 400 on user,user);
- the append stays prefix-cache safe (0212): cycle N's stable prefix reappears at the
  head of cycle N+1.
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

GUIDANCE = "Correction: also report the disk usage; the final answer must mention DISK USAGE."


def _build_runtime(captured: List[Dict[str, Any]]) -> Runtime:
    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed(
                {
                    "content": "Looking around.",
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
                }
            )
        if iteration == 2:
            return EffectOutcome.completed(
                {
                    "content": "One more check.",
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "sub"}, "call_id": "call_2"}],
                }
            )
        return EffectOutcome.completed({"content": "All done.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )


def _workflow():
    return create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="react_guidance_durable",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )


def _drive(runtime: Runtime, workflow, run_id: str, *, inject_after_first_llm_call: bool, captured: List[Dict[str, Any]]) -> RunState:
    injected = False
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if inject_after_first_llm_call and not injected and len(captured) >= 1:
            # Same mutation the gateway inject_guidance command / BaseAgent.inject_message perform.
            run = runtime.get_state(run_id)
            run.vars["_runtime"]["inbox"].append({"role": "system", "content": GUIDANCE})
            runtime.run_store.save(run)
            injected = True
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return state


def _messages_with(payload: Dict[str, Any], needle: str) -> List[Dict[str, Any]]:
    return [
        m
        for m in (payload.get("messages") or [])
        if isinstance(m, dict) and needle in str(m.get("content") or "")
    ]


def test_midrun_guidance_is_visible_on_every_subsequent_cycle() -> None:
    captured: List[Dict[str, Any]] = []
    runtime = _build_runtime(captured)
    workflow = _workflow()
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "Summarize the workspace", "messages": []}, "_runtime": {"inbox": []}},
        session_id="sess-guidance",
    )
    state = _drive(runtime, workflow, run_id, inject_after_first_llm_call=True, captured=captured)
    assert len(captured) == 3

    # Cycle 1 predates the injection: guidance absent.
    assert not _messages_with(captured[0], GUIDANCE)

    # Cycle 2 drains it; cycle 3 must STILL see it (the pre-ruling ephemeral tail lost it here).
    for payload in captured[1:]:
        hits = _messages_with(payload, GUIDANCE)
        assert hits, "guidance missing from LLM payload"
        assert all(m.get("role") == "user" for m in hits)
        assert any("[Operator guidance" in str(m.get("content") or "") for m in hits)

    # The old ephemeral tail marker is gone everywhere.
    for payload in captured:
        assert not _messages_with(payload, "[guidance]")

    # Durable transcript carries exactly one operator_guidance user message; inbox is drained.
    context = state.vars.get("context") or {}
    durable = [
        m
        for m in (context.get("messages") or [])
        if isinstance(m, dict) and (m.get("metadata") or {}).get("kind") == "operator_guidance"
    ]
    assert len(durable) == 1
    assert durable[0].get("role") == "user"
    assert GUIDANCE in str(durable[0].get("content") or "")
    assert state.vars.get("_runtime", {}).get("inbox") == []

    # Prefix-cache safety (0212): cycle 2's stable prefix reappears byte-identical at the
    # head of cycle 3 — the durable guidance append is append-only.
    stable_prefix = (captured[1].get("messages") or [])[:-1]
    head = (captured[2].get("messages") or [])[: len(stable_prefix)]
    assert json.dumps(head, sort_keys=True) == json.dumps(stable_prefix, sort_keys=True)


def test_guidance_before_first_assistant_turn_never_yields_consecutive_user_turns() -> None:
    captured: List[Dict[str, Any]] = []
    runtime = _build_runtime(captured)
    workflow = _workflow()
    run_id = runtime.start(
        workflow=workflow,
        vars={
            "context": {"task": "Summarize the workspace", "messages": []},
            # Guidance queued before the run's first cycle: durable history would hold
            # user(task), user(guidance) — the payload boundary must merge them.
            "_runtime": {"inbox": [{"role": "system", "content": GUIDANCE}]},
        },
        session_id="sess-guidance-early",
    )
    _drive(runtime, workflow, run_id, inject_after_first_llm_call=False, captured=captured)

    for payload in captured:
        msgs = payload.get("messages") or []
        roles = [m.get("role") for m in msgs]
        assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), f"user,user adjacency in payload: {roles}"

    # Both the task and the guidance are visible on every cycle.
    for payload in captured:
        assert _messages_with(payload, "Summarize the workspace")
        assert _messages_with(payload, GUIDANCE)
