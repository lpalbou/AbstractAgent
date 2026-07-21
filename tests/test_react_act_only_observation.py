"""Tool observations render as PLAIN SERVED CONTENT — the act-only ref layer is dead.

History: this file pinned the `$act_only` ref-minting contract (frozen visit
seam spec, a2a 0013 v2 §2 — refs in the durable transcript, dereferenced at
send time by abstractruntime/identity/act_only.py). Under laurent's A ruling
(2026-07-20, relayed c3337: "everything lives in the runtime, diary = the
AI's experiential notes"), runtime DELETED the send-time dereference and the
adapter deleted its minting half — a minted ref would rest as literal JSON
nothing resolves. The HOME is the privacy boundary; the diary book's
sole-author property lives in the WRITE-boundary capture (runtime's wrapper),
which was never in this adapter.

These pins guard against RESURRECTION of the ref layer and pin the plain
rendering that replaced it.
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


def _run_scripted_loop(
    llm_script: List[Dict[str, Any]],
    tool_outputs: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], RunState]:
    captured: List[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        idx = min(max(iteration - 1, 0), len(llm_script) - 1)
        return EffectOutcome.completed(dict(llm_script[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = []
        for tc in payload.get("tool_calls") or []:
            call_id = str(tc.get("call_id") or "")
            spec = tool_outputs.get(call_id, {"success": True, "output": "ok", "error": None})
            results.append(
                {
                    "call_id": call_id,
                    "name": tc.get("name"),
                    "success": bool(spec.get("success", True)),
                    "output": spec.get("output"),
                    "error": spec.get("error"),
                }
            )
        return EffectOutcome.completed({"mode": "executed", "results": results})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    workflow = create_react_workflow(
        logic=ReActLogic(
            tools=[
                # act_only=True on the DEFINITION is now render-inert: the field may
                # persist in core's ToolDefinition, but the adapter no longer keys
                # any suppression on it.
                ToolDefinition(name="diary_read", description="Read a diary entry", parameters={}, act_only=True),
                ToolDefinition(name="diary_list", description="List diary entries", parameters={}),
            ]
        ),
        workflow_id="react_plain_obs",
        provider="stub",
        model="stub",
        allowed_tools=["diary_read", "diary_list"],
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "Reconnect with the diary", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-plain-obs",
    )
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, state


def _durable_tool_messages(state: RunState) -> List[Dict[str, Any]]:
    context = state.vars.get("context") if isinstance(state.vars, dict) else {}
    msgs = context.get("messages") if isinstance(context, dict) else []
    return [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]


def test_served_diary_content_rests_plainly_in_the_transcript() -> None:
    """The ruling's positive half: what the home SERVES is what rests — the
    entity re-reads its own past without a resolver in the way."""
    llm_script = [
        {
            "content": "Reaching for the entry.",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_ab12"}, "call_id": "call_1"}],
        },
        {"content": "Done reconnecting.", "tool_calls": []},
    ]
    served = "2026-07-19 — a quiet line about the garden."
    tool_outputs = {"call_1": {"success": True, "output": served, "error": None}}
    payloads, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    assert len(tool_msgs) == 1
    content = str(tool_msgs[0].get("content"))
    assert content == f"[diary_read]: {served}"
    # No act_only metadata marker survives on the message.
    assert "act_only" not in (tool_msgs[0].get("metadata") or {})
    # The same plain bytes ride the next LLM payload.
    payload_tool_msgs = [m for m in payloads[1].get("messages", []) if m.get("role") == "tool"]
    assert payload_tool_msgs[0].get("content") == content


def test_no_ref_shape_is_ever_minted() -> None:
    """Resurrection guard: even a handler still emitting the OLD lone-key
    frame shape gets plain dict rendering — the adapter never re-mints the
    `$act_only` REF CONTRACT (canonical sorted-key JSON whose bytes a
    dereference pass would parse-and-resolve). The dict's repr resting
    plainly is the ruling working: nothing resolves, nothing wedges."""
    frame = {"tool": "diary_read", "entry_id": "diary_ab12", "gist": "one line"}
    llm_script = [
        {
            "content": "Reading.",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_ab12"}, "call_id": "call_1"}],
        },
        {"content": "Finished.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": True, "output": {"$act_only": dict(frame)}, "error": None}}
    _, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    content = str(tool_msgs[0].get("content"))
    # Plain rendering of the dict (str()), NOT the canonical ref serialization
    # the dead resolver understood, and no act_only message metadata.
    assert content.startswith("[diary_read]: ")
    assert json.dumps({"$act_only": frame}, ensure_ascii=False, sort_keys=True) not in content
    assert "act_only" not in (tool_msgs[0].get("metadata") or {})


def test_declared_act_only_field_is_render_inert() -> None:
    """ToolDefinition.act_only no longer keys suppression: a failed call
    renders its error like any tool (the old branch replaced served/failed
    output with frame records)."""
    llm_script = [
        {
            "content": "Reading.",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_zz99"}, "call_id": "call_1"}],
        },
        {"content": "Understood.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": False, "output": None, "error": "diary door refused: asleep"}}
    _, state = _run_scripted_loop(llm_script, tool_outputs)

    content = str(_durable_tool_messages(state)[0].get("content"))
    assert "act-only record" not in content
    assert "diary door refused: asleep" in content
