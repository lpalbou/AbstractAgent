"""Sight lane (commons c3969 shape A / c4089 operator ruling): tool-result media fold.

An agent can CAPTURE media mid-loop (camera tools return paths or
``{"$artifact": id}`` refs) but could not SEE it — media entered a run only
at run start (`context.attachments`). These tests pin the consumer half:

- a SUCCESSFUL tool result whose dict output DECLARES ``media`` gets its refs
  folded into the NEXT reason call's ``payload.media`` (where runtime
  llm_client's existing ``$artifact`` resolution takes over);
- consumption is ONE-SHOT (image tokens ride exactly one call; the durable
  transcript keeps the textual ref — re-look = analyze_media or re-capture);
- declared-only, success-only (never sniffed from prose, never from failures);
- deduped against context attachments; burst-bounded most-recent-wins;
- per-turn state (reset_react_turn clears it);
- the max-iterations conclude call consumes pending refs when the budget wall
  lands right after a capture batch.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.media import (
    extract_media_from_tool_result,
    media_item_key,
    merge_media_lists,
    normalize_media_items,
)
from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


# ---------------------------------------------------------------------------
# Harness: scripted LLM (by call order) + scripted tool outputs (by call_id).
# ---------------------------------------------------------------------------

def _run_scripted_loop(
    llm_script: List[Dict[str, Any]],
    tool_outputs: Dict[str, Dict[str, Any]],
    *,
    vars_extra: Optional[Dict[str, Any]] = None,
    max_ticks: int = 80,
) -> Tuple[List[Dict[str, Any]], RunState]:
    captured: List[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        idx = min(len(captured) - 1, len(llm_script) - 1)
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
                ToolDefinition(name="camera_preview_photo", description="Save a live-view frame", parameters={}),
                ToolDefinition(name="list_files", description="List files", parameters={}),
            ]
        ),
        workflow_id="react_sight_lane",
        provider="stub",
        model="stub",
        allowed_tools=["camera_preview_photo", "list_files"],
    )
    run_vars: Dict[str, Any] = {
        "context": {"task": "Look at the scene.", "messages": []},
        "_runtime": {"inbox": []},
    }
    if vars_extra:
        for k, v in vars_extra.items():
            if isinstance(v, dict) and isinstance(run_vars.get(k), dict):
                run_vars[k].update(v)
            else:
                run_vars[k] = v
    run_id = runtime.start(workflow=workflow, vars=run_vars, actor_id=None, session_id="sess-sight")
    for _ in range(max_ticks):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, state


def _capture_call(call_id: str = "call_1") -> Dict[str, Any]:
    return {
        "content": "Taking a look.",
        "tool_calls": [{"name": "camera_preview_photo", "arguments": {}, "call_id": call_id}],
    }


# ---------------------------------------------------------------------------
# End-to-end fold behavior.
# ---------------------------------------------------------------------------

def test_tool_result_media_rides_next_reason_call_one_shot() -> None:
    refs = [{"$artifact": "art-photo-1", "filename": "preview.jpg"}]
    llm_script = [
        _capture_call("call_1"),
        {  # sees the image, keeps working with a plain tool
            "content": "I see a desk. Checking files.",
            "tool_calls": [{"name": "list_files", "arguments": {}, "call_id": "call_2"}],
        },
        {"content": "Done.", "tool_calls": []},
    ]
    tool_outputs = {
        "call_1": {"success": True, "output": {"path": "/tmp/preview.jpg", "media": refs}},
        "call_2": {"success": True, "output": "notes.txt"},
    }
    payloads, _state = _run_scripted_loop(llm_script, tool_outputs)

    assert len(payloads) == 3
    assert "media" not in payloads[0]
    assert payloads[1].get("media") == refs
    # One-shot: the call AFTER consumption carries no media again.
    assert "media" not in payloads[2]


def test_context_attachments_persist_while_captures_are_one_shot_and_deduped() -> None:
    staged = {"$artifact": "att-1", "filename": "brief.pdf"}
    captured_dup = {"$artifact": "att-1"}  # same artifact captured again
    captured_new = {"$artifact": "art-2"}
    llm_script = [
        _capture_call("call_1"),
        {
            "content": "Comparing.",
            "tool_calls": [{"name": "list_files", "arguments": {}, "call_id": "call_2"}],
        },
        {"content": "Done.", "tool_calls": []},
    ]
    tool_outputs = {
        "call_1": {"success": True, "output": {"media": [captured_dup, captured_new]}},
        "call_2": {"success": True, "output": "ok"},
    }
    payloads, _state = _run_scripted_loop(
        llm_script,
        tool_outputs,
        vars_extra={"context": {"attachments": [staged]}},
    )

    # Call 1: context attachment only.
    assert payloads[0].get("media") == [staged]
    # Call 2: staged copy wins the dedup (context first), new capture appended.
    assert payloads[1].get("media") == [staged, captured_new]
    # Call 3: capture consumed; context attachment still rides.
    assert payloads[2].get("media") == [staged]


def test_failed_or_undeclared_results_fold_nothing() -> None:
    llm_script = [
        {
            "content": "Trying two captures and a listing.",
            "tool_calls": [
                {"name": "camera_preview_photo", "arguments": {}, "call_id": "call_fail"},
                {"name": "camera_preview_photo", "arguments": {}, "call_id": "call_str"},
                {"name": "list_files", "arguments": {}, "call_id": "call_plain"},
            ],
        },
        {"content": "Done.", "tool_calls": []},
    ]
    tool_outputs = {
        # Failure declaring media: a failed capture's media claim is not evidence.
        "call_fail": {"success": False, "output": {"media": ["/tmp/ghost.jpg"]}, "error": "shutter jammed"},
        # String output: no declared field to read (never prose-sniffed).
        "call_str": {"success": True, "output": "saved /tmp/somewhere.jpg"},
        # Dict output without media key.
        "call_plain": {"success": True, "output": {"files": ["a.txt"]}},
    }
    payloads, _state = _run_scripted_loop(llm_script, tool_outputs)
    assert len(payloads) == 2
    assert "media" not in payloads[1]


def test_burst_cap_keeps_most_recent_and_is_not_silent() -> None:
    paths = [f"/tmp/burst_{i}.jpg" for i in range(9)]
    llm_script = [
        _capture_call("call_1"),
        {"content": "Done.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": True, "output": {"media": list(paths)}}}
    payloads, _state = _run_scripted_loop(llm_script, tool_outputs)

    folded = payloads[1].get("media")
    assert isinstance(folded, list)
    # Cap = 6, most-recent wins.
    assert folded == paths[-6:]


def test_multi_batch_accumulation_across_observe_passes() -> None:
    """Two capture results in ONE model turn both reach the next call (the
    observe→act queue-split loop accumulates, never overwrites)."""
    llm_script = [
        {
            "content": "Two angles.",
            "tool_calls": [
                {"name": "camera_preview_photo", "arguments": {}, "call_id": "call_a"},
                {"name": "camera_preview_photo", "arguments": {}, "call_id": "call_b"},
            ],
        },
        {"content": "Done.", "tool_calls": []},
    ]
    tool_outputs = {
        "call_a": {"success": True, "output": {"media": [{"$artifact": "shot-a"}]}},
        "call_b": {"success": True, "output": {"media": [{"$artifact": "shot-b"}]}},
    }
    payloads, _state = _run_scripted_loop(llm_script, tool_outputs)
    assert payloads[1].get("media") == [{"$artifact": "shot-a"}, {"$artifact": "shot-b"}]


def test_conclude_call_consumes_pending_media_at_budget_wall() -> None:
    """Budget exhausted right after a capture: reason never runs again — the
    conclusion call is the last model call and must carry the refs."""
    refs = [{"$artifact": "art-last-look"}]
    llm_script = [
        _capture_call("call_1"),
        {"content": "Best-effort conclusion: the scene shows a desk.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": True, "output": {"media": refs}}}
    payloads, _state = _run_scripted_loop(
        llm_script,
        tool_outputs,
        vars_extra={"_limits": {"max_iterations": 1}},
    )
    assert len(payloads) == 2
    assert payloads[1].get("media") == refs


def test_reset_react_turn_clears_pending_media() -> None:
    run_vars: Dict[str, Any] = {
        "scratchpad": {"iteration": 3, "cycles": []},
        "_temp": {"pending_media": [{"$artifact": "stale"}]},
        "_limits": {},
    }
    reset_react_turn(run_vars)
    assert "pending_media" not in run_vars["_temp"]


# ---------------------------------------------------------------------------
# Helper contracts (media.py).
# ---------------------------------------------------------------------------

def test_extract_media_from_tool_result_contract() -> None:
    ok = {"success": True, "output": {"media": ["/tmp/a.jpg", {"$artifact": "x"}, {"junk": 1}, "", 42]}}
    assert extract_media_from_tool_result(ok) == ["/tmp/a.jpg", {"$artifact": "x"}]
    assert extract_media_from_tool_result({"success": False, "output": {"media": ["/tmp/a.jpg"]}}) is None
    assert extract_media_from_tool_result({"success": True, "output": "prose /tmp/a.jpg"}) is None
    assert extract_media_from_tool_result({"success": True, "output": {}}) is None
    assert extract_media_from_tool_result({"success": True, "output": {"media": []}}) is None
    assert extract_media_from_tool_result("not-a-dict") is None  # type: ignore[arg-type]


def test_normalize_media_items_shapes() -> None:
    assert normalize_media_items(["  /a.png ", {"artifact_id": "b"}, {"nope": True}, None, 7]) == [
        "/a.png",
        {"artifact_id": "b"},
    ]
    assert normalize_media_items([]) is None
    assert normalize_media_items("not-a-list") is None
    assert normalize_media_items(("/t.png",)) == ["/t.png"]


def test_media_item_key_and_merge_dedup_order() -> None:
    a = {"$artifact": "same", "filename": "one.jpg"}
    b = {"artifact_id": "same"}
    c = "/tmp/c.jpg"
    assert media_item_key(a) == media_item_key(b) == "artifact:same"
    assert media_item_key(c) == "path:/tmp/c.jpg"
    assert media_item_key({"junk": 1}) is None
    # First occurrence wins; unkeyable items pass through.
    merged = merge_media_lists([a, c], [b, {"junk": 1}, c])
    assert merged == [a, c, {"junk": 1}]
    assert merge_media_lists(None, None) is None
