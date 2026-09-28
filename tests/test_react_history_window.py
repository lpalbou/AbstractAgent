"""ReAct sends THE runtime history window when the host asks for it
(`_runtime.history_window_tokens`; runtime 0.7.0's entity visit sets 50k).
Unset, the whole transcript rides as before. The durable transcript is never
windowed."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.turn_grounding import strip_turn_grounding

OLD = [m for i in range(6) for m in ({"role": "user", "content": f"old {i} " + "w " * 2000},
                                     {"role": "assistant", "content": f"reply {i}"})]


def _run(runtime_vars: Dict[str, Any]) -> tuple:
    captured: List[Dict[str, Any]] = []

    def llm(run: RunState, effect: Effect, dnn: Optional[str]) -> EffectOutcome:
        captured.append(json.loads(json.dumps(effect.payload)))
        return EffectOutcome.completed({"content": "done", "tool_calls": []})

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
                 effect_handlers={EffectType.LLM_CALL: llm})
    wf = create_react_workflow(logic=ReActLogic(tools=[ToolDefinition(name="t", description="t", parameters={})]),
                               workflow_id="react_window", provider="stub", model="stub", allowed_tools=["t"])
    messages = [dict(m) for m in OLD] + [{"role": "user", "content": "now"}]
    run_id = rt.start(workflow=wf, vars={"context": {"task": "now", "messages": messages}, "_runtime": dict(runtime_vars)})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=40)
    assert state.status == RunStatus.COMPLETED
    return captured[0]["messages"], state


def test_the_window_key_windows_the_request_and_records_it() -> None:
    wire, state = _run({"history_window_tokens": 3000})
    report = state.vars["_runtime"]["session_history"]
    assert report["max_tokens"] == 3000 and report["dropped_messages"] > 0
    assert len(wire) == report["replayed_messages"] < len(OLD) + 1
    assert strip_turn_grounding(wire[0]["content"]).startswith("[#TRUNCATION: ")
    assert strip_turn_grounding(wire[-1]["content"]) == "now"
    stored = state.vars["context"]["messages"]
    assert len(stored) >= len(OLD) + 1 and stored[0]["content"].startswith("old 0 ")  # durable transcript whole


def test_without_the_key_the_whole_transcript_rides() -> None:
    wire, state = _run({})
    assert "session_history" not in state.vars["_runtime"]
    assert len(wire) == len(OLD) + 1 and wire[0]["content"].startswith("old 0 ")


def test_the_turn_in_progress_is_kept_whole_from_the_hosts_turn_start() -> None:
    """`_runtime.history_window_turn_start` marks where the turn in progress
    began: from there on it is one turn, kept whole (and reported oversize),
    however many user-role messages the loop added inside it."""
    wire, state = _run({"history_window_tokens": 3000, "history_window_turn_start": 0})
    report = state.vars["_runtime"]["session_history"]
    assert report["dropped_messages"] == 0 and report["oversize_turn_kept"] is True
    assert len(wire) == len(OLD) + 1 and wire[0]["content"].startswith("old 0 ")
