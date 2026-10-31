"""Public communication is opt-in, actor-authored and durable without extra inference."""

from __future__ import annotations

import json
from typing import Any

import pytest

from abstractagent.adapters.public_updates import (
    MAX_TEXT_CHARS, PUBLIC_UPDATE_INSTRUCTION, conversation_enabled, parse_public_update,
)
from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
from abstractagent.logic.builtins import ASK_USER_TOOL
from abstractagent.logic.react import ReActLogic
from abstractagent.agents.react import ReactAgent
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import EffectType, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


pytestmark = pytest.mark.basic


def _block(text: str, sources: list[str] | None = None) -> str:
    return '<public_update>' + json.dumps({"text": text, "source_call_ids": sources or []}) + '</public_update>'


def _harness(script: list[dict[str, Any]], *, enabled: Any = True, question_result=None):
    captured, emitted, asks = [], [], []
    calls = iter(script)

    def llm(run, effect, next_node):
        captured.append({"messages": json.loads(json.dumps(effect.payload.get("messages", []))),
                         "system_prompt": effect.payload.get("system_prompt"),
                         "tools": effect.payload.get("tools")})
        return EffectOutcome.completed(next(calls))

    def execute(run, effect, next_node):
        return EffectOutcome.completed({"mode": "executed", "results": [
            {"name": c["name"], "call_id": c["call_id"], "success": True,
             "output": "The museum's site lists parking nearby."}
            for c in effect.payload["tool_calls"]
        ]})

    def ask(run, effect, next_node):
        asks.append(effect.payload)
        return EffectOutcome.completed(question_result or {"response": "Keep the museum."})

    ledger = InMemoryLedgerStore()
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=ledger, effect_handlers={
        EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: execute, EffectType.ASK_USER: ask,
    })
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={}), ASK_USER_TOOL]),
        on_step=lambda name, data: emitted.append((name, data)),
    )
    run_id = runtime.start(workflow=workflow, session_id="conversation-session", vars={
        "context": {"task": "Plan an outing.", "messages": []},
        "_runtime": {"conversation": {"enabled": enabled}, "review_mode": False},
    })
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            break
    assert state.status == RunStatus.COMPLETED, state.error
    return state, captured, emitted, ledger.list(run_id), asks


def _read(call_id="read-1"):
    return {"name": "read_file", "arguments": {"file_path": "hours.txt"}, "call_id": call_id}


@pytest.mark.parametrize("enabled", [False, None, "true", 1])
def test_disabled_keeps_original_prompt_tools_and_no_public_event(enabled):
    state, payloads, events, ledger, _ = _harness([
        {"content": "Reading the hours.", "tool_calls": [_read()]},
        {"content": "The museum has parking.", "tool_calls": []},
    ], enabled=enabled)
    assert len(payloads) == 2
    baseline = ReActLogic(tools=[]).build_request(task="", messages=[]).system_prompt
    assert all(p["system_prompt"] == baseline for p in payloads)
    ask_spec = next(t for t in payloads[0]["tools"] if t["name"] == "ask_user")
    assert ask_spec == ASK_USER_TOOL.to_dict()
    assert not any(name.startswith("public_update") for name, _ in events)
    assert not any((r.get("effect") or {}).get("type") == "emit_event" and r["effect"]["payload"].get("name") == "public_update" for r in ledger)


def test_joined_observations_publish_in_normal_continuation_and_durable_event():
    note = "The museum lists parking nearby; I still need to check opening hours."
    authored = _block(note, ["read-1", "read-2"]) + "\nChecking opening hours."
    state, payloads, events, ledger, _ = _harness([
        {"content": "", "reasoning": "Private deliberation.", "tool_calls": [_read(), _read("read-2")]},
        {"content": authored, "reasoning": "Do not show this private text.", "tool_calls": [_read("read-3")]},
        {"content": "The museum is an option.", "tool_calls": []},
    ])
    assert len(payloads) == 3  # no per-tool or public-update model call
    assert all(p["system_prompt"] == payloads[0]["system_prompt"] for p in payloads)
    assert PUBLIC_UPDATE_INSTRUCTION in payloads[0]["system_prompt"]
    updates = [data for name, data in events if name == "public_update"]
    assert len(updates) == 1
    update = updates[0]
    assert update["text"] == note
    assert update["source_call_ids"] == ["read-1", "read-2"]
    assert update["run_id"] == state.run_id and update["session_id"] == "conversation-session"
    assert update["id"] == state.run_id + ":public_update:2"
    assert "private" not in json.dumps(update).lower()
    assert state.vars["scratchpad"]["cycles"][1]["public_update"] == update
    assert any(m.get("role") == "assistant" and m.get("content") == authored for m in payloads[2]["messages"])
    durable = [r for r in ledger if r["status"] == "completed" and (r.get("effect") or {}).get("type") == "emit_event"
               and r["effect"]["payload"].get("name") == "public_update"]
    assert len(durable) == 1 and durable[0]["effect"]["payload"]["payload"] == update
    # Every prior request's complete stable messages remain an exact prefix.
    for first, second in zip(payloads, payloads[1:]):
        assert second["messages"][:len(first["messages"])] == first["messages"]


def test_reasoning_fallback_never_publishes_and_final_needs_no_update():
    _, payloads, events, _, _ = _harness([
        {"content": "", "reasoning": _block("Private reasoning must not become a brief."), "tool_calls": [_read()]},
        {"content": "Here is the answer.", "tool_calls": []},
    ])
    assert len(payloads) == 2
    assert not any(name == "public_update" for name, _ in events)


@pytest.mark.parametrize("content,code", [
    ('<public_update>{oops}</public_update>', "invalid_json"),
    (_block("x" * (MAX_TEXT_CHARS + 1)), "invalid_text"),
    (_block("I checked it.", ["proposed-not-observed"]), "unknown_or_invalid_sources"),
    ('<public_update>{"text":"Unfinished"}', "unclosed_block"),
    (_block("One") + _block("Two"), "multiple_blocks"),
])
def test_malformed_update_is_rejected_without_affecting_valid_tool_calls(content, code):
    state, payloads, events, ledger, _ = _harness([
        {"content": content, "tool_calls": [_read()]},
        {"content": "Finished.", "tool_calls": []},
    ])
    assert len(payloads) == 2
    assert not any(name == "public_update" for name, _ in events)
    assert [d["code"] for n, d in events if n == "public_update_rejected"] == [code]
    assert state.vars["scratchpad"]["cycles"][0]["public_update_rejected"]["code"] == code
    assert any((r.get("effect") or {}).get("payload", {}).get("name") == "public_update_rejected"
               and r["status"] == "completed" for r in ledger)


@pytest.mark.parametrize("response", [
    {"content": _block("Cut off."), "finish_reason": "length"},
    {"content": _block("Cancelled."), "cancelled": True},
    {"content": _block("Cancelled."), "finish_reason": "cancelled"},
])
def test_incomplete_or_cancelled_content_never_publishes(response):
    _, update, rejection = parse_public_update(response, known_call_ids=set())
    assert update is None and rejection == "incomplete_response"


@pytest.mark.parametrize("content", [
    'Example: ' + _block("Example only."),
    '```xml\n' + _block("Example only.") + '\n```',
    '> ' + _block("Quoted only."),
])
def test_marker_examples_are_not_interpreted_or_stripped(content):
    assert parse_public_update({"content": content}, known_call_ids=set()) == (None, None, None)


def test_config_is_frozen_per_turn_and_reset_for_entity_composition():
    runtime_ns = {"conversation": {"enabled": True}}
    temp = {}
    assert conversation_enabled(runtime_ns, temp)
    runtime_ns["conversation"]["enabled"] = False
    assert conversation_enabled(runtime_ns, temp)
    values = {"_runtime": runtime_ns, "_temp": temp}
    reset_react_turn(values)
    assert not conversation_enabled(runtime_ns, temp)


def test_optional_clarification_forwards_policy_and_timeout_is_host_input():
    policy = {"importance": "clarification", "timeout_s": 180,
              "fallback": "Keep both options and compare parking."}
    result = {"timed_out": True, "fallback": policy["fallback"],
              "question": {"id": "question-1", "outcome": "expired"}}
    state, payloads, events, _, asks = _harness([
        {"content": "", "tool_calls": [{"name": "ask_user", "call_id": "ask-1", "arguments": {
            "question": "Does Jeanne have a car?", **policy,
        }}]},
        {"content": "I kept both options.", "tool_calls": []},
    ], question_result=result)
    assert asks[0]["question"] == policy
    spec = next(t for t in payloads[0]["tools"] if t["name"] == "ask_user")
    assert spec["required_args"] == ["question"]
    assert {"importance", "timeout_s", "fallback"} <= set(spec["parameters"])
    messages = state.vars["context"]["messages"]
    expired = [m for m in messages if (m.get("metadata") or {}).get("kind") == "question_expired"]
    assert len(expired) == 1 and expired[0]["role"] == "system"
    assert "did not answer or approve" in expired[0]["content"]
    assert policy["fallback"] in expired[0]["content"]
    assert not any(m["role"] == "user" and policy["fallback"] in m["content"] for m in messages)
    assert len([n for n, _ in events if n == "question_expired"]) == 1


def test_default_ask_user_keeps_indefinite_payload_and_answer():
    state, _, _, _, asks = _harness([
        {"content": "", "tool_calls": [{"name": "ask_user", "call_id": "ask-1",
                                         "arguments": {"question": "Which museum?"}}]},
        {"content": "I'll keep that museum.", "tool_calls": []},
    ], enabled=False)
    assert asks == [{"prompt": "Which museum?", "choices": None, "allow_free_text": True}]
    assert any(m["role"] == "user" and m["content"].endswith("[User response]: Keep the museum.")
               for m in state.vars["context"]["messages"])


def test_python_facade_exposes_explicit_run_opt_in():
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    agent = ReactAgent(runtime=runtime, tools=[], review_mode=False)
    config = {"enabled": True}
    run_id = agent.start("Plan an outing.", conversation=config)
    config["enabled"] = False
    assert runtime.get_state(run_id).vars["_runtime"]["conversation"] == {"enabled": True}
    with pytest.raises(ValueError, match="conversation must be an object"):
        agent.start("Plan an outing.", conversation="true")
