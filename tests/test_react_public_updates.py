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


def _tool_payloads(ledger):
    return [r["effect"]["payload"] for r in ledger
            if r.get("status") == "started" and (r.get("effect") or {}).get("type") == "tool_calls"
            and isinstance((r.get("effect") or {}).get("payload", {}).get("tool_calls"), list)]


def test_public_action_uses_only_same_proposal_update_and_host_stamped_call_ids():
    reason = "I want to check the museum's hours so we can choose a suitable time."
    no_id = _read(); no_id.pop("call_id")
    later = _read("later"); later["arguments"] = {"file_path": "parking.txt"}
    state, requests, _, ledger, _ = _harness([
        {"content": _block(reason), "tool_calls": [no_id]},
        {"content": "Checking parking.", "tool_calls": [later]},
        {"content": "Done.", "tool_calls": []},
    ])
    payloads = _tool_payloads(ledger)
    assert len(payloads) == 2 and len(requests) == 3
    action = payloads[0]["public_action"]
    assert action == {"text": reason, "update_id": state.run_id + ":public_update:1", "run_id": state.run_id,
                      "iteration": 1, "call_ids": [payloads[0]["tool_calls"][0]["call_id"]]}
    assert action["call_ids"] == ["call_1"]
    assert "public_action" not in payloads[1]  # never reuse the earlier public brief
    assert "public_action_binding" not in state.vars["_temp"]


def test_public_action_follows_sequential_queue_splits_without_expanding_a_batch():
    reason = "I want to check the outing details so the plan fits your needs."
    _, requests, _, ledger, asks = _harness([
        {"content": _block(reason), "tool_calls": [_read("before"), {
            "name": "ask_user", "call_id": "ask", "arguments": {"question": "Keep the museum?"}}, _read("after")]},
        {"content": "Done.", "tool_calls": []},
    ])
    payloads = _tool_payloads(ledger)
    assert len(requests) == 2 and len(asks) == 1 and len(payloads) == 2
    assert [p["public_action"]["call_ids"] for p in payloads] == [["before"], ["after"]]
    assert len({p["public_action"]["update_id"] for p in payloads}) == 1
    assert all(p["public_action"]["text"] == reason for p in payloads)


@pytest.mark.parametrize("enabled,response", [
    (False, {"content": _block("Public words in an opted-out run."), "tool_calls": [_read()]}),
    (True, {"content": "", "reasoning": _block("PRIVATE"), "tool_calls": [_read()]}),
    (True, {"content": _block("Unverified", ["not-observed"]), "tool_calls": [_read()]}),
])
def test_unpublished_or_disabled_content_never_becomes_approval_context(enabled, response):
    _, _, _, ledger, _ = _harness([response, {"content": "Done.", "tool_calls": []}], enabled=enabled)
    assert all("public_action" not in p for p in _tool_payloads(ledger))


def test_public_action_rejects_replaced_arguments_duplicate_ids_and_stale_cycle():
    from copy import deepcopy
    from abstractagent.adapters.react_runtime import _public_action_call_signature, _public_action_for_batch
    from abstractruntime.core.models import RunState
    call = {"name": "read_file", "arguments": {"items": list(range(40))}, "call_id": "c1"}
    update = {"id": "r:public_update:1", "run_id": "r", "iteration": 1, "text": "Check the full list.", "source_call_ids": []}
    run = RunState(run_id="r", workflow_id="w", status=RunStatus.RUNNING, current_node="act", vars={
        "scratchpad": {"cycles": [{"i": 1, "public_update": update}]},
        "_temp": {"conversation_enabled": True, "public_action_binding": {
            "update": deepcopy(update), "cycle": 1, "calls": {"c1": _public_action_call_signature(call)}}}})
    assert _public_action_for_batch(run, [call])["call_ids"] == ["c1"]
    replacement = deepcopy(call); replacement["arguments"]["items"][-1] = 999
    assert _public_action_for_batch(run, [replacement]) is None  # includes arguments after item 25
    assert _public_action_for_batch(run, [call, call]) is None
    run.vars["scratchpad"]["cycles"].append({"i": 2})
    assert _public_action_for_batch(run, [call]) is None
    reset_react_turn(run.vars)
    assert "public_action_binding" not in run.vars["_temp"]


def test_review_replacement_cannot_inherit_an_actors_earlier_public_reason():
    from abstractruntime.core.models import RunState
    class Ctx:
        def now_iso(self): return "2026-11-01T12:00:00Z"
    workflow = create_react_workflow(logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]))
    run = RunState(run_id="r", workflow_id="w", status=RunStatus.RUNNING, current_node="review_parse", vars={
        "context": {"task": "Plan an outing", "messages": []}, "scratchpad": {"iteration": 1},
        "_runtime": {"allowed_tools": ["read_file"], "conversation": {"enabled": True}},
        "_temp": {"conversation_enabled": True, "public_action_binding": {"text": "STALE"},
                  "review_llm_response": {"data": {"complete": False, "missing": ["hours"], "next_prompt": "",
                      "next_tool_calls": [{"name": "read_file", "arguments": {"file_path": "hours.txt"}}]}}}})
    plan = workflow.get_node("review_parse")(run, Ctx())
    assert plan.next_node == "act" and "public_action_binding" not in run.vars["_temp"]
    action = workflow.get_node("act")(run, Ctx())
    assert "public_action" not in action.effect.payload


def test_real_runtime_approval_wait_uses_authored_reason_without_extra_inference():
    from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
    from abstractruntime.integrations.abstractcore.tool_executor import ApprovalToolExecutor, MappingToolExecutor, ToolApprovalPolicy
    reason = "I want to check the museum's opening hours so we can choose a convenient time."
    script = iter([{"content": _block(reason), "reasoning": "PRIVATE", "tool_calls": [_read("hours")]},
                   {"content": "I left the plan unchanged.", "tool_calls": []}])
    requests, executed = [], []
    def llm(run, effect, next_node):
        requests.append(effect.payload); return EffectOutcome.completed(next(script))
    tools = ApprovalToolExecutor(delegate=MappingToolExecutor({"read_file": lambda **args: executed.append(args)}),
                                 policy=ToolApprovalPolicy(require_approval_tools={"read_file"}))
    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), effect_handlers={
        EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: make_tool_calls_handler(tools=tools)})
    rt.set_tool_executor_for_resume(tools)
    wf = create_react_workflow(logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]))
    rid = rt.start(workflow=wf, vars={"context": {"task": "Plan an outing.", "messages": []},
                                   "_runtime": {"conversation": {"enabled": True}, "review_mode": False}})
    state = rt.tick(workflow=wf, run_id=rid, max_steps=30)
    assert state.status == RunStatus.WAITING and len(requests) == 1 and not executed
    assert state.waiting.details["public_action"] == {
        "text": reason, "update_id": rid + ":public_update:1", "run_id": rid, "iteration": 1, "call_ids": ["hours"]}
    assert "PRIVATE" not in json.dumps(state.waiting.details)
    done = rt.resume(workflow=wf, run_id=rid, wait_key=state.waiting.wait_key, payload={"approved": False}, max_steps=30)
    assert done.status == RunStatus.COMPLETED and len(requests) == 2 and not executed
