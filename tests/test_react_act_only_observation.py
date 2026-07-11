"""Act-only tool observations render as `$act_only` references (frozen visit seam spec).

Contract source: a2a/threads/0013-visit-seam-spec (spec v2 §2 + the agent-seat pins,
2026-07-10). Act-only tools (e.g. entity diary reads) must never land tool-surfaced
CONTENT in any at-rest surface outside its one home: the durable transcript carries an
ACT-FRAME REFERENCE as the tool message's content — one exact JSON object with a lone
`$act_only` top-level key — which the runtime LLM_CALL handler dereferences at the
provider boundary into a wire copy. The adapter is NOT the privacy mechanism (the
effect handler is); these tests pin the adapter's canonical-shape half:

- handler-authored refs are honored (handler authority) when dereferenceable;
- tools declared `act_only` (core's first-class ToolDefinition field; getattr-based,
  fail-closed by absence) render frames canonically, and raw output from a
  misbehaving handler is suppressed LOUDLY before it becomes permanent;
- message identity (role/tool_call_id/position) and byte-stability survive the
  payload boundary — refs flow into LLM_CALL payloads byte-identical;
- the emit/on_step observability lane carries only the ref;
- normal tools are untouched (regression);
- WEDGE GUARD (cross-package, against runtime's shipped send-time dereference in
  abstractruntime/identity/act_only.py): the lone-key REF shape is reserved for
  frames that reference book content (entry_id present). Runtime's LLM wrapper
  loudly FAILS a call on any unresolvable ref, and refs are durable — a
  non-dereferenceable ref (failure/suppression record) would wedge every later
  LLM call in the run. Those records render as labeled NON-REF text instead:
  still no words at rest, inert to the dereference pass. Interop is pinned by
  running runtime's own parser over both shapes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import (
    _ACT_ONLY_KEY,
    _act_only_ref_content,
    create_react_workflow,
)
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


_DIARY_FRAME = {
    "tool": "diary_read",
    "entry_id": "diary_ab12",
    "reason": "reconnect with the garden entry",
    "gist": "a quiet line about the garden",
}

_SENTINEL = "PRIVATE-WORDS-SENTINEL-XYZZY"


def _make_tools() -> List[ToolDefinition]:
    # Core's first-class `act_only` field (shipped 2026-07-10, 0013 thread) — declared
    # constructor-native. The adapter reads it getattr-based (fail-closed by absence),
    # so the check also holds against older cores and dict-shaped tool specs.
    diary_read = ToolDefinition(name="diary_read", description="Read a diary entry", parameters={}, act_only=True)
    diary_list = ToolDefinition(name="diary_list", description="List diary entries", parameters={})
    list_files = ToolDefinition(name="list_files", description="List files", parameters={})
    return [diary_read, diary_list, list_files]


def _run_scripted_loop(
    llm_script: List[Dict[str, Any]],
    tool_outputs: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]], RunState]:
    """Drive the real runtime through a scripted loop.

    llm_script: one response dict per LLM iteration.
    tool_outputs: call_id -> {"success": bool, "output": Any, "error": Any}.
    Returns (captured LLM payloads, emitted (step, data) events, final state).
    """
    captured: List[Dict[str, Any]] = []
    events: List[Tuple[str, Dict[str, Any]]] = []

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
        logic=ReActLogic(tools=_make_tools()),
        on_step=lambda step, data: events.append((step, json.loads(json.dumps(data, default=str)))),
        workflow_id="react_act_only",
        provider="stub",
        model="stub",
        allowed_tools=["diary_read", "diary_list", "list_files"],
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "Reconnect with the diary", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-act-only",
    )
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, events, state


def _durable_tool_messages(state: RunState) -> List[Dict[str, Any]]:
    context = state.vars.get("context") if isinstance(state.vars, dict) else {}
    msgs = context.get("messages") if isinstance(context, dict) else []
    return [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]


def _adapter_owned_vars(state: RunState) -> Dict[str, Any]:
    """Run vars minus `_runtime.node_traces` — the runtime KERNEL's effect-result trace.

    Honest boundary (found while building this, reported to the seam-spec thread): the
    runtime copies every raw effect result into `_runtime.node_traces.<node>.steps[]`
    BEFORE observe_node runs. The adapter cannot (and must not — that would be post-hoc
    redaction of another component's record) scrub it. For act-only tools the words
    never reach the result channel BY HANDLER CONTRACT — that is the real mechanism;
    node_traces then holds act-frames only. These tests pin the surfaces the ADAPTER
    owns: durable transcript, scratchpad, LLM payloads, emit lane.
    """
    vars_copy = json.loads(json.dumps(state.vars))
    runtime_ns = vars_copy.get("_runtime")
    if isinstance(runtime_ns, dict):
        runtime_ns.pop("node_traces", None)
    return vars_copy


def test_handler_authored_ref_renders_canonically_and_rides_payload_byte_identical() -> None:
    llm_script = [
        {
            "content": "Reaching for the entry.",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_ab12"}, "call_id": "call_1"}],
        },
        {"content": "Done reconnecting.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": True, "output": {_ACT_ONLY_KEY: dict(_DIARY_FRAME)}, "error": None}}
    payloads, events, state = _run_scripted_loop(llm_script, tool_outputs)

    # Durable transcript: content is one exact JSON object with the lone $act_only key.
    tool_msgs = _durable_tool_messages(state)
    assert len(tool_msgs) == 1
    content = str(tool_msgs[0].get("content") or "")
    parsed = json.loads(content)
    assert set(parsed.keys()) == {_ACT_ONLY_KEY}
    assert parsed[_ACT_ONLY_KEY]["entry_id"] == "diary_ab12"
    assert parsed[_ACT_ONLY_KEY]["gist"] == _DIARY_FRAME["gist"]
    # Canonical serialization (deterministic bytes).
    assert content == _act_only_ref_content(parsed[_ACT_ONLY_KEY])
    assert tool_msgs[0].get("metadata", {}).get("act_only") is True

    # Payload boundary: the ref reaches the next LLM_CALL byte-identical, with the
    # tool message's identity (role + tool_call_id) intact.
    assert len(payloads) == 2
    payload_tool_msgs = [m for m in payloads[1].get("messages", []) if m.get("role") == "tool"]
    assert len(payload_tool_msgs) == 1
    assert payload_tool_msgs[0].get("content") == content
    assert payload_tool_msgs[0].get("tool_call_id") == "call_1"

    # Emit lane carries only the ref.
    observe_events = [d for (s, d) in events if s == "observe"]
    assert len(observe_events) == 1
    assert observe_events[0]["result"] == content

    # Scratchpad cycle observation stores the wrapped frame, not tool-surfaced content.
    scratchpad = state.vars.get("scratchpad") or {}
    cycles = scratchpad.get("cycles") or []
    obs = cycles[0]["observations"][0]
    assert set(obs["output"].keys()) == {_ACT_ONLY_KEY}
    assert obs["rendered"] == content


def test_declared_act_only_tool_suppresses_raw_handler_output_loudly() -> None:
    llm_script = [
        {
            "content": "Reading.",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_ab12"}, "call_id": "call_1"}],
        },
        {"content": "Finished.", "tool_calls": []},
    ]
    # Misbehaving handler: returns raw words for a declared act-only tool.
    tool_outputs = {"call_1": {"success": True, "output": _SENTINEL, "error": None}}
    payloads, events, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    assert len(tool_msgs) == 1
    content = str(tool_msgs[0].get("content"))
    # Suppression records are act-only but reference nothing — they render as labeled
    # NON-REF text (wedge guard: a durable unresolvable ref would fail every later
    # LLM call through runtime's send-time dereference).
    assert content.startswith("[diary_read]: act-only record")
    assert "#FALLBACK" in content

    # The raw words never became permanent on any ADAPTER-OWNED surface: durable
    # transcript + scratchpad (adapter-owned vars), every LLM payload, every emitted
    # event. (`_runtime.node_traces` is the kernel's raw effect-result trace, written
    # before observe_node runs — see _adapter_owned_vars; keeping words out of the
    # result channel entirely is the handler's contract, and the runtime seat owns
    # that surface.)
    assert _SENTINEL not in json.dumps(_adapter_owned_vars(state))
    assert _SENTINEL not in json.dumps(payloads)
    assert _SENTINEL not in json.dumps(events)


def test_failed_act_only_call_renders_error_channel_only() -> None:
    llm_script = [
        {
            "content": "Reading.",
            "tool_calls": [{"name": "diary_read", "arguments": {"entry_id": "diary_zz99"}, "call_id": "call_1"}],
        },
        {"content": "Understood, the door refused.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": False, "output": _SENTINEL, "error": "diary door refused: asleep"}}
    payloads, events, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    content = str(tool_msgs[0].get("content"))
    # A failure references no entry: labeled non-ref record text (wedge guard), with
    # diagnostics from the ERROR channel only.
    assert content.startswith("[diary_read]: act-only record")
    assert "diary door refused: asleep" in content
    # Raw output is never rendered on any adapter-owned surface (node_traces
    # excluded — kernel-owned, see helper).
    assert _SENTINEL not in json.dumps(_adapter_owned_vars(state))
    assert _SENTINEL not in json.dumps(payloads)
    assert _SENTINEL not in json.dumps(events)


def test_handler_authored_ref_from_undeclared_tool_is_honored() -> None:
    llm_script = [
        {
            "content": "Listing.",
            "tool_calls": [{"name": "diary_list", "arguments": {}, "call_id": "call_1"}],
        },
        {"content": "Listed.", "tool_calls": []},
    ]
    # Handler marks the result act-only AND names an entry: rendered as a REF even
    # though the local `diary_list` definition carries no act_only declaration
    # (the effect handler is the enforcement authority).
    frame = {"tool": "diary_read", "entry_id": "diary_ab12", "reason": "survey follow-up"}
    tool_outputs = {"call_1": {"success": True, "output": {_ACT_ONLY_KEY: dict(frame)}, "error": None}}
    _, _, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    parsed = json.loads(str(tool_msgs[0].get("content")))
    assert set(parsed.keys()) == {_ACT_ONLY_KEY}
    assert parsed[_ACT_ONLY_KEY]["entry_id"] == "diary_ab12"
    assert tool_msgs[0].get("metadata", {}).get("act_only") is True


def test_handler_authored_frame_without_entry_id_renders_as_record_text() -> None:
    llm_script = [
        {
            "content": "Listing.",
            "tool_calls": [{"name": "diary_list", "arguments": {}, "call_id": "call_1"}],
        },
        {"content": "Listed.", "tool_calls": []},
    ]
    # Act-only honored (no raw output at rest) but NOT dereferenceable (no entry_id):
    # labeled non-ref record text — inert to runtime's send-time dereference pass.
    frame = {"tool": "diary_list", "count": 3, "reason": "survey"}
    tool_outputs = {"call_1": {"success": True, "output": {_ACT_ONLY_KEY: dict(frame)}, "error": None}}
    _, _, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    content = str(tool_msgs[0].get("content"))
    assert content.startswith("[diary_list]: act-only record")
    assert '"count": 3' in content
    assert tool_msgs[0].get("metadata", {}).get("act_only") is True


def test_interop_with_runtime_shipped_dereference_parser() -> None:
    """Cross-package pin: my rendered bytes against runtime's SHIPPED parser.

    abstractruntime/identity/act_only.py (committed bce7d90) is the send-time
    dereference this rendering feeds. Two-way contract: my REF content parses as a
    ref and substitutes in place preserving message identity; my RECORD content
    (suppression/failure/no-entry frames) parses as NOT-a-ref and passes through
    untouched — so a record can never wedge the dereference pass.
    """
    act_only_rt = pytest.importorskip("abstractruntime.identity.act_only")

    ref_content = _act_only_ref_content({"tool": "diary_read", "entry_id": "diary_ab12", "gist": "one line"})
    record_content = "[diary_read]: act-only record (no content at rest): {\"error\": \"x\"}"

    # Ref: parses; dereference substitutes content only, identity untouched.
    assert act_only_rt.parse_act_only_ref(ref_content) == {
        "tool": "diary_read",
        "entry_id": "diary_ab12",
        "gist": "one line",
    }
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": ref_content, "tool_call_id": "call_1"},
        {"role": "tool", "content": record_content, "tool_call_id": "call_2"},
    ]
    wire, resolved = act_only_rt.dereference_act_only_messages(
        messages, read_entry=lambda ref: f"WORDS({ref['entry_id']})"
    )
    assert resolved == 1
    assert wire[1]["content"] == "WORDS(diary_ab12)"
    assert wire[1]["tool_call_id"] == "call_1"  # identity untouched
    assert wire[2]["content"] == record_content  # record inert, passes through
    assert messages[1]["content"] == ref_content  # original never mutated

    # Record: not a ref for runtime's parser.
    assert act_only_rt.parse_act_only_ref(record_content) is None


def test_normal_tools_render_exactly_as_before() -> None:
    llm_script = [
        {
            "content": "Checking files.",
            "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
        },
        {"content": "All done.", "tool_calls": []},
    ]
    tool_outputs = {"call_1": {"success": True, "output": "file_a.txt\nfile_b.txt", "error": None}}
    payloads, events, state = _run_scripted_loop(llm_script, tool_outputs)

    tool_msgs = _durable_tool_messages(state)
    assert len(tool_msgs) == 1
    content = str(tool_msgs[0].get("content"))
    assert content == "[list_files]: file_a.txt\nfile_b.txt"
    assert "act_only" not in (tool_msgs[0].get("metadata") or {})
    # And the observation output stays the raw handler output (host observability).
    scratchpad = state.vars.get("scratchpad") or {}
    obs = (scratchpad.get("cycles") or [])[0]["observations"][0]
    assert obs["output"] == "file_a.txt\nfile_b.txt"
