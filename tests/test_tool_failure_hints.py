"""First-failure diagnosis: explain the error, do not just report the repeat.

Operator, 2026-08-21: "any hint/guidance must use the information available to
give more specific / actionable hints … you must explain what was the problem
so that the model can self correct … with the actual parameters used or
defective. Normally we should never hit the exact 3 repeats."

The live incident is the specification: twelve identical
`open_attachment(artifact_id="a1", handle="attachment")` calls, twelve
identical "no attachment matches" errors, and not one word back to the model
about WHICH argument was wrong or what to run instead.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.adapters.tool_failure_hints import (
    classify_tool_error,
    diagnose_tool_failure,
    render_hint_block,
)
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


def test_error_families_are_recognised_from_real_framework_wording() -> None:
    assert classify_tool_error("Error: no attachment matches handle 'x' in this session.") == "not_found"
    assert classify_tool_error("Error: File '/tmp/x.png' does not exist") == "not_found"
    assert classify_tool_error("ERROR: temporary backend timeout (503) - please retry.") == "transient"
    assert classify_tool_error("Error: permission denied: outside the workspace") == "denied"
    assert classify_tool_error("Error: end_line 0 is ambiguous; must be >= start_line") == "invalid_argument"
    assert classify_tool_error("Error: multiple attachments match 'notes.txt'.") == "ambiguous"
    # `edit_file`'s missing anchor is a not-found, and the not-found remedy
    # (re-read the target, then build the call from what it contains) is the
    # right one — so it deliberately does NOT fall to `conflict`.
    assert classify_tool_error("Error: pattern not found in file") == "not_found"
    assert classify_tool_error("Error: file already exists; refusing to overwrite") == "conflict"
    assert classify_tool_error("") == "unknown"
    assert classify_tool_error("something nobody has ever written") == "unknown"


def test_the_incident_call_is_diagnosed_by_its_actual_parameters() -> None:
    d = diagnose_tool_failure(
        name="open_attachment",
        arguments={"artifact_id": "a1", "handle": "attachment", "start_line": 1, "end_line": 220},
        error_text="Error: no attachment matches handle 'attachment' in this session.",
        tool_specs=[
            {"name": "read_file", "required_args": ["file_path"]},
            {"name": "search_files", "required_args": ["pattern"]},
        ],
        available_tools=["open_attachment", "read_file", "search_files"],
        transcript_text="the user asked which open-source models are best; web_search returned blog posts",
    )
    assert d is not None
    text = d["text"]
    # The failing call, verbatim, with the values that were actually sent.
    assert "open_attachment(artifact_id='a1'" in text
    # WHY — the fact nobody ever told the model.
    assert "artifact_id='a1'" in text and "appears NOWHERE earlier in this conversation" in text
    # WHAT TO DO — with real parameter names from the real toolset.
    assert "read_file(file_path=...)" in text
    assert "search_files(pattern=...)" in text


def test_a_value_the_conversation_actually_supplied_is_never_called_invented() -> None:
    """The check must not accuse the model of inventing an id it was given —
    a false accusation is worse than silence."""
    d = diagnose_tool_failure(
        name="fetch_record",
        arguments={"record_id": "REG-2026-0614"},
        error_text="Error: record 'REG-2026-0614' not found in the registry.",
        available_tools=["fetch_record"],
        transcript_text="search_records returned: REG-2026-0614 — June outage, 4h",
    )
    assert d is not None
    assert "appears NOWHERE" not in d["text"]
    assert "does not exist here" in d["text"]


def test_a_transient_failure_is_not_blamed_on_the_arguments() -> None:
    d = diagnose_tool_failure(
        name="spec_service",
        arguments={"part_number": "widget-9000"},
        error_text="ERROR: temporary backend timeout (503). The service is briefly unavailable - please retry.",
        available_tools=["spec_service"],
        transcript_text="widget-9000",
    )
    assert d is not None
    assert d["class"] == "transient"
    assert "SERVICE-side failure, not a problem with your arguments" in d["text"]
    assert "no wait in it" in d["text"]


def test_only_tools_this_run_actually_has_are_ever_suggested() -> None:
    """Suggesting a tool the model cannot call is worse than suggesting none."""
    d = diagnose_tool_failure(
        name="open_attachment",
        arguments={"artifact_id": "zz9"},
        error_text="Error: no attachment matches handle 'zz9' in this session.",
        available_tools=["open_attachment"],  # nothing else on the bench
        transcript_text="unrelated",
    )
    assert d is not None
    assert "Available here:" not in d["text"]
    assert "read_file" not in d["text"]


def test_nothing_to_say_returns_none_rather_than_padding() -> None:
    assert (
        diagnose_tool_failure(
            name="weird_tool",
            arguments={"x": 1},
            error_text="glorp",
            available_tools=["weird_tool"],
            transcript_text="",
        )
        is None
    )
    assert render_hint_block([]) == ""


def test_identical_failures_share_a_signature_and_different_ones_do_not() -> None:
    common = dict(
        name="fetch_record",
        error_text="Error: record 'x' not found in the registry.",
        available_tools=["fetch_record"],
        transcript_text="",
    )
    a = diagnose_tool_failure(arguments={"record_id": "x"}, **common)
    b = diagnose_tool_failure(arguments={"record_id": "x"}, **common)
    c = diagnose_tool_failure(arguments={"record_id": "y"}, **common)
    assert a and b and c
    assert a["signature"] == b["signature"]
    assert a["signature"] != c["signature"]


# ---------------------------------------------------------------------------
# Wiring: the diagnosis reaches the model on the FIRST failure
# ---------------------------------------------------------------------------

def _drive_failing_tool(llm_script: List[Dict[str, Any]], *, error: str) -> tuple:
    payloads: List[Dict[str, Any]] = []
    events: List[tuple] = []
    call_n = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payloads.append(json.loads(json.dumps(effect.payload, default=str)))
        idx = min(call_n["n"], len(llm_script) - 1)
        call_n["n"] += 1
        return EffectOutcome.completed(dict(llm_script[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {
                        "call_id": tc.get("call_id"),
                        "name": tc.get("name"),
                        "success": False,
                        "output": "",
                        "error": error,
                    }
                    for tc in (payload.get("tool_calls") or [])
                ],
            }
        )

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(
            tools=[
                ToolDefinition(name="fetch_record", description="Fetch by id", parameters={"record_id": {"type": "string"}}),
                ToolDefinition(name="search_files", description="Search", parameters={"pattern": {"type": "string"}}),
            ]
        ),
        on_step=lambda s, d: events.append((s, json.loads(json.dumps(d, default=str)))),
        workflow_id="wf-hint",
        provider="stub",
        model="stub",
        allowed_tools=["fetch_record", "search_files"],
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Find the June outage record", "messages": []},
            "_runtime": {"inbox": []},
            "_limits": {"max_iterations": 12},
        },
    )
    for _ in range(200):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    return events, payloads, rt.get_state(rid)


def test_the_model_is_told_why_after_ONE_failure_not_after_three() -> None:
    call = {"content": "Fetching.", "tool_calls": [{"name": "fetch_record", "arguments": {"record_id": "REG-JUNE"}, "call_id": "c1"}]}
    final = {"content": "I could not find it.", "tool_calls": []}
    events, payloads, state = _drive_failing_tool([call, final], error="Error: record 'REG-JUNE' not found in the registry.")

    hints = [d for (s, d) in events if s == "tool_failure_hint"]
    assert len(hints) == 1, "the diagnosis must fire on the FIRST failure"
    assert hints[0]["classes"] == ["not_found"]
    # No stuck-streak was ever needed.
    assert not [d for (s, d) in events if s == "stuck_streak"]
    # And it reached the model on the very next call.
    nxt = json.dumps(payloads[1])
    assert "[tool failure]" in nxt
    assert "fetch_record(record_id='REG-JUNE')" in nxt
    assert "search_files(pattern=...)" in nxt
    assert (state.output or {}).get("outcome") == "final_answer"


def test_a_repeat_of_an_already_diagnosed_failure_trips_one_repeat_earlier() -> None:
    """Failing repeats trip at 2, not 3: the model was already handed the
    explanation, so a second identical batch is ignoring it, not lacking it."""
    call = {"content": "Fetching.", "tool_calls": [{"name": "fetch_record", "arguments": {"record_id": "REG-JUNE"}, "call_id": "c1"}]}
    final = {"content": "Giving up.", "tool_calls": []}
    events, _payloads, state = _drive_failing_tool([call, call, final], error="Error: record 'REG-JUNE' not found in the registry.")

    streaks = [d for (s, d) in events if s == "stuck_streak"]
    assert streaks, "a second identical FAILING batch must trip the guard"
    assert streaks[0]["span"] == 2 and streaks[0]["failed"] is True
    assert streaks[0]["action"] == "nudged"
    assert (state.output or {}).get("outcome") == "final_answer"
