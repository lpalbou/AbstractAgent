"""A generation aborted mid-tool-call is a FAULT, never an assistant turn.

Operator evidence, 2026-08-02 (LM Studio 0.3.x, qwen/qwen3.6-35b-a3b, server
log `~/.lmstudio/server-logs/2026-08/2026-08-02.1.log`): when a tool call is
cut mid-generation the server drops it —

    [ERROR][qwen/qwen3.6-35b-a3b] Failed to generate a tool call
    (this tool call will be omitted from the response)

— and answers HTTP 200 with the tool call's PREFACE as content::

    "message": {"role":"assistant",
                "content":"Good, I have the context. Let me continue building…",
                "tool_calls":[]},
    "finish_reason": "stop",
    "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}

`finish_reason` lies ("stop"), the body carries no error field, and the loop
used to fall through to `_looks_like_deferred_action` — recovering by GUESSING
at the model's intent and logging a `parse_retry_plan_only`, so the operator saw
an ordinary cycle where a tool call had actually been lost.

The tell that does survive is the usage block: text cannot have been produced
by a completion that consumed zero prompt tokens and produced zero completion
tokens. These tests pin that detector, the named retry it now routes to, and —
per the evidence law — that ABSENT usage stays UNKNOWN rather than becoming a
fault verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import (
    _truncation_kind,
    _zero_usage_with_speech,
    create_react_workflow,
)
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


ABORTED = {
    "content": "Good, I have the context. Let me continue building the game files. "
    "I need to create the game loop and main entry point.\n\n",
    "tool_calls": [],
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}


def _base_vars(*, task: str) -> Dict[str, Any]:
    return {"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}}


@pytest.mark.basic
def test_zero_usage_with_speech_is_the_abort_marker() -> None:
    # The operator's exact wire shape.
    assert _zero_usage_with_speech(ABORTED) is True
    assert _truncation_kind(ABORTED, "stop", []) == "aborted_generation"


@pytest.mark.basic
def test_absent_usage_is_unknown_not_a_fault() -> None:
    """Evidence law: missing evidence is UNKNOWN, never a clean verdict either way."""
    for usage in (None, {}, {"total_tokens": None}):
        resp = dict(ABORTED, usage=usage)
        assert _zero_usage_with_speech(resp) is False, usage
        assert _truncation_kind(resp, "stop", []) == "", usage


@pytest.mark.basic
def test_real_usage_and_carried_tool_calls_are_never_lost_work() -> None:
    real = dict(ABORTED, usage={"prompt_tokens": 5462, "completion_tokens": 40, "total_tokens": 5502})
    assert _truncation_kind(real, "stop", []) == ""
    # A response that DID carry its tool calls made progress, whatever else it says.
    assert _truncation_kind(ABORTED, "stop", [object()]) == ""
    # The output-cap lane keeps its own name.
    assert _truncation_kind({"content": "x", "usage": {"total_tokens": 9}}, "length", []) == "output_cap"


@pytest.mark.basic
def test_provider_annotation_wins_over_finish_reason() -> None:
    """abstractcore annotates `metadata.truncation_kind`; the loop trusts it first."""
    annotated = {
        "content": "Let me continue.",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {"total_tokens": 120},  # non-zero: only the annotation knows
        "metadata": {"output_truncated": True, "truncation_kind": "aborted_generation"},
    }
    assert _truncation_kind(annotated, "stop", []) == "aborted_generation"


def _run_loop_with(second_response: Dict[str, Any]) -> Tuple[RunState, List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Drive a real ReAct run whose 2nd LLM call returns `second_response`."""
    llm_payloads: List[Dict[str, Any]] = []
    steps: List[Tuple[str, Dict[str, Any]]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(dict(payload))
        idx = len(llm_payloads)
        if idx == 1:
            return EffectOutcome.completed(
                {
                    "content": "Checking workspace.",
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}],
                    "finish_reason": "tool_calls",
                    "usage": {"total_tokens": 900},
                }
            )
        if idx == 2:
            return EffectOutcome.completed(dict(second_response))
        if idx == 3:
            return EffectOutcome.completed(
                {
                    "content": "Writing the file.",
                    "tool_calls": [
                        {"name": "execute_command", "arguments": {"command": "mkdir -p p"}, "call_id": "c2"}
                    ],
                    "finish_reason": "tool_calls",
                    "usage": {"total_tokens": 1200},
                }
            )
        return EffectOutcome.completed(
            {"content": "Done.", "tool_calls": [], "finish_reason": "stop", "usage": {"total_tokens": 1300}}
        )

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
            if isinstance(tc, dict)
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
                ToolDefinition(name="list_files", description="List", parameters={}),
                ToolDefinition(name="execute_command", description="Cmd", parameters={}),
            ]
        ),
        workflow_id="react_agent_aborted_generation",
        provider="stub",
        model="stub",
        allowed_tools=["list_files", "execute_command"],
        on_step=lambda step, data: steps.append((step, dict(data))),
    )
    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="Build a game"), actor_id=None, session_id=None)
    for _ in range(100):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    return runtime.get_state(run_id), steps, llm_payloads


@pytest.mark.basic
def test_aborted_generation_is_a_named_visible_retry_not_a_plan_only_guess() -> None:
    state, steps, llm_payloads = _run_loop_with(ABORTED)

    assert state.status == RunStatus.COMPLETED
    names = [s for s, _ in steps]

    # (1) DETECTION + HANDLING: the lost tool call routes to the truncation
    # retry, NOT to the intent-guessing followthrough heuristic.
    assert "parse_retry_truncated" in names
    assert "parse_retry_plan_only" not in names

    # (2) VISIBILITY: the record names the fault and counts it.
    payload = next(d for s, d in steps if s == "parse_retry_truncated")
    assert payload["kind"] == "aborted_generation"
    assert payload["finish_reason"] == "stop"
    assert payload["truncated_cycles"] == 1
    assert "Good, I have the context" in str(payload["content_preview"])

    # (3) The lost cycle is marked in the durable scratchpad, so "13 cycles"
    # can be read as productive-vs-recovery instead of an opaque number.
    scratchpad = state.vars["scratchpad"]
    assert scratchpad["truncated_cycles"] == 1
    lost = [c for c in scratchpad["cycles"] if c.get("lost_tool_call")]
    assert len(lost) == 1 and lost[0]["truncated"] == "aborted_generation"

    # (4) RETRY asks for a SMALLER unit of work, and never claims the model
    # merely forgot to call a tool.
    retry_tail = str((llm_payloads[2].get("messages") or [{}])[-1].get("content") or "").lower()
    assert "cut off mid-generation" in retry_tail
    assert "smaller" in retry_tail
    assert "you did not call any tools" not in retry_tail

    # (5) The partial preface never becomes the answer.
    assert "Good, I have the context" not in str(state.vars.get("_temp", {}).get("final_answer") or "")


@pytest.mark.basic
def test_ordinary_deferred_action_still_uses_the_followthrough_heuristic() -> None:
    """No land-grab: a real (fully accounted) plan-only turn keeps its own lane."""
    state, steps, _ = _run_loop_with(
        {
            "content": "Let me continue building the game files. I need to create the game loop.",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5462, "completion_tokens": 40, "total_tokens": 5502},
        }
    )
    names = [s for s, _ in steps]
    assert state.status == RunStatus.COMPLETED
    assert "parse_retry_plan_only" in names
    assert "parse_retry_truncated" not in names
    assert int(state.vars["scratchpad"].get("truncated_cycles", 0)) == 0
