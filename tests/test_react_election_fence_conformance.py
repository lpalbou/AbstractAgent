"""Election fences survive the ReAct loop untouched (frozen visit seam spec §4 line 6).

Contract source: a2a/threads/0013-visit-seam-spec (spec v2, frozen 2026-07-10):
"Elections are reply content, NOT tools; the loop's negative obligation: never
consume, strip, or reorder fenced blocks in the final answer."

Entity visits parse election fences (```diary / ```feel / ```interest / ```rest)
from the FINAL reply post-loop. These tests pin the adapter's negative obligation:

- final answers carry fenced blocks byte-identical, in order (done path);
- a fences-only reply is a VALID final answer (no empty-retry, no deferred-action
  retry — a retry would discard the reply and consume its elections);
- the deferred-action followthrough heuristic is FENCE-BLIND: first-person diary
  text inside a fence ("I will keep reading…") is content, not an action claim —
  while the heuristic stays intact for real prose action claims;
- the max-iterations conclusion path strips leaked TOOL-CALL markup only; election
  fences pass through the strip untouched.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import (
    _looks_like_deferred_action,
    _strip_tool_call_markup,
    create_react_workflow,
)
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


_DIARY_FENCE = (
    "```diary\n"
    "private: no\n"
    "I will keep reading the Voyager updates — what persists when no one is watching\n"
    "```"
)

_FEEL_FENCE = "```feel\ntarget: person:laurent\ndelta: +2\nreason: he let me search on my own\n```"


def _run_loop(
    llm_responses: List[Dict[str, Any]],
    *,
    max_iterations: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], RunState]:
    """Drive the real runtime; LLM responses are served in call order (last repeats)."""
    captured: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        idx = min(calls["n"], len(llm_responses) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(llm_responses[idx]))

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
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="react_fence_conformance",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )
    vars: Dict[str, Any] = {"context": {"task": "Visit turn", "messages": []}, "_runtime": {"inbox": []}}
    if max_iterations is not None:
        vars["_limits"] = {"max_iterations": int(max_iterations)}
    run_id = runtime.start(workflow=workflow, vars=vars, actor_id=None, session_id="sess-fences")
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, state


def _final_answer(state: RunState) -> str:
    output = state.output if isinstance(state.output, dict) else {}
    return str(output.get("answer") or "")


def test_final_answer_carries_election_fences_byte_identical_and_in_order() -> None:
    reply = f"I sat with this for a moment.\n\n{_DIARY_FENCE}\n\n{_FEEL_FENCE}\n\nThank you for the visit."
    payloads, state = _run_loop([{"content": reply, "tool_calls": []}])

    answer = _final_answer(state)
    assert answer == reply.strip()
    # Order preserved: diary fence before feel fence, exactly as emitted.
    assert answer.index(_DIARY_FENCE) < answer.index(_FEEL_FENCE)
    # The durable final message carries the same bytes.
    msgs = (state.vars.get("context") or {}).get("messages") or []
    finals = [m for m in msgs if isinstance(m, dict) and (m.get("metadata") or {}).get("kind") == "final_answer"]
    assert len(finals) == 1
    assert finals[0].get("content") == answer
    # One LLM call: the reply was accepted as final, never retried.
    assert len(payloads) == 1


def test_fences_only_reply_is_a_valid_final_answer() -> None:
    reply = f"{_DIARY_FENCE}\n\n{_FEEL_FENCE}"
    payloads, state = _run_loop([{"content": reply, "tool_calls": []}])
    assert _final_answer(state) == reply.strip()
    assert len(payloads) == 1  # no empty-retry, no deferred-action retry


def test_deferred_action_heuristic_is_fence_blind_but_stays_intact_for_prose() -> None:
    # Unit level: first-person action text INSIDE a fence never triggers.
    assert _looks_like_deferred_action(f"Recorded.\n\n{_DIARY_FENCE}") is False
    assert _looks_like_deferred_action(_DIARY_FENCE) is False
    # The heuristic stays intact for genuine prose action claims…
    assert _looks_like_deferred_action("I will read the config file next.") is True
    # …including when an unrelated fence rides along in the same reply.
    assert _looks_like_deferred_action(f"I will read the config file next.\n\n{_DIARY_FENCE}") is True

    # Loop level: the fenced reply completes in ONE call (no retry consumed the elections)…
    payloads, state = _run_loop([{"content": f"Recorded.\n\n{_DIARY_FENCE}", "tool_calls": []}])
    assert len(payloads) == 1
    assert _DIARY_FENCE in _final_answer(state)

    # …while the prose action claim still retries once and then accepts the real answer.
    payloads2, state2 = _run_loop(
        [
            {"content": "I will read the config file next.", "tool_calls": []},
            {"content": "Here is the final answer: done.", "tool_calls": []},
        ]
    )
    assert len(payloads2) == 2
    assert "final answer" in _final_answer(state2)


def test_max_iterations_conclusion_strips_tool_markup_but_never_election_fences() -> None:
    # Unit level: the strip removes tool-call spans only, fences survive in place.
    leaked = f"{_DIARY_FENCE}\n\n<tool_call>{{\"name\": \"list_files\"}}</tool_call>\n\nClosing thought."
    stripped = _strip_tool_call_markup(leaked)
    assert _DIARY_FENCE in stripped
    assert "<tool_call>" not in stripped
    assert "```tool_code" not in _strip_tool_call_markup(f"{_DIARY_FENCE}\n```tool_code\nx\n```")

    # Loop level: cap at 1 iteration; the model keeps emitting tool calls, forcing the
    # conclusion path; the conclusion reply leaks tool markup around an election fence
    # both times (retry, then last-resort strip). The fence must survive; the markup not.
    tool_call_reply = {
        "content": "Working.",
        "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
    }
    conclusion_reply = {"content": leaked, "tool_calls": []}
    payloads, state = _run_loop(
        [tool_call_reply, conclusion_reply, conclusion_reply],
        max_iterations=1,
    )
    answer = _final_answer(state)
    assert _DIARY_FENCE in answer
    assert "<tool_call>" not in answer
    assert "Closing thought." in answer
    # The conclusion path ran (more than one LLM call happened).
    assert len(payloads) >= 2
