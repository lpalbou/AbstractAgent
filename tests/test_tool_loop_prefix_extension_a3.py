"""Tool-loop prefix extension (mission A3, 2026-09-22).

The operator's live run 081d8daa went COLD from its third LLM call on, and the
next run of the session went cold too. Three agent-side defects combined:

1. PAIRING: a model that announces tool calls without ids got two different
   fallbacks (`call_{i+1}` in the durable assistant message, `str(idx)` in the
   batch the executor answered). Every announced id was "unanswered" (the payload
   said `[tool result missing (host error): web_search]`) and every real result
   was folded into one giant `[unpaired tool result]` USER message.
2. CARRIER STAMP: that user carrier was the payload's last message, so the
   runtime grounding pass stamped IT — and the next iteration rebuilt it without
   the envelope and stamped the new one (118-char divergence at message 13 of 21).
3. VOLATILE TAIL: `[loop] iteration N of M.` rode a trailing message that the
   next iteration dropped, so iteration N's prompt was never a prefix of N+1's.

The contract pinned here, with no model: every LLM payload of a tool loop is an
EXACT message-level prefix of the next one (after the runtime's own grounding
pass, which runs before the effect handler), results ride `tool` messages paired
to the announced ids, and the only envelope is the one the task was sent with.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.adapters.transcripts import (
    ensure_tool_call_ids,
    sanitize_transcript_messages,
    synthetic_call_id,
)
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolCall, ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.turn_grounding import SYNTHETIC_MESSAGE_KEY

pytestmark = pytest.mark.basic

ENVELOPE = "<runtime_metadata>"


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _big(name: str, it: int, slot: int) -> str:
    # Large, deterministic, web_search-shaped output (the live run's results were 4-30k chars).
    return json.dumps({"success": True, "query": f"{name} {it}.{slot}", "results": [f"row {i} " * 20 for i in range(40)]})


class _Loop:
    """Real Runtime + real react workflow; scripted LLM + scripted tool executor."""

    def __init__(self, *, tool_rounds: int = 3, calls_per_round: int = 3, ghost: bool = False) -> None:
        self.tool_rounds = tool_rounds
        self.calls_per_round = calls_per_round
        self.ghost = ghost
        self.payloads: List[List[Dict[str, Any]]] = []
        self.runtime = Runtime(
            run_store=InMemoryRunStore(),
            ledger_store=InMemoryLedgerStore(),
            effect_handlers={EffectType.LLM_CALL: self._llm, EffectType.TOOL_CALLS: self._tools},
        )
        self.workflow = create_react_workflow(
            logic=ReActLogic(tools=[ToolDefinition(name="web_search", description="Search", parameters={})]),
            workflow_id="a3_loop",
            provider="stub",
            model="stub",
            allowed_tools=["web_search"],
        )

    def _llm(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        self.payloads.append(json.loads(json.dumps(payload.get("messages") or [])))
        it = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if it <= self.tool_rounds:
            # NO call ids — exactly how the MLX lane's parsed tool calls arrive.
            calls = [{"name": "web_search", "arguments": {"query": f"q{it}.{s}"}} for s in range(self.calls_per_round)]
            return EffectOutcome.completed({"content": f"Searching {it}.", "tool_calls": calls})
        return EffectOutcome.completed({"content": "Summary.", "tool_calls": []})

    def _tools(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        it = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        calls = (effect.payload or {}).get("tool_calls") or []
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": _big("web_search", it, s), "error": None}
            for s, tc in enumerate(calls)
        ]
        if self.ghost:
            results.append({"call_id": "ghost", "name": "web_search", "success": True, "output": _big("ghost", it, 9), "error": None})
        return EffectOutcome.completed({"mode": "executed", "results": results})

    def run(self, task: str = "research the debate") -> RunState:
        run_id = self.runtime.start(
            workflow=self.workflow,
            vars={"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}},
            actor_id=None,
            session_id="sess-a3",
        )
        for _ in range(200):
            state = self.runtime.tick(workflow=self.workflow, run_id=run_id, max_steps=1)
            if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                break
        state = self.runtime.get_state(run_id)
        assert state.status == RunStatus.COMPLETED, f"run failed: {state.error}"
        return state


@pytest.mark.parametrize("ghost", [False, True], ids=["paired-only", "with-unpaired-carrier"])
def test_every_iteration_payload_is_an_exact_prefix_of_the_next(ghost: bool) -> None:
    loop = _Loop(ghost=ghost)
    loop.run()
    assert len(loop.payloads) == 4, "3 tool rounds + the final answer"
    for n in range(len(loop.payloads) - 1):
        a, b = loop.payloads[n], loop.payloads[n + 1]
        assert len(b) > len(a)
        for i, (ma, mb) in enumerate(zip(a, b)):
            assert _dumps(ma) == _dumps(mb), (
                f"iteration {n + 1} -> {n + 2} diverges at message {i} ({ma.get('role')}); "
                "a divergence anywhere but the appended tail costs the whole cached prefix after it"
            )


def test_results_ride_tool_messages_paired_to_the_announced_ids() -> None:
    loop = _Loop()
    loop.run()
    last = loop.payloads[-1]
    announced: List[str] = []
    for m in last:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            announced.extend(tc["id"] for tc in m["tool_calls"])
    answered = [m.get("tool_call_id") for m in last if m.get("role") == "tool"]
    assert announced and sorted(answered) == sorted(announced)
    assert set(announced) == {synthetic_call_id(i) for i in range(3)}
    for m in last:
        text = str(m.get("content") or "")
        assert "[tool result missing" not in text, "a real result was reported as a host error"
        assert "[unpaired tool result]" not in text, "a real result was folded into a user carrier"
    assert all('"success": true' in str(m["content"]) for m in last if m.get("role") == "tool")


@pytest.mark.parametrize("ghost", [False, True], ids=["paired-only", "with-unpaired-carrier"])
def test_only_the_task_carries_the_grounding_envelope_carriers_and_tails_never(ghost: bool) -> None:
    loop = _Loop(ghost=ghost)
    state = loop.run()
    for payload in loop.payloads:
        stamped = [i for i, m in enumerate(payload) if ENVELOPE in str(m.get("content") or "")]
        assert stamped == [0], f"envelope must live on the task only, found at {stamped}"
    last = loop.payloads[-1]
    carriers = [m for m in last if m.get(SYNTHETIC_MESSAGE_KEY) == "tool_result"]
    tails = [m for m in last if m.get(SYNTHETIC_MESSAGE_KEY) == "loop_tail"]
    durable_tails = [m for m in state.vars["context"]["messages"] if m.get(SYNTHETIC_MESSAGE_KEY) == "loop_tail"]
    if ghost:
        assert len(carriers) == 3, "one genuinely unpaired (ghost) result per round, marked"
        # The payload ends with a (user-role) carrier: chat-shape rule, chrome dropped.
        assert tails == [] and durable_tails == []
    else:
        assert carriers == []
        assert [t["content"] for t in tails] == [f"[loop] iteration {i} of 20." for i in (2, 3, 4)]
        assert [m["content"] for m in durable_tails] == [t["content"] for t in tails], "sent bytes == stored bytes"


# ---------------------------------------------------------------------------
# Unit level: the two seams, isolated
# ---------------------------------------------------------------------------


def test_ensure_tool_call_ids_fills_only_missing_ids_with_the_shared_formula() -> None:
    calls = [ToolCall(name="a", arguments={}), ToolCall(name="b", arguments={}, call_id="provider-7"), {"name": "c"}]
    ensure_tool_call_ids(calls)
    assert calls[0].call_id == "call_1"
    assert calls[1].call_id == "provider-7"
    assert calls[2]["call_id"] == "call_3"


def test_sanitize_marks_carriers_and_the_mark_survives_an_adjacent_user_merge() -> None:
    durable = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "id": "call_1", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "r1", "metadata": {"call_id": "call_1"}},
        {"role": "tool", "content": "stray", "metadata": {"call_id": "nobody"}},
        {"role": "user", "content": "operator guidance"},
    ]
    out = sanitize_transcript_messages(durable)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "user"]
    assert out[2]["tool_call_id"] == "call_1" and out[2]["content"] == "r1"
    merged = out[3]
    assert merged["content"].startswith("[unpaired tool result]: stray")
    assert merged["content"].endswith("operator guidance")
    assert merged[SYNTHETIC_MESSAGE_KEY] == "tool_result"
