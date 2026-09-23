"""ReAct conversational byte-stability (mission A, 2026-09-22).

Companion to `test_react_prompt_prefix_stability.py`, which pins the prefix
contract WITHIN one run (across tool-loop iterations). This file pins it
ACROSS conversational turns, which is where the framework was still losing the
whole prompt cache every turn:

- the current user turn was sent DECORATED (`<runtime_metadata>…` envelope at
  the head, `[loop] iteration 1 of 20.` merged at the tail) and stored PLAIN,
  so one turn later the same message re-rendered with different bytes and
  turn N's prompt stopped being a byte-prefix of turn N+1's;
- measured through the real stack on the MLX native lane (4B pair, 8 turns,
  `untracked/missionA/replay_runtime.py`): 185-388 tokens re-prefilled every
  turn before, 105-121 after — and the ceiling (longest common prefix with the
  previous prompt) went from ~94 tokens short of the previous prompt to exactly
  its full length.

No model is involved here: the contract is on bytes, and a scripted LLM handler
drives the real Runtime + the real react workflow. History for turn 2 is
reconstructed through `abstractruntime.session_chat_messages`, which is exactly
what the gateway's `_seed_session_history` calls on every new turn of a session.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.session_history import session_chat_messages
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _tools() -> List[ToolDefinition]:
    return [
        ToolDefinition(name="list_files", description="List", parameters={}),
        ToolDefinition(name="execute_command", description="Cmd", parameters={}),
    ]


def _workflow():
    return create_react_workflow(
        logic=ReActLogic(tools=_tools()),
        workflow_id="react_turn_bytes",
        provider="stub",
        model="stub",
        allowed_tools=["list_files", "execute_command"],
    )


class _Harness:
    """Real Runtime + real react workflow; the LLM is scripted, nothing else is.

    NOTE: only `messages` and `system_prompt` are captured, never the whole
    payload — `params` carries a runtime-injected `on_progress` CALLABLE, so a
    whole-payload `json.dumps` raises (and is not what this file is about).
    """

    def __init__(self, script) -> None:
        self.script = script
        self.messages: List[List[Dict[str, Any]]] = []
        self.system_prompts: List[str] = []
        self.run_store = InMemoryRunStore()
        self.ledger_store = InMemoryLedgerStore()
        self.runtime = Runtime(
            run_store=self.run_store,
            ledger_store=self.ledger_store,
            effect_handlers={
                EffectType.LLM_CALL: self._llm,
                EffectType.TOOL_CALLS: self._tools_exec,
            },
        )
        self.workflow = _workflow()

    def _llm(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        self.messages.append(json.loads(json.dumps(payload.get("messages") or [])))
        self.system_prompts.append(str(payload.get("system_prompt") or ""))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        return EffectOutcome.completed(self.script(len(self.messages), iteration))

    def _tools_exec(self, run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
                    for tc in (payload.get("tool_calls") or [])
                ],
            }
        )

    def turn(self, task: str, *, session_id: str = "sess-bytes") -> RunState:
        """One conversational turn, seeded exactly as the gateway seeds one."""
        seeded = session_chat_messages(
            run_store=self.run_store,
            ledger_store=self.ledger_store,
            session_id=session_id,
        )
        run_id = self.runtime.start(
            workflow=self.workflow,
            vars={"context": {"task": task, "messages": seeded}, "_runtime": {"inbox": []}},
            actor_id=None,
            session_id=session_id,
        )
        for _ in range(60):
            state = self.runtime.tick(workflow=self.workflow, run_id=run_id, max_steps=1)
            if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                break
        state = self.runtime.get_state(run_id)
        assert state.status == RunStatus.COMPLETED, f"turn failed: {state.error}"
        return state


def _chat_script(call_no: int, iteration: int) -> Dict[str, Any]:
    del iteration
    return {"content": f"Answer {call_no}.", "tool_calls": []}


def test_turn_two_payload_starts_with_turn_ones_payload_byte_for_byte() -> None:
    h = _Harness(_chat_script)
    h.turn("identify yourself")
    h.turn("what is your key purpose?")

    assert len(h.messages) == 2, "a no-tool chat turn is one LLM call"
    first, second = h.messages[0], h.messages[1]

    # The cacheable head must be byte-identical too, or nothing after it matters.
    assert h.system_prompts[0] == h.system_prompts[1]

    assert len(second) == len(first) + 2  # previous answer + the new question
    assert _dumps(second[: len(first)]) == _dumps(first), (
        "turn 1's messages must reappear byte-identical at the head of turn 2 — "
        "including the runtime grounding envelope the turn was sent with"
    )
    assert second[len(first)] == {"role": "assistant", "content": "Answer 1."}
    assert str(second[-1]["content"]).endswith("what is your key purpose?")


def test_the_chat_shape_user_turn_carries_no_loop_chrome() -> None:
    h = _Harness(_chat_script)
    h.turn("identify yourself")

    sent = h.messages[0]
    assert len(sent) == 1
    content = str(sent[0]["content"])
    assert content.startswith("<runtime_metadata>")
    assert content.endswith("identify yourself")
    assert "[loop]" not in content, (
        "the loop-position line is chrome; merging it into a DURABLE user message "
        "is what made that message un-replayable byte-for-byte"
    )


def test_the_durable_transcript_stores_exactly_what_was_sent() -> None:
    h = _Harness(_chat_script)
    state = h.turn("identify yourself")

    sent = h.messages[0][0]["content"]
    stored = state.vars["context"]["messages"][0]
    assert stored["role"] == "user"
    assert stored["content"] == sent, "sent bytes and stored bytes are the same bytes"


def test_a_tool_loop_rides_its_chrome_on_a_durable_marked_tail_and_stamps_nothing_else() -> None:
    """Mission A3 contract (supersedes mission A's "trailing volatile" pin).

    Chrome still never enters the task message. In the tool-loop shape it is now
    appended to the DURABLE transcript, marked adapter-authored, so iteration N's
    payload is an exact prefix of iteration N+1's; and it no longer attracts the
    per-iteration grounding envelope (the old pin asserted the envelope landed on
    the volatile tail — that WAS the cache-killing divergence: bytes present in
    iteration N, gone in N+1)."""

    def script(call_no: int, iteration: int) -> Dict[str, Any]:
        del call_no
        if iteration == 1:
            return {
                "content": "Looking.",
                "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}],
            }
        return {"content": "Done.", "tool_calls": []}

    h = _Harness(script)
    state = h.turn("create a project folder")

    assert len(h.messages) == 2
    first, second = h.messages[0], h.messages[1]

    # Iteration 1 (chat shape): the task message alone, undecorated by chrome.
    assert len(first) == 1
    assert "[loop]" not in str(first[0]["content"])

    # Iteration 2 (tool-loop shape): task, assistant tool_calls, tool result, tail.
    tail = second[-1]
    assert tail.get("role") == "user"
    assert tail.get("volatile") is None
    assert tail.get("_af_synthetic") == "loop_tail"
    assert str(tail.get("content") or "") == "[loop] iteration 2 of 20."
    assert _dumps(second[: len(first)]) == _dumps(first)
    # ONE envelope in the whole payload: the task's, stamped when it was sent.
    assert sum("<runtime_metadata>" in str(m.get("content") or "") for m in second) == 1
    assert str(second[0]["content"]).startswith("<runtime_metadata>")
    # The tail is durable: stored exactly as it was sent.
    stored_tails = [m for m in state.vars["context"]["messages"] if m.get("_af_synthetic") == "loop_tail"]
    assert [m["content"] for m in stored_tails] == ["[loop] iteration 2 of 20."]
