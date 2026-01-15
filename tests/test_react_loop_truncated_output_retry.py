from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


def _base_vars(*, task: str) -> Dict[str, Any]:
    return {"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}}


@pytest.mark.basic
def test_react_truncated_output_retries_and_continues() -> None:
    """Regression guard: if the model hits output limit, do not treat as final.

    This simulates a provider returning `finish_reason="length"` with no tool calls.
    The adapter must retry (with corrective guidance) and continue tool-driven progress.
    """

    llm_payloads: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(dict(payload))

        idx = len(llm_payloads)
        if idx == 1:
            return EffectOutcome.completed(
                {
                    "content": "Checking workspace.",
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
                    "finish_reason": "tool_calls",
                }
            )
        if idx == 2:
            # Provider hit output limit and returned no tool calls.
            return EffectOutcome.completed(
                {
                    "content": "Long plan that would normally continue…",
                    "tool_calls": [],
                    "finish_reason": "length",
                }
            )
        if idx == 3:
            sys = str(payload.get("system_prompt") or "")
            assert "output token limit" in sys.lower()
            assert "keep tool call arguments small" in sys.lower()
            msgs = payload.get("messages")
            assert isinstance(msgs, list) and msgs
            assert str(payload.get("prompt") or "") == ""
            return EffectOutcome.completed(
                {
                    "content": "Creating folder.",
                    "tool_calls": [{"name": "execute_command", "arguments": {"command": "mkdir -p project"}, "call_id": "call_2"}],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        tool_calls = payload.get("tool_calls")
        assert isinstance(tool_calls, list)
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            assert isinstance(tc, dict)
            name = tc.get("name")
            call_id = tc.get("call_id")
            results.append({"call_id": call_id, "name": name, "success": True, "output": "ok", "error": None})
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
        workflow_id="react_agent_truncated_retry",
        provider="stub",
        model="stub",
        allowed_tools=["list_files", "execute_command"],
    )

    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="Create a folder"), actor_id=None, session_id=None)

    for _ in range(100):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    assert len(llm_payloads) >= 3


@pytest.mark.integration
def test_react_truncated_output_retry_survives_restart(tmp_path: Any) -> None:
    """Level B guard: retry bookkeeping persists across restarts (file-backed stores)."""

    llm_payloads: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(dict(payload))

        idx = len(llm_payloads)
        if idx == 1:
            return EffectOutcome.completed(
                {
                    "content": "Checking workspace.",
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
                    "finish_reason": "tool_calls",
                }
            )
        if idx == 2:
            return EffectOutcome.completed(
                {
                    "content": "Long plan…",
                    "tool_calls": [],
                    "finish_reason": "length",
                }
            )
        if idx == 3:
            sys = str(payload.get("system_prompt") or "")
            assert "truncated" in sys.lower() or "output token limit" in sys.lower()
            return EffectOutcome.completed(
                {
                    "content": "Creating folder.",
                    "tool_calls": [{"name": "execute_command", "arguments": {"command": "mkdir -p project"}, "call_id": "call_2"}],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        tool_calls = payload.get("tool_calls")
        assert isinstance(tool_calls, list)
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            assert isinstance(tc, dict)
            name = tc.get("name")
            call_id = tc.get("call_id")
            results.append({"call_id": call_id, "name": name, "success": True, "output": "ok", "error": None})
        return EffectOutcome.completed({"mode": "executed", "results": results})

    run_store = JsonFileRunStore(tmp_path / "runs")
    ledger_store = JsonlLedgerStore(tmp_path / "ledgers")

    workflow = create_react_workflow(
        logic=ReActLogic(
            tools=[
                ToolDefinition(name="list_files", description="List", parameters={}),
                ToolDefinition(name="execute_command", description="Cmd", parameters={}),
            ]
        ),
        workflow_id="react_agent_truncated_retry_restart",
        provider="stub",
        model="stub",
        allowed_tools=["list_files", "execute_command"],
    )

    # Runtime #1: run through the truncated response and stop after the adapter schedules a retry.
    runtime1 = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    run_id = runtime1.start(workflow=workflow, vars=_base_vars(task="Create a folder"), actor_id=None, session_id=None)

    for _ in range(50):
        state = runtime1.tick(workflow=workflow, run_id=run_id, max_steps=1)
        scratchpad = state.vars.get("scratchpad") if isinstance(state.vars, dict) else None
        if isinstance(scratchpad, dict) and int(scratchpad.get("truncated_response_retry_count") or 0) >= 1:
            break

    # Runtime #2: restart and finish.
    runtime2 = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    for _ in range(100):
        state2 = runtime2.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state2.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime2.get_state(run_id)
    assert final.status == RunStatus.COMPLETED
    assert len(llm_payloads) >= 3
