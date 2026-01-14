from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _base_vars(*, task: str) -> Dict[str, Any]:
    return {"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}}


@pytest.mark.basic
def test_react_followthrough_retries_plan_only_no_tool_call_steps() -> None:
    """If the model outputs an intermediate "I'll do X next" message with no tool calls,
    the adapter should retry instead of terminating the loop early.
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
            # Plan-like followthrough without tool calls: should trigger retry.
            return EffectOutcome.completed(
                {
                    "content": "Let me start by setting up the project structure and creating the main file.",
                    "tool_calls": [],
                    "finish_reason": "stop",
                }
            )
        if idx == 3:
            sys = str(payload.get("system_prompt") or "")
            assert "did not call any tools" in sys.lower() or "call the next tool" in sys.lower()
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
        results = [{"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None} for tc in tool_calls]
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
        workflow_id="react_agent_followthrough_retry",
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

