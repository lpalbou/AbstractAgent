from __future__ import annotations

from typing import Any, Dict, Optional

import pytest


@pytest.mark.basic
def test_react_keeps_tools_on_first_iteration_when_active_attachments_present() -> None:
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic
    from abstractcore.tools import ToolDefinition
    from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.scheduler.registry import WorkflowRegistry
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    expected_media = [{"$artifact": "a1", "filename": "notes.tsv"}]
    seen_payloads: list[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = dict(effect.payload or {})
        seen_payloads.append(payload)
        assert payload.get("media") == expected_media
        tools = payload.get("tools")
        assert isinstance(tools, list) and tools
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        assert params.get("tool_choice") != "none"
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )

    tool = ToolDefinition(name="tool_a", description="A", parameters={})
    wf = create_react_workflow(logic=ReActLogic(tools=[tool]), workflow_id="react_agent")
    runtime.workflow_registry.register(wf)

    vars: Dict[str, Any] = {"context": {"task": "Use attachments.", "messages": [], "attachments": expected_media}, "_runtime": {"inbox": []}}
    run_id = runtime.start(workflow=wf, vars=vars)

    for _ in range(50):
        state = runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED
    assert seen_payloads


@pytest.mark.basic
def test_react_keeps_tools_on_first_iteration_without_attachments() -> None:
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic
    from abstractcore.tools import ToolDefinition
    from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.scheduler.registry import WorkflowRegistry
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    seen_payloads: list[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = dict(effect.payload or {})
        seen_payloads.append(payload)
        tools = payload.get("tools")
        assert isinstance(tools, list) and tools
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        assert params.get("tool_choice") != "none"
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )

    tool = ToolDefinition(name="tool_a", description="A", parameters={})
    wf = create_react_workflow(logic=ReActLogic(tools=[tool]), workflow_id="react_agent")
    runtime.workflow_registry.register(wf)

    vars: Dict[str, Any] = {"context": {"task": "No attachments.", "messages": []}, "_runtime": {"inbox": []}}
    run_id = runtime.start(workflow=wf, vars=vars)

    for _ in range(50):
        state = runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED
    assert seen_payloads


@pytest.mark.basic
def test_react_includes_media_when_active_attachments_present() -> None:
    # Regression guard: attachments should continue to be forwarded as `media`.
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic
    from abstractcore.tools import ToolDefinition
    from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.scheduler.registry import WorkflowRegistry
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    expected_media = [{"$artifact": "a1", "filename": "notes.tsv"}]
    seen_payloads: list[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = dict(effect.payload or {})
        seen_payloads.append(payload)
        assert payload.get("media") == expected_media
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )

    tool = ToolDefinition(name="tool_a", description="A", parameters={})
    wf = create_react_workflow(logic=ReActLogic(tools=[tool]), workflow_id="react_agent")
    runtime.workflow_registry.register(wf)

    vars: Dict[str, Any] = {"context": {"task": "Use attachments.", "messages": [], "attachments": expected_media}, "_runtime": {"inbox": []}}
    run_id = runtime.start(workflow=wf, vars=vars)

    for _ in range(50):
        state = runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED
    assert seen_payloads
