from __future__ import annotations

from typing import Any, Dict, Optional


def test_react_runtime_forwards_context_attachments_as_llm_call_media() -> None:
    """Level A: ensure artifact-backed attachments reach EffectType.LLM_CALL via `payload.media`."""
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic
    from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.scheduler.registry import WorkflowRegistry
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    expected = [{"$artifact": "a1", "filename": "notes.txt"}]
    seen_media: list[Any] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = dict(effect.payload or {})
        seen_media.append(payload.get("media"))
        assert payload.get("media") == expected
        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )

    wf = create_react_workflow(logic=ReActLogic(tools=[]), workflow_id="react_agent")
    runtime.workflow_registry.register(wf)

    vars: Dict[str, Any] = {"context": {"task": "Use attachments.", "messages": [], "attachments": expected}, "_runtime": {"inbox": []}}
    run_id = runtime.start(workflow=wf, vars=vars)

    for _ in range(50):
        state = runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED
    assert seen_media and seen_media[0] == expected

