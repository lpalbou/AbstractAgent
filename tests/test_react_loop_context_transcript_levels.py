from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore


class _Ctx:
    def now_iso(self) -> str:
        return "2026-01-13T00:00:00+00:00"


def _base_vars(*, task: str) -> Dict[str, Any]:
    # Keep vars minimal; the adapter creates missing namespaces/limits.
    return {"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}}


def _strip_runtime_prefixes(content: str) -> str:
    text = str(content or "")
    if text.startswith("<runtime_metadata>") and "</runtime_metadata>" in text:
        text = text.split("</runtime_metadata>", 1)[1].lstrip()
    if text.startswith("[") and "]" in text:
        text = text.split("]", 1)[1].lstrip()
    return text


@pytest.mark.basic
def test_react_loop_context_transcript_level_a_basic() -> None:
    captured_llm_payloads: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured_llm_payloads.append({"iteration": run.vars.get("_limits", {}).get("current_iteration"), "payload": dict(payload)})

        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed(
                {
                    "content": "Checking the workspace.",
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
                }
            )
        if iteration == 2:
            # After observing list_files, create a folder via execute_command.
            return EffectOutcome.completed(
                {
                    "content": "Creating the project folder.",
                    "tool_calls": [{"name": "execute_command", "arguments": {"command": "mkdir -p project"}, "call_id": "call_2"}],
                }
            )
        # Done.
        return EffectOutcome.completed({"content": "Created the project folder.", "tool_calls": []})

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
            if name == "list_files":
                results.append({"call_id": call_id, "name": "list_files", "success": True, "output": "a.txt", "error": None})
            elif name == "execute_command":
                results.append(
                    {
                        "call_id": call_id,
                        "name": "execute_command",
                        "success": True,
                        "output": {"success": True, "command": tc.get("arguments", {}).get("command")},
                        "error": None,
                    }
                )
            else:
                results.append({"call_id": call_id, "name": str(name or ""), "success": False, "output": None, "error": "unknown"})
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
        workflow_id="react_agent_test",
        provider="stub",
        model="stub",
        allowed_tools=["list_files", "execute_command"],
    )

    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="Create a project folder"), actor_id=None, session_id=None)

    # Drive to completion.
    for _ in range(50):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED

    # LLM-visible context assertions.
    assert len(captured_llm_payloads) >= 2

    first = captured_llm_payloads[0]["payload"]
    first_msgs = first.get("messages")
    assert isinstance(first_msgs, list) and first_msgs
    assert first_msgs[0].get("role") == "user"
    first_content = _strip_runtime_prefixes(str(first_msgs[0].get("content") or ""))
    # Adjacency guard (2026-07-09): on a first turn the volatile [loop] tail MERGES into the
    # trailing user message (alternation-strict templates reject user,user), so the task
    # message carries task + tail rather than a second consecutive user message.
    assert first_content.startswith("Create a project folder")
    assert "[loop] iteration 1" in first_content
    roles_first = [m.get("role") for m in first_msgs]
    assert all(not (a == b == "user") for a, b in zip(roles_first, roles_first[1:]))
    params1 = first.get("params") if isinstance(first.get("params"), dict) else {}
    assert "max_tokens" not in params1, "ReAct should not enforce tiny per-step output caps by default"

    second_payload = captured_llm_payloads[1]["payload"]
    second_msgs = second_payload.get("messages")
    assert isinstance(second_msgs, list)
    assert any(m.get("role") == "tool" and "a.txt" in str(m.get("content") or "") for m in second_msgs)
    # Context fidelity (0213): the assistant tool-call turn is preserved AND carries the model's own
    # reasoning content in the transcript (not dropped to content="").
    assistant_tool_msgs = [
        m for m in second_msgs if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)
    ]
    assert assistant_tool_msgs
    assert any("Checking the workspace." in str(m.get("content") or "") for m in assistant_tool_msgs)
    # Prompt-prefix cache stability (0212): the scratchpad is NO LONGER rendered into the system
    # prompt (that mutated the cached prefix and double-carried observations). The system prompt must
    # stay stable across iterations, so it must not contain per-cycle scratchpad state.
    sys1 = captured_llm_payloads[0]["payload"].get("system_prompt")
    sys2 = second_payload.get("system_prompt")
    assert isinstance(sys2, str)
    assert "Scratchpad" not in sys2
    assert "[cycle 1]" not in sys2
    # The system prompt prefix is byte-identical between iterations (cache-friendly).
    assert sys1 == sys2
    # Volatile loop position rides the TRAILING message, never the cached prefix.
    assert second_msgs[-1].get("role") == "user"
    assert "iteration 2" in str(second_msgs[-1].get("content") or "")
    assert "Iteration:" not in str(sys2)


@pytest.mark.integration
def test_react_loop_context_transcript_level_b_restart(tmp_path: Any) -> None:
    captured_llm_payloads: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured_llm_payloads.append({"iteration": run.vars.get("_limits", {}).get("current_iteration"), "payload": dict(payload)})

        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed(
                {"content": "Check.", "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}]}
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        tool_calls = payload.get("tool_calls")
        assert isinstance(tool_calls, list)
        results = [{"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "x.txt", "error": None} for tc in tool_calls]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    run_store = JsonFileRunStore(tmp_path / "runs")
    ledger_store = JsonlLedgerStore(tmp_path / "ledgers")

    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="react_agent_test_restart",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )

    # Runtime #1: run until after first tool observation is persisted.
    runtime1 = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    run_id = runtime1.start(workflow=workflow, vars=_base_vars(task="List files"), actor_id=None, session_id=None)
    for _ in range(10):
        state = runtime1.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status == RunStatus.WAITING:
            break
        # Stop once we have appended the tool observation.
        msgs = state.vars.get("context", {}).get("messages") if isinstance(state.vars.get("context"), dict) else []
        if isinstance(msgs, list) and any(isinstance(m, dict) and m.get("role") == "tool" for m in msgs):
            break

    # Runtime #2: simulate restart and continue.
    runtime2 = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    for _ in range(20):
        state2 = runtime2.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state2.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final_state = runtime2.get_state(run_id)
    assert final_state.status == RunStatus.COMPLETED

    # The second LLM call (post-restart) must still see the prior tool result in its messages.
    assert len(captured_llm_payloads) >= 2
    second_msgs = captured_llm_payloads[1]["payload"].get("messages")
    assert isinstance(second_msgs, list)
    assert any(m.get("role") == "tool" and "x.txt" in str(m.get("content") or "") for m in second_msgs)


@pytest.mark.e2e
def test_react_loop_context_transcript_level_c_lmstudio(tmp_path: Any) -> None:
    """Level C: real provider/tool execution (opt-in).

    Enable with:
      ABSTRACT_E2E_LMSTUDIO=1
    """
    if os.environ.get("ABSTRACT_E2E_LMSTUDIO") != "1":
        pytest.skip("Set ABSTRACT_E2E_LMSTUDIO=1 to run this test.")

    from abstractcore.tools.common_tools import list_files
    from abstractruntime.integrations.abstractcore.effect_handlers import build_effect_handlers
    from abstractruntime.integrations.abstractcore.llm_client import LocalAbstractCoreLLMClient
    from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-next-80b")

    # Deterministic workspace for the tool call.
    workdir = tmp_path / "dir"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "hello.txt").write_text("hi", encoding="utf-8")

    class RecordingLLM:
        def __init__(self, inner: Any):
            self.inner = inner
            self.calls: list[dict[str, Any]] = []

        def generate(self, *, prompt: str, messages: Optional[List[Dict[str, str]]] = None, system_prompt: Optional[str] = None, tools: Any = None, params: Any = None) -> Dict[str, Any]:
            self.calls.append(
                {
                    "prompt": prompt,
                    "messages": [dict(m) for m in (messages or [])],
                    "system_prompt": system_prompt,
                }
            )
            return self.inner.generate(prompt=prompt, messages=messages, system_prompt=system_prompt, tools=tools, params=params)

    llm_inner = LocalAbstractCoreLLMClient(provider="lmstudio", model=model, llm_kwargs={"base_url": base_url})
    llm = RecordingLLM(llm_inner)
    tool_exec = MappingToolExecutor.from_tools([list_files])

    runtime = Runtime(
        run_store=JsonFileRunStore(tmp_path / "runs"),
        ledger_store=JsonlLedgerStore(tmp_path / "ledgers"),
        effect_handlers=build_effect_handlers(llm=llm, tools=tool_exec),
    )

    workflow = create_react_workflow(
        logic=ReActLogic(tools=[list_files._tool_definition]),
        workflow_id="react_agent_e2e",
        provider="lmstudio",
        model=model,
        allowed_tools=["list_files"],
    )

    task = f"List the files in directory '{workdir}' using the list_files tool, then answer with the filenames."
    run_id = runtime.start(workflow=workflow, vars=_base_vars(task=task))

    for _ in range(50):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED

    # Expect at least two LLM calls: one to request the tool, one to use the observation.
    assert len(llm.calls) >= 2
    second = llm.calls[1]
    msgs = second.get("messages")
    assert isinstance(msgs, list)
    assert any(m.get("role") == "tool" for m in msgs), "Expected a tool message in the second LLM call context"


@pytest.mark.basic
def test_ask_user_orphaned_tool_calls_repaired_in_llm_payload() -> None:
    """Multi-turn correctness on strict providers (found live 2026-07-09, pre-existing at HEAD):
    ask_user resolves via an ASK_USER wait + a user message — never a tool message — so the
    assistant `tool_calls` turn stays unanswered in durable history and native OpenAI 400s the
    NEXT request. The sanitizer must synthesize an adjacent tool result in the payload (durable
    history untouched)."""
    captured: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(dict(payload))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed(
                {"content": "", "tool_calls": [{"name": "ask_user", "arguments": {"question": "Favorite color?"}, "call_id": "call_ask"}]}
            )
        return EffectOutcome.completed({"content": "Your color is ultramarine.", "tool_calls": []})

    def ask_user_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, effect, default_next_node
        from abstractruntime.core.models import WaitReason, WaitState

        return EffectOutcome.waiting(WaitState(reason=WaitReason.USER, wait_key="ask-1", result_key="_temp.user_response"))

    from abstractagent.logic.builtins import ASK_USER_TOOL

    logic = ReActLogic(tools=[ASK_USER_TOOL])
    workflow = create_react_workflow(logic=logic)
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.ASK_USER: ask_user_handler},
    )
    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="ask me my color, then conclude"))
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=20)
    assert state.status == RunStatus.WAITING

    state = runtime.resume(workflow=workflow, run_id=run_id, wait_key="ask-1", payload={"response": "ultramarine"}, max_steps=20)
    assert state.status == RunStatus.COMPLETED

    # The post-resume LLM request must contain NO orphaned assistant tool_calls: every id is
    # answered by an adjacent tool message (the synthetic interactive-handled result).
    final_msgs = captured[-1].get("messages") or []
    for idx, m in enumerate(final_msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = {str(tc.get("id")) for tc in m["tool_calls"]}
            j = idx + 1
            while j < len(final_msgs) and final_msgs[j].get("role") == "tool":
                ids.discard(str(final_msgs[j].get("tool_call_id") or ""))
                j += 1
            assert not ids, f"orphaned tool_call ids in payload: {ids}"

    # Durable history is untouched: no synthetic tool message persisted.
    run = runtime._run_store.load(run_id)
    durable = (run.vars.get("context") or {}).get("messages") or []
    assert not any(m.get("role") == "tool" and "handled interactively" in str(m.get("content") or "") for m in durable)


@pytest.mark.basic
def test_loop_tail_merges_into_trailing_user_message() -> None:
    """Adjacency guard (Critic-3, 2026-07-09): when the payload already ends with a user
    message (first turn), the volatile [loop] tail merges INTO it — alternation-strict
    templates reject user,user, and a separate banner steals the grounding envelope's
    last-user-message slot from the real task."""
    captured: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        captured.append(dict(effect.payload if isinstance(effect.payload, dict) else {}))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed(
                {"content": "Looking.", "tool_calls": [{"name": "list_files", "arguments": {}, "call_id": "c1"}]}
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        calls = (effect.payload or {}).get("tool_calls") or []
        return EffectOutcome.completed(
            {"mode": "executed",
             "results": [{"call_id": c.get("call_id"), "name": c.get("name"), "success": True,
                          "output": "a.txt", "error": None} for c in calls]}
        )

    list_files = ToolDefinition(name="list_files", description="List files", parameters={})
    logic = ReActLogic(tools=[list_files])
    workflow = create_react_workflow(logic=logic)
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="list the files"))
    state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=30)
    assert state.status == RunStatus.COMPLETED

    # FIRST call: ends with a user message -> tail MERGED (no user,user anywhere).
    first_msgs = captured[0].get("messages") or []
    roles = [m.get("role") for m in first_msgs]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), roles
    assert "[loop] iteration 1" in str(first_msgs[-1].get("content") or "")
    assert "list the files" in str(first_msgs[-1].get("content") or "")  # task + tail together

    # SECOND call: ends with tool results -> tail is a separate trailing user message
    # (the grounding envelope may be prefixed into it — both are volatile per-call state).
    second_msgs = captured[1].get("messages") or []
    assert second_msgs[-1].get("role") == "user"
    assert "[loop] iteration 2" in str(second_msgs[-1].get("content") or "")
    roles2 = [m.get("role") for m in second_msgs]
    assert all(not (a == b == "user") for a, b in zip(roles2, roles2[1:])), roles2
