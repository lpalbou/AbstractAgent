from __future__ import annotations

from abstractagent.agents.codeact import CodeActAgent
from abstractagent.agents.react import ReactAgent
from abstractruntime import EffectType, RunState, RunStatus, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def test_agents_include_recall_memory_tool_definition() -> None:
    react = ReactAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    assert react.logic is not None
    assert any(t.name == "recall_memory" for t in react.logic.tools)
    assert any(t.name == "inspect_vars" for t in react.logic.tools)
    assert any(t.name == "remember" for t in react.logic.tools)
    assert any(t.name == "compact_memory" for t in react.logic.tools)

    codeact = CodeActAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    assert codeact.logic is not None
    assert any(t.name == "recall_memory" for t in codeact.logic.tools)
    assert any(t.name == "inspect_vars" for t in codeact.logic.tools)
    assert any(t.name == "remember" for t in codeact.logic.tools)
    assert any(t.name == "compact_memory" for t in codeact.logic.tools)


def test_react_adapter_intercepts_recall_memory_as_memory_query_effect() -> None:
    agent = ReactAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "recall_memory", "arguments": {"span_id": "abc"}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_QUERY
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "recall_memory"
    assert plan.effect.payload.get("call_id") == "c1"
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_codeact_adapter_intercepts_recall_memory_as_memory_query_effect() -> None:
    agent = CodeActAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "recall_memory", "arguments": {"query": "alice"}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_QUERY
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "recall_memory"
    assert plan.effect.payload.get("call_id") == "c1"
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_react_adapter_intercepts_inspect_vars_as_vars_query_effect() -> None:
    agent = ReactAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "inspect_vars", "arguments": {"path": "scratchpad"}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.VARS_QUERY
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "inspect_vars"
    assert plan.effect.payload.get("call_id") == "c1"
    assert plan.effect.payload.get("path") == "scratchpad"
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_codeact_adapter_intercepts_inspect_vars_as_vars_query_effect() -> None:
    agent = CodeActAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "inspect_vars", "arguments": {"path": "scratchpad.foo", "keys_only": True}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.VARS_QUERY
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "inspect_vars"
    assert plan.effect.payload.get("call_id") == "c1"
    assert plan.effect.payload.get("path") == "scratchpad.foo"
    assert plan.effect.payload.get("keys_only") is True
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_react_adapter_intercepts_remember_as_memory_tag_effect() -> None:
    agent = ReactAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "remember", "arguments": {"span_id": "abc", "tags": {"topic": "api"}}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_TAG
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "remember"
    assert plan.effect.payload.get("call_id") == "c1"
    assert plan.effect.payload.get("span_id") == "abc"
    assert plan.effect.payload.get("tags") == {"topic": "api"}
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_codeact_adapter_intercepts_remember_as_memory_tag_effect() -> None:
    agent = CodeActAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "remember", "arguments": {"span_id": "abc", "tags": {"person": "alice"}, "merge": True}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_TAG
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "remember"
    assert plan.effect.payload.get("call_id") == "c1"
    assert plan.effect.payload.get("tags") == {"person": "alice"}
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_react_adapter_intercepts_compact_memory_as_memory_compact_effect() -> None:
    agent = ReactAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "compact_memory", "arguments": {"preserve_recent": 2, "compression_mode": "standard"}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_COMPACT
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "compact_memory"
    assert plan.effect.payload.get("call_id") == "c1"
    assert plan.effect.payload.get("preserve_recent") == 2
    assert plan.effect.payload.get("compression_mode") == "standard"
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []


def test_codeact_adapter_intercepts_compact_memory_as_memory_compact_effect() -> None:
    agent = CodeActAgent(runtime=_runtime(), tools=[], max_iterations=1, max_tokens=1024)
    wf = agent.workflow
    act_node = wf.get_node("act")

    run = RunState(
        run_id="run",
        workflow_id=wf.workflow_id,
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {},
            "_runtime": {},
            "_temp": {"pending_tool_calls": [{"name": "compact_memory", "arguments": {"preserve_recent": 0, "compression_mode": "heavy", "focus": "api"}, "call_id": "c1"}]},
            "_limits": {},
        },
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )

    plan = act_node(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.MEMORY_COMPACT
    assert plan.effect.result_key == "_temp.tool_results"
    assert plan.next_node == "observe"
    assert plan.effect.payload.get("tool_name") == "compact_memory"
    assert plan.effect.payload.get("call_id") == "c1"
    assert plan.effect.payload.get("focus") == "api"
    assert run.vars.get("_temp", {}).get("pending_tool_calls") == []
