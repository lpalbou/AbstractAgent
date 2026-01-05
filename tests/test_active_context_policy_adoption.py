from __future__ import annotations

from abstractagent.agents.codeact import CodeActAgent
from abstractagent.agents.react import ReactAgent
from abstractruntime import RunState, RunStatus, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


def _runtime() -> Runtime:
    return Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())


def _run_vars(*, max_history_messages: int) -> dict:
    # Include a system message + multiple conversation messages.
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "m2"},
        {"role": "user", "content": "m3"},
        {"role": "assistant", "content": "m4"},
    ]
    return {
        "context": {"task": "t", "messages": messages},
        "scratchpad": {"iteration": 0, "max_iterations": 5},
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {
            "max_iterations": 5,
            "current_iteration": 0,
            "max_history_messages": int(max_history_messages),
            "max_tokens": 1024,
        },
    }


def _run(*, vars: dict) -> RunState:
    return RunState(
        run_id="run",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="reason",
        vars=vars,
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )


def test_react_adapter_uses_runtime_active_context_policy() -> None:
    agent = ReactAgent(runtime=_runtime(), tools=[], max_iterations=5, max_history_messages=2, max_tokens=1024)
    reason_node = agent.workflow.get_node("reason")
    run = _run(vars=_run_vars(max_history_messages=2))

    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    msgs = payload.get("messages")
    assert isinstance(msgs, list)
    rendered = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in msgs if isinstance(m, dict))

    # System preserved, history limit applied to non-system only.
    assert "system: SYS" in rendered
    assert "user: m3" in rendered
    assert "assistant: m4" in rendered
    assert "user: m1" not in rendered
    assert "assistant: m2" not in rendered


def test_codeact_adapter_uses_runtime_active_context_policy() -> None:
    agent = CodeActAgent(runtime=_runtime(), tools=[], max_iterations=5, max_history_messages=2, max_tokens=1024)
    reason_node = agent.workflow.get_node("reason")
    run = _run(vars=_run_vars(max_history_messages=2))

    plan = reason_node(run, _Ctx())
    assert plan.effect is not None
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    msgs = payload.get("messages")
    assert isinstance(msgs, list)
    rendered = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in msgs if isinstance(m, dict))

    assert "system: SYS" in rendered
    assert "user: m3" in rendered
    assert "assistant: m4" in rendered
    assert "user: m1" not in rendered
    assert "assistant: m2" not in rendered
