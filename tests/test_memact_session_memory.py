from __future__ import annotations

from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, RunStatus, Runtime

from abstractagent.agents.memact import MemActAgent


def test_memact_agent_seeds_active_memory_from_session() -> None:
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    agent = MemActAgent(runtime=runtime, tools=[])

    agent.session_active_memory = {"version": 1, "persona": "p", "current_tasks": ["Do X"]}

    run_id = agent.start("task")
    state = runtime.tick(workflow=agent.workflow, run_id=run_id, max_steps=1)

    assert state.status == RunStatus.RUNNING
    runtime_ns = state.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    mem = runtime_ns.get("active_memory")
    assert isinstance(mem, dict)
    tasks = mem.get("current_tasks")
    assert isinstance(tasks, list)
    assert any(isinstance(t, dict) and t.get("text") == "Do X" for t in tasks)

