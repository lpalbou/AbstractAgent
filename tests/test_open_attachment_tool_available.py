from __future__ import annotations


def _tool_names(agent) -> set[str]:
    logic = getattr(agent, "logic", None)
    assert logic is not None
    tools = getattr(logic, "tools", None)
    assert isinstance(tools, list)
    out: set[str] = set()
    for t in tools:
        name = getattr(t, "name", None)
        if isinstance(name, str) and name.strip():
            out.add(name.strip())
    return out


def test_open_attachment_tool_is_available_in_default_agents() -> None:
    from abstractagent.agents.codeact import CodeActAgent
    from abstractagent.agents.memact import MemActAgent
    from abstractagent.agents.react import ReactAgent
    from abstractruntime.core.runtime import Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())

    assert "open_attachment" in _tool_names(ReactAgent(runtime=runtime, tools=[]))
    assert "open_attachment" in _tool_names(CodeActAgent(runtime=runtime, tools=[]))
    assert "open_attachment" in _tool_names(MemActAgent(runtime=runtime, tools=[]))

