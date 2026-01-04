from __future__ import annotations

from abstractagent.agents.codeact import CodeActAgent
from abstractagent.agents.react import ReactAgent
from abstractruntime.core.config import RuntimeConfig
from abstractruntime.core.runtime import Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def test_react_agent_start_seeds_runtime_provider_and_model_from_runtime_config() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        config=RuntimeConfig(provider="lmstudio", model="qwen/qwen3-next-80b"),
    )

    agent = ReactAgent(runtime=rt, tools=[])
    run_id = agent.start("t")

    state = run_store.load(run_id)
    assert state is not None
    runtime_ns = state.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    assert runtime_ns.get("provider") == "lmstudio"
    assert runtime_ns.get("model") == "qwen/qwen3-next-80b"


def test_codeact_agent_start_seeds_runtime_provider_and_model_from_runtime_config() -> None:
    run_store = InMemoryRunStore()
    ledger_store = InMemoryLedgerStore()
    rt = Runtime(
        run_store=run_store,
        ledger_store=ledger_store,
        config=RuntimeConfig(provider="anthropic", model="claude-haiku-4-5"),
    )

    agent = CodeActAgent(runtime=rt, tools=[])
    run_id = agent.start("t")

    state = run_store.load(run_id)
    assert state is not None
    runtime_ns = state.vars.get("_runtime")
    assert isinstance(runtime_ns, dict)
    assert runtime_ns.get("provider") == "anthropic"
    assert runtime_ns.get("model") == "claude-haiku-4-5"


