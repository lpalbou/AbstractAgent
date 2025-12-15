from abstractruntime import InMemoryLedgerStore, InMemoryRunStore, RunStatus, Runtime

from abstractagent.agents.react import ReactAgent


def test_react_agent_seeds_messages_from_session():
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
    )

    agent = ReactAgent(runtime=runtime, tools=[])
    agent.session_messages = [
        {"role": "user", "content": "Investigate abstractcore/"},
        {"role": "assistant", "content": "That path does not exist. Did you mean abstractcode/?"},
    ]

    run_id = agent.start("I meant abstractcode/")

    # Execute only the init node so we can inspect message seeding without invoking any LLM handler.
    state = runtime.tick(workflow=agent.workflow, run_id=run_id, max_steps=1)

    assert state.status == RunStatus.RUNNING
    assert state.current_node == "reason"
    assert isinstance(state.vars.get("context"), dict)
    assert isinstance(state.vars["context"].get("messages"), list)
    last = state.vars["context"]["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == "I meant abstractcode/"
    assert isinstance(last.get("timestamp"), str)
    assert last["timestamp"]
    assert isinstance(last.get("metadata"), dict)
