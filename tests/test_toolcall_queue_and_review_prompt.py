from __future__ import annotations

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import EffectType
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


def _run(*, vars: dict, current_node: str) -> RunState:
    return RunState(
        run_id="run",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node=current_node,
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


def test_react_review_prompt_includes_user_responses_and_ask_prompts() -> None:
    logic = ReActLogic(
        tools=[
            ToolDefinition(name="search_files", description="search", parameters={}),
            ToolDefinition(name="ask_user", description="ask", parameters={}),
        ]
    )
    wf = create_react_workflow(logic=logic, workflow_id="wf")

    run = _run(
        current_node="review",
        vars={
            "context": {
                "task": "t",
                "messages": [
                    {"role": "user", "content": "t"},
                    {"role": "assistant", "content": "[Agent question]: Did it freeze?", "metadata": {"kind": "ask_user_prompt"}},
                    {"role": "user", "content": "[User response]: the game was frozen"},
                    {"role": "tool", "content": "observation[search_files] (success): ..."},
                ],
            },
            "scratchpad": {"review_count": 0},
            "_runtime": {"inbox": [], "review_mode": True, "allowed_tools": ["search_files", "ask_user"]},
            "_temp": {"final_answer": "candidate"},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("review")(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    prompt = str(payload.get("prompt") or "")
    assert "the game was frozen" in prompt
    assert "Agent question" in prompt or "ask_user prompts" in prompt


def test_react_toolcall_queue_executes_tools_before_ask_user_and_continues() -> None:
    logic = ReActLogic(
        tools=[
            ToolDefinition(name="search_files", description="search", parameters={}),
            ToolDefinition(name="ask_user", description="ask", parameters={}),
        ]
    )
    wf = create_react_workflow(logic=logic, workflow_id="wf")

    run = _run(
        current_node="act",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"review_count": 0, "used_tools": False},
            "_runtime": {"inbox": [], "allowed_tools": ["search_files", "ask_user"]},
            "_temp": {
                "pending_tool_calls": [
                    {"name": "search_files", "arguments": {"pattern": "x", "path": "."}},
                    {"name": "ask_user", "arguments": {"question": "Did it freeze?"}},
                ]
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    # First act executes normal tools batch and leaves ask_user queued.
    step1 = wf.get_node("act")(run, _Ctx())
    assert step1.effect is not None
    assert step1.effect.type == EffectType.TOOL_CALLS
    assert step1.next_node == "observe"
    assert run.vars["_temp"]["pending_tool_calls"] and run.vars["_temp"]["pending_tool_calls"][0]["name"] == "ask_user"

    # After observe, we should continue to act (not reason) because ask_user remains queued.
    run.vars["_temp"]["tool_results"] = {"results": [{"name": "search_files", "success": True, "output": "ok", "call_id": "1"}]}
    step2 = wf.get_node("observe")(run, _Ctx())
    assert step2.next_node == "act"

    # Next act executes ask_user.
    step3 = wf.get_node("act")(run, _Ctx())
    assert step3.effect is not None
    assert step3.effect.type == EffectType.ASK_USER
    assert step3.next_node == "handle_user_response"

    # The asked question should be persisted as an assistant message for context/review.
    msgs = run.vars["context"]["messages"]
    assert any(
        isinstance(m, dict)
        and m.get("role") == "assistant"
        and isinstance(m.get("metadata"), dict)
        and m["metadata"].get("kind") == "ask_user_prompt"
        for m in msgs
    )


def test_react_empty_llm_response_triggers_recovery_retry() -> None:
    logic = ReActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})])
    wf = create_react_workflow(logic=logic, workflow_id="wf")

    run = _run(
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"iteration": 0, "max_iterations": 2},
            "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
            "_temp": {"llm_response": {"content": "", "tool_calls": []}},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "empty_response_retry"


def test_react_review_parse_retries_when_incomplete_without_tool_calls() -> None:
    logic = ReActLogic(tools=[ToolDefinition(name="read_file", description="read", parameters={})])
    wf = create_react_workflow(logic=logic, workflow_id="wf")

    run = _run(
        current_node="review_parse",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 3, "allowed_tools": ["read_file"]},
            "_temp": {
                "final_answer": "candidate",
                "review_llm_response": {"data": {"complete": False, "missing": ["x"], "next_prompt": "Do something", "next_tool_calls": []}},
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("review_parse")(run, _Ctx())
    assert plan.next_node == "review"
    inbox = run.vars["_runtime"].get("inbox")
    assert isinstance(inbox, list) and inbox


def test_codeact_review_prompt_includes_user_responses_and_ask_prompts() -> None:
    logic = CodeActLogic(
        tools=[
            ToolDefinition(name="edit_file", description="edit", parameters={}),
            ToolDefinition(name="ask_user", description="ask", parameters={}),
        ]
    )
    wf = create_codeact_workflow(logic=logic, on_step=None)

    run = _run(
        current_node="review",
        vars={
            "context": {
                "task": "t",
                "messages": [
                    {"role": "user", "content": "t"},
                    {"role": "assistant", "content": "[Agent question]: Did it freeze?", "metadata": {"kind": "ask_user_prompt"}},
                    {"role": "user", "content": "[User response]: the game was frozen"},
                    {"role": "tool", "content": "observation[edit_file] (success): ..."},
                ],
            },
            "scratchpad": {"review_count": 0},
            "_runtime": {"inbox": [], "review_mode": True, "allowed_tools": ["edit_file", "ask_user"]},
            "_temp": {"final_answer": "candidate"},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("review")(run, _Ctx())
    assert plan.effect is not None
    assert plan.effect.type == EffectType.LLM_CALL
    payload = plan.effect.payload if isinstance(plan.effect.payload, dict) else {}
    prompt = str(payload.get("prompt") or "")
    assert "the game was frozen" in prompt


def test_codeact_toolcall_queue_executes_tools_before_ask_user_and_continues() -> None:
    logic = CodeActLogic(
        tools=[
            ToolDefinition(name="edit_file", description="edit", parameters={}),
            ToolDefinition(name="ask_user", description="ask", parameters={}),
        ]
    )
    wf = create_codeact_workflow(logic=logic, on_step=None)

    run = _run(
        current_node="act",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"review_count": 0},
            "_runtime": {"inbox": [], "allowed_tools": ["edit_file", "ask_user"]},
            "_temp": {
                "pending_tool_calls": [
                    {"name": "edit_file", "arguments": {"file_path": "x", "pattern": "a", "replacement": "b"}},
                    {"name": "ask_user", "arguments": {"question": "Did it freeze?"}},
                ]
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    step1 = wf.get_node("act")(run, _Ctx())
    assert step1.effect is not None
    assert step1.effect.type == EffectType.TOOL_CALLS
    assert run.vars["_temp"]["pending_tool_calls"] and run.vars["_temp"]["pending_tool_calls"][0]["name"] == "ask_user"

    run.vars["_temp"]["tool_results"] = {"results": [{"name": "edit_file", "success": True, "output": "ok", "call_id": "1"}]}
    step2 = wf.get_node("observe")(run, _Ctx())
    assert step2.next_node == "act"

    step3 = wf.get_node("act")(run, _Ctx())
    assert step3.effect is not None
    assert step3.effect.type == EffectType.ASK_USER


def test_codeact_empty_llm_response_triggers_recovery_retry() -> None:
    logic = CodeActLogic(tools=[ToolDefinition(name="edit_file", description="edit", parameters={})])
    wf = create_codeact_workflow(logic=logic, on_step=None)

    run = _run(
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"iteration": 0, "max_iterations": 2},
            "_runtime": {"inbox": [], "allowed_tools": ["edit_file"]},
            "_temp": {"llm_response": {"content": "", "tool_calls": []}},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"].get("inbox")
    assert isinstance(inbox, list) and inbox


def test_codeact_review_parse_retries_when_incomplete_without_tool_calls() -> None:
    logic = CodeActLogic(tools=[ToolDefinition(name="edit_file", description="edit", parameters={})])
    wf = create_codeact_workflow(logic=logic, on_step=None)

    run = _run(
        current_node="review_parse",
        vars={
            "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 3, "allowed_tools": ["edit_file"]},
            "_temp": {
                "final_answer": "candidate",
                "review_llm_response": {"data": {"complete": False, "missing": ["x"], "next_prompt": "Do something", "next_tool_calls": []}},
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("review_parse")(run, _Ctx())
    assert plan.next_node == "review"
    inbox = run.vars["_runtime"].get("inbox")
    assert isinstance(inbox, list) and inbox


