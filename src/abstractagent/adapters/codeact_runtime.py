"""AbstractRuntime adapter for CodeAct agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces

from ..logic.codeact import CodeActLogic


def _new_message(
    ctx: Any,
    *,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    timestamp: Optional[str] = None
    now_iso = getattr(ctx, "now_iso", None)
    if callable(now_iso):
        timestamp = str(now_iso())
    if not timestamp:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "metadata": metadata or {},
    }


def ensure_codeact_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Ensure namespaced vars exist and migrate legacy flat keys in-place.

    Returns:
        Tuple of (context, scratchpad, runtime_ns, temp, limits) dicts.
    """
    ensure_namespaces(run.vars)
    limits = ensure_limits(run.vars)
    context = run.vars["context"]
    scratchpad = run.vars["scratchpad"]
    runtime_ns = run.vars["_runtime"]
    temp = run.vars["_temp"]

    if "task" in run.vars and "task" not in context:
        context["task"] = run.vars.pop("task")
    if "messages" in run.vars and "messages" not in context:
        context["messages"] = run.vars.pop("messages")
    if "iteration" in run.vars and "iteration" not in scratchpad:
        scratchpad["iteration"] = run.vars.pop("iteration")
    if "max_iterations" in run.vars and "max_iterations" not in scratchpad:
        scratchpad["max_iterations"] = run.vars.pop("max_iterations")
    if "_inbox" in run.vars and "inbox" not in runtime_ns:
        runtime_ns["inbox"] = run.vars.pop("_inbox")

    for key in ("llm_response", "tool_results", "pending_tool_calls", "user_response", "final_answer", "pending_code"):
        if key in run.vars and key not in temp:
            temp[key] = run.vars.pop(key)

    if not isinstance(context.get("messages"), list):
        context["messages"] = []
    if not isinstance(runtime_ns.get("inbox"), list):
        runtime_ns["inbox"] = []

    iteration = scratchpad.get("iteration")
    if not isinstance(iteration, int):
        try:
            scratchpad["iteration"] = int(iteration or 0)
        except (TypeError, ValueError):
            scratchpad["iteration"] = 0

    max_iterations = scratchpad.get("max_iterations")
    if max_iterations is None:
        scratchpad["max_iterations"] = 25
    elif not isinstance(max_iterations, int):
        try:
            scratchpad["max_iterations"] = int(max_iterations)
        except (TypeError, ValueError):
            scratchpad["max_iterations"] = 25

    if scratchpad["max_iterations"] < 1:
        scratchpad["max_iterations"] = 1

    return context, scratchpad, runtime_ns, temp, limits


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def create_codeact_workflow(
    *,
    logic: CodeActLogic,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> WorkflowSpec:
    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)

    tool_defs = logic.tools
    tool_specs = [t.to_dict() for t in tool_defs]
    toolset_id = _compute_toolset_id(tool_specs)

    def init_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_codeact_vars(run)
        scratchpad["iteration"] = 0
        limits["current_iteration"] = 0

        task = str(context.get("task", "") or "")
        context["task"] = task
        messages = context["messages"]
        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append(_new_message(ctx, role="user", content=task))

        runtime_ns.setdefault("tool_specs", tool_specs)
        runtime_ns.setdefault("toolset_id", toolset_id)
        runtime_ns.setdefault("inbox", [])

        emit("init", {"task": task})
        return StepPlan(node_id="init", next_node="reason")

    def reason_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_codeact_vars(run)

        # Read from _limits (canonical) with fallback to scratchpad (backward compat)
        if "current_iteration" in limits:
            iteration = int(limits.get("current_iteration", 0) or 0)
            max_iterations = int(limits.get("max_iterations", 25) or 25)
        else:
            # Backward compatibility: use scratchpad
            iteration = int(scratchpad.get("iteration", 0) or 0)
            max_iterations = int(scratchpad.get("max_iterations") or 25)

        if max_iterations < 1:
            max_iterations = 1

        if iteration >= max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")

        # Update both for transition period
        scratchpad["iteration"] = iteration + 1
        limits["current_iteration"] = iteration + 1

        inbox = runtime_ns.get("inbox", [])
        guidance = ""
        if isinstance(inbox, list) and inbox:
            inbox_messages = [str(m.get("content", "") or "") for m in inbox if isinstance(m, dict)]
            guidance = " | ".join([m for m in inbox_messages if m])
            runtime_ns["inbox"] = []

        req = logic.build_request(
            task=str(context.get("task", "") or ""),
            messages=list(context.get("messages") or []),
            guidance=guidance,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,  # Pass vars for _limits access
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        payload = {"prompt": req.prompt, "tools": [t.to_dict() for t in req.tools]}
        if req.max_tokens is not None:
            payload["params"] = {"max_tokens": req.max_tokens}

        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.llm_response",
            ),
            next_node="parse",
        )

    def parse_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_codeact_vars(run)
        response = temp.get("llm_response", {})
        content, tool_calls = logic.parse_response(response)

        if content:
            context["messages"].append(_new_message(ctx, role="assistant", content=content))

        temp.pop("llm_response", None)
        emit("parse", {"has_tool_calls": bool(tool_calls), "content_preview": (content[:100] if content else "(no content)")})

        if tool_calls:
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            return StepPlan(node_id="parse", next_node="act")

        code = logic.extract_code(content)
        if code:
            temp["pending_code"] = code
            return StepPlan(node_id="parse", next_node="execute_code")

        temp["final_answer"] = content
        return StepPlan(node_id="parse", next_node="done")

    def act_node(run: RunState, ctx) -> StepPlan:
        _, _, _, temp, _ = ensure_codeact_vars(run)
        tool_calls = temp.get("pending_tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        if not tool_calls:
            return StepPlan(node_id="act", next_node="reason")

        # Handle ask_user specially with ASK_USER effect.
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            if tc.get("name") != "ask_user":
                continue
            args = tc.get("arguments") or {}
            question = str(args.get("question") or "Please provide input:")
            choices = args.get("choices")
            choices = list(choices) if isinstance(choices, list) else None

            temp["pending_tool_calls"] = tool_calls[i + 1 :]
            emit("ask_user", {"question": question, "choices": choices or []})
            return StepPlan(
                node_id="act",
                effect=Effect(
                    type=EffectType.ASK_USER,
                    payload={"prompt": question, "choices": choices, "allow_free_text": True},
                    result_key="_temp.user_response",
                ),
                next_node="handle_user_response",
            )

        for tc in tool_calls:
            if isinstance(tc, dict):
                emit("act", {"tool": tc.get("name", ""), "args": tc.get("arguments", {})})

        formatted_calls: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                formatted_calls.append(
                    {"name": tc.get("name", ""), "arguments": tc.get("arguments", {}), "call_id": tc.get("call_id", "1")}
                )
            elif isinstance(tc, ToolCall):
                formatted_calls.append(
                    {"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id or "1"}
                )

        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls},
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def execute_code_node(run: RunState, ctx) -> StepPlan:
        _, _, _, temp, _ = ensure_codeact_vars(run)
        code = temp.get("pending_code")
        if not isinstance(code, str) or not code.strip():
            return StepPlan(node_id="execute_code", next_node="reason")

        temp.pop("pending_code", None)
        emit("act", {"tool": "execute_python", "args": {"code": "(inline)", "timeout_s": 10.0}})

        return StepPlan(
            node_id="execute_code",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={
                    "tool_calls": [
                        {
                            "name": "execute_python",
                            "arguments": {"code": code, "timeout_s": 10.0},
                            "call_id": "code",
                        }
                    ]
                },
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def observe_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_codeact_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []

        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "tool") or "tool")
            success = bool(r.get("success"))
            output = r.get("output", "")
            error = r.get("error", "")
            rendered = logic.format_observation(
                name=name,
                output=(output if success else (error or output)),
                success=success,
            )
            emit("observe", {"tool": name, "result": rendered[:150]})
            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={"name": name, "call_id": r.get("call_id"), "success": success},
                )
            )

        temp.pop("tool_results", None)
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_codeact_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(_new_message(ctx, role="user", content=f"[User response]: {response_text}"))
        temp.pop("user_response", None)

        if temp.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="reason")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, limits = ensure_codeact_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer})

        # Prefer _limits.current_iteration, fall back to scratchpad
        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or 0)

        return StepPlan(
            node_id="done",
            complete_output={
                "answer": answer,
                "iterations": iterations,
                "messages": list(context.get("messages") or []),
            },
        )

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, _, limits = ensure_codeact_vars(run)

        # Prefer _limits, fall back to scratchpad
        max_iterations = int(limits.get("max_iterations", 0) or scratchpad.get("max_iterations", 25) or 25)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations})

        messages = list(context.get("messages") or [])
        last_content = messages[-1]["content"] if messages else "Max iterations reached"
        return StepPlan(
            node_id="max_iterations",
            complete_output={
                "answer": last_content,
                "iterations": max_iterations,
                "messages": messages,
            },
        )

    return WorkflowSpec(
        workflow_id="codeact_agent",
        entry_node="init",
        nodes={
            "init": init_node,
            "reason": reason_node,
            "parse": parse_node,
            "act": act_node,
            "execute_code": execute_code_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )

