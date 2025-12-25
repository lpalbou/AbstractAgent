"""AbstractRuntime adapter for ReAct-like agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy

from ..logic.react import ReActLogic


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

    import uuid

    meta = dict(metadata or {})
    meta.setdefault("message_id", f"msg_{uuid.uuid4().hex}")

    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "metadata": meta,
    }


def ensure_react_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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

    for key in ("llm_response", "tool_results", "pending_tool_calls", "user_response", "final_answer"):
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

    # Track whether any external tools were actually executed during this run.
    # This is used to reliably trigger a final "synthesis" pass so the agent
    # returns a user-facing answer instead of echoing tool observations.
    used_tools = scratchpad.get("used_tools")
    if not isinstance(used_tools, bool):
        scratchpad["used_tools"] = bool(used_tools) if used_tools is not None else False

    return context, scratchpad, runtime_ns, temp, limits


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def create_react_workflow(
    *,
    logic: ReActLogic,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    workflow_id: str = "react_agent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> WorkflowSpec:
    """Adapt ReActLogic to an AbstractRuntime workflow."""

    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)

    tool_defs = logic.tools
    tool_specs = [t.to_dict() for t in tool_defs]
    toolset_id = _compute_toolset_id(tool_specs)
    allowlist: Optional[list[str]] = None
    if isinstance(allowed_tools, list):
        allowlist = [str(t).strip() for t in allowed_tools if isinstance(t, str) and t.strip()]
        if not allowlist:
            allowlist = []
    else:
        # Default allowlist: the tools the logic provided (defense-in-depth vs executor having extra tools).
        allowlist = [str(t.name) for t in tool_defs if getattr(t, "name", None)]

    def init_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_react_vars(run)
        scratchpad["iteration"] = 0
        limits["current_iteration"] = 0

        task = str(context.get("task", "") or "")
        context["task"] = task
        messages = context["messages"]

        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append(_new_message(ctx, role="user", content=task))

        # Ensure toolset metadata is present for audit/debug.
        runtime_ns.setdefault("tool_specs", tool_specs)
        runtime_ns.setdefault("toolset_id", toolset_id)
        runtime_ns.setdefault("allowed_tools", list(allowlist or []))
        runtime_ns.setdefault("inbox", [])

        emit("init", {"task": task})
        return StepPlan(node_id="init", next_node="reason")

    def reason_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_react_vars(run)

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

        task = str(context.get("task", "") or "")
        messages_view = ActiveContextPolicy.select_active_messages_for_llm_from_run(run)

        inbox = runtime_ns.get("inbox", [])
        guidance = ""
        if isinstance(inbox, list) and inbox:
            inbox_messages = [str(m.get("content", "") or "") for m in inbox if isinstance(m, dict)]
            guidance = " | ".join([m for m in inbox_messages if m])
            runtime_ns["inbox"] = []

        req = logic.build_request(
            task=task,
            messages=messages_view,
            guidance=guidance,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,  # Pass vars for _limits access
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        payload = {"prompt": req.prompt}
        tools_payload = [t.to_dict() for t in req.tools]
        if tools_payload:
            payload["tools"] = tools_payload
        # Provider/model can be configured statically (create_react_workflow args)
        # or injected dynamically through durable vars in `_runtime` (Visual Agent pins).
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()
        params: Dict[str, Any] = {}
        if req.max_tokens is not None:
            params["max_tokens"] = req.max_tokens
        # Tool calling is formatting-sensitive; bias toward deterministic output when tools are present.
        params["temperature"] = 0.2 if tools_payload else 0.7
        payload["params"] = params

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
        context, scratchpad, _, temp, _ = ensure_react_vars(run)
        response = temp.get("llm_response", {})
        content, tool_calls = logic.parse_response(response)

        context["messages"].append(_new_message(ctx, role="assistant", content=content))

        emit(
            "parse",
            {
                "has_tool_calls": bool(tool_calls),
                "content_preview": content[:100] if content else "(no content)",
            },
        )
        temp.pop("llm_response", None)

        if tool_calls:
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            return StepPlan(node_id="parse", next_node="act")

        temp["final_answer"] = content
        # If we used tools, always run a final "synthesis" LLM pass to produce a
        # clean user-facing answer (models may otherwise echo tool transcript lines).
        if bool(scratchpad.get("used_tools")):
            return StepPlan(node_id="parse", next_node="finalize")
        return StepPlan(node_id="parse", next_node="done")

    def act_node(run: RunState, ctx) -> StepPlan:
        _, _, _, temp, _ = ensure_react_vars(run)
        tool_calls = temp.get("pending_tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        if not tool_calls:
            return StepPlan(node_id="act", next_node="reason")

        builtin_effect_tools = {"ask_user", "recall_memory", "remember", "compact_memory"}

        # Handle schema-only built-ins specially (ASK_USER, MEMORY_QUERY).
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            args = tc.get("arguments") or {}

            if isinstance(name, str) and name in builtin_effect_tools:
                if isinstance(allowlist, list) and name not in allowlist:
                    temp["pending_tool_calls"] = tool_calls[i + 1 :]
                    temp["tool_results"] = {
                        "results": [
                            {
                                "call_id": str(tc.get("call_id") or ""),
                                "name": name,
                                "success": False,
                                "output": None,
                                "error": f"Tool '{name}' is not allowed for this agent",
                            }
                        ]
                    }
                    emit("act_blocked", {"tool": name})
                    return StepPlan(node_id="act", next_node="observe")

            if name == "ask_user":
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

            if name == "recall_memory":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "recall_memory")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_query", {"query": payload.get("query"), "span_id": payload.get("span_id")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_QUERY,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "remember":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "remember")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_tag", {"span_id": payload.get("span_id"), "tags": payload.get("tags")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_TAG,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "compact_memory":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "compact_memory")
                payload.setdefault("call_id", tc.get("call_id") or "compact")
                emit(
                    "memory_compact",
                    {"preserve_recent": payload.get("preserve_recent"), "mode": payload.get("compression_mode"), "focus": payload.get("focus")},
                )
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_COMPACT,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

        for tc in tool_calls:
            if isinstance(tc, dict):
                emit("act", {"tool": tc.get("name", ""), "args": tc.get("arguments", {})})

        formatted_calls: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                formatted_calls.append(
                    {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                        "call_id": tc.get("call_id", "1"),
                    }
                )
            elif isinstance(tc, ToolCall):
                formatted_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "call_id": tc.call_id or "1",
                    }
                )

        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls, "allowed_tools": list(allowlist or [])},
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def observe_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_react_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []
        if results:
            scratchpad["used_tools"] = True

        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "tool") or "tool")
            success = bool(r.get("success"))
            output = r.get("output", "")
            error = r.get("error", "")
            rendered = logic.format_observation(
                name=name,
                output=str(output if success else (error or output)),
                success=success,
            )
            emit("observe", {"tool": name, "result": rendered[:150]})
            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={
                        "name": name,
                        "call_id": r.get("call_id"),
                        "success": success,
                    },
                )
            )

        temp.pop("tool_results", None)
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def finalize_node(run: RunState, ctx) -> StepPlan:
        """Final synthesis pass to ensure we return a user-facing answer.

        This is intentionally tool-free: tools have already been executed, and we
        want a single clean response that uses the observations.
        """
        context, _, _, _, _ = ensure_react_vars(run)
        task = str(context.get("task", "") or "")
        # NOTE: We intentionally use a prompt-only synthesis request (instead of
        # passing message dicts) because host messages can contain extra metadata
        # fields that some providers/models don't accept.
        #
        # We also keep the prompt small: include only the most recent tool outputs.
        messages = list(context.get("messages") or [])
        tool_msgs: list[str] = []
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            tool_msgs.append(content.strip())
            if len(tool_msgs) >= 6:
                break
        tool_msgs.reverse()

        obs_blocks: list[str] = []
        for t in tool_msgs:
            # Prevent prompt bloat from huge file reads.
            obs_blocks.append(t if len(t) <= 2500 else (t[:2500] + "…"))

        observations = "\n\n".join(obs_blocks) if obs_blocks else "(no tool observations captured)"
        prompt = (
            "Write the final user-facing answer.\n\n"
            f"Task:\n{task}\n\n"
            f"Observations (tool outputs):\n{observations}\n\n"
            "Answer:\n"
        )

        emit("finalize", {"tool_messages": len(tool_msgs)})

        payload: Dict[str, Any] = {"prompt": prompt, "params": {"temperature": 0.2}}
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        return StepPlan(
            node_id="finalize",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.final_llm_response",
            ),
            next_node="finalize_parse",
        )

    def finalize_parse_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_react_vars(run)
        resp = temp.get("final_llm_response", {})
        if not isinstance(resp, dict):
            resp = {}
        content = resp.get("content")
        answer = "" if content is None else str(content)
        emit("finalize_parse", {"content_preview": answer[:100] if answer else "(empty)"})
        temp["final_answer"] = answer
        temp.pop("final_llm_response", None)
        return StepPlan(node_id="finalize_parse", next_node="done")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_react_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(
            _new_message(ctx, role="user", content=f"[User response]: {response_text}")
        )
        temp.pop("user_response", None)

        if temp.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="reason")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, limits = ensure_react_vars(run)
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
        context, scratchpad, _, _, limits = ensure_react_vars(run)

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
        workflow_id=str(workflow_id or "react_agent"),
        entry_node="init",
        nodes={
            "init": init_node,
            "reason": reason_node,
            "parse": parse_node,
            "act": act_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "finalize": finalize_node,
            "finalize_parse": finalize_parse_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )
