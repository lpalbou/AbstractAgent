"""MemAct agent implementation (memory-enhanced).

MemAct is the only agent that uses `abstractruntime.memory.active_memory`.
ReAct and CodeAct remain conventional SOTA agents.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolDefinition
from abstractruntime import RunState, RunStatus, Runtime, WorkflowSpec

from .base import BaseAgent
from ..adapters.loop_hooks import LoopHooks
from ..adapters.memact_runtime import create_memact_workflow
from ..logic.builtins import (
    ASK_USER_TOOL,
    COMPACT_MEMORY_TOOL,
    DELEGATE_AGENT_TOOL,
    INSPECT_VARS_TOOL,
    OPEN_ATTACHMENT_TOOL,
    RECALL_MEMORY_TOOL,
    REMEMBER_TOOL,
    REMEMBER_NOTE_TOOL,
)
from ..logic.memact import MemActLogic


def _tool_definitions_from_callables(tools: List[Callable[..., Any]]) -> List[ToolDefinition]:
    tool_defs: List[ToolDefinition] = []
    for t in tools:
        tool_def = getattr(t, "_tool_definition", None)
        if tool_def is None:
            tool_def = ToolDefinition.from_function(t)
        tool_defs.append(tool_def)
    return tool_defs


def _copy_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            out.append(dict(m))
    return out


def _deepcopy_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return value


class MemActAgent(BaseAgent):
    """Memory-enhanced agent with runtime-owned Active Memory blocks."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        tools: Optional[List[Callable[..., Any]]] = None,
        on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        hooks: Optional[LoopHooks] = None,
        max_iterations: int = 20,
        max_history_messages: int = -1,
        # max_tokens = the context ACCOUNTING ceiling (drives warn_tokens_pct /
        # context_warning), NOT an output cap. For an output-token cap use
        # max_output_tokens (fable5 B-F9 honesty split, 2026-07-13).
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        # DEPRECATED-IGNORED (backlog 0013 / C4): MemAct never had plan or
        # review nodes, so these were accepted-and-ignored — the silent
        # fall-open class. The 0013 hard removal (TypeError) broke shipped
        # abstractcode agent-switch call sites (regression adversary P0,
        # 2026-07-13), so a one-release shim warns loudly on non-default
        # values and ignores the legacy defaults. Removal lands next release;
        # MemAct verifier parity stays a deliberate non-goal.
        plan_mode: Optional[bool] = None,
        review_mode: Optional[bool] = None,
        review_max_rounds: Optional[int] = None,
    ):
        if plan_mode or review_mode or (review_max_rounds is not None and int(review_max_rounds) != 3):
            import warnings

            warnings.warn(
                "MemActAgent has no plan/review nodes — plan_mode/review_mode/review_max_rounds "
                "are deprecated-ignored (no-ops; removal next release)",
                DeprecationWarning,
                stacklevel=2,
            )
        self.hooks = hooks
        self._max_iterations = int(max_iterations)
        if self._max_iterations < 1:
            self._max_iterations = 1
        self._max_history_messages = int(max_history_messages)
        if self._max_history_messages != -1 and self._max_history_messages < 1:
            self._max_history_messages = 1
        self._max_tokens = max_tokens
        self._max_output_tokens = max_output_tokens

        self.logic: Optional[MemActLogic] = None
        self.session_active_memory: Optional[Dict[str, Any]] = None
        super().__init__(
            runtime=runtime,
            tools=tools,
            on_step=on_step,
            actor_id=actor_id,
            session_id=session_id,
        )

    def _create_workflow(self) -> WorkflowSpec:
        tool_defs = _tool_definitions_from_callables(self.tools)
        tool_defs = [
            ASK_USER_TOOL,
            OPEN_ATTACHMENT_TOOL,
            RECALL_MEMORY_TOOL,
            INSPECT_VARS_TOOL,
            REMEMBER_TOOL,
            REMEMBER_NOTE_TOOL,
            COMPACT_MEMORY_TOOL,
            DELEGATE_AGENT_TOOL,
            *tool_defs,
        ]
        logic = MemActLogic(tools=tool_defs)
        self.logic = logic
        return create_memact_workflow(logic=logic, on_step=self.on_step, hooks=self.hooks)

    def _sync_session_caches_from_state(self, state: Optional[RunState]) -> None:
        super()._sync_session_caches_from_state(state)
        if state is None or not hasattr(state, "vars") or not isinstance(state.vars, dict):
            return
        runtime_ns = state.vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            return
        mem = runtime_ns.get("active_memory")
        if isinstance(mem, dict):
            self.session_active_memory = _deepcopy_json(mem)

    def start(
        self,
        task: str,
        *,
        allowed_tools: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        attachments: Optional[List[Any]] = None,
        # Named system-prompt slots (docs/skills-attachment.md); byte-stable
        # for the run (cache prefix).
        skills_block: Optional[str] = None,
        system_prompt_extra: Optional[str] = None,
    ) -> str:
        task = str(task or "").strip()
        if not task:
            raise ValueError("task must be a non-empty string")

        try:
            base_limits = dict(self.runtime.config.to_limits_dict())
        except Exception:
            base_limits = {}
        limits: Dict[str, Any] = dict(base_limits)
        limits.setdefault("warn_iterations_pct", 80)
        limits.setdefault("warn_tokens_pct", 80)
        limits["max_iterations"] = int(self._max_iterations)
        limits["current_iteration"] = 0
        limits["max_history_messages"] = int(self._max_history_messages)
        limits["estimated_tokens_used"] = 0
        try:
            max_tokens_override = int(self._max_tokens) if self._max_tokens is not None else None
        except Exception:
            max_tokens_override = None
        if isinstance(max_tokens_override, int) and max_tokens_override > 0:
            limits["max_tokens"] = max_tokens_override
        # No facade-level ceiling fallback (operator ruling 2026-07-13,
        # "overengineering"): the accounting ceiling is the RUNTIME'S to
        # default (config -> model registry -> DEFAULT_MAX_TOKENS inside
        # to_limits_dict, and again at its own read sites). A facade literal
        # was a third copy that fired only when the runtime lookup failed —
        # fabricating a warning threshold unrelated to any real window in
        # exactly the case where nothing is known. Missing ceiling = the
        # context_warning stays silent, which is the honest signal.
        # Output cap: the honest name (max_tokens above is accounting-only).
        try:
            out_cap = int(self._max_output_tokens) if self._max_output_tokens is not None else None
        except Exception:
            out_cap = None
        if isinstance(out_cap, int) and out_cap > 0:
            limits["max_output_tokens"] = out_cap

        runtime_ns: Dict[str, Any] = {
            "inbox": [],
        }
        if isinstance(skills_block, str) and skills_block.strip():
            runtime_ns["skills_block"] = skills_block
        if isinstance(system_prompt_extra, str) and system_prompt_extra.strip():
            runtime_ns["system_prompt_extra"] = system_prompt_extra
        if temperature is not None:
            try:
                runtime_ns["temperature"] = float(temperature)
            except Exception:
                pass
        if seed is not None:
            try:
                runtime_ns["seed"] = int(seed)
            except Exception:
                pass
        if isinstance(self.session_active_memory, dict):
            runtime_ns["active_memory"] = _deepcopy_json(self.session_active_memory)
        if isinstance(allowed_tools, list):
            normalized = [str(t).strip() for t in allowed_tools if isinstance(t, str) and t.strip()]
            runtime_ns["allowed_tools"] = normalized

        vars: Dict[str, Any] = {
            "context": {"task": task, "messages": _copy_messages(self.session_messages)},
            "scratchpad": {"iteration": 0, "max_iterations": int(self._max_iterations)},
            "_runtime": runtime_ns,
            "_temp": {},
            "_limits": limits,
        }
        if attachments:
            items: list[Any]
            if isinstance(attachments, tuple):
                items = list(attachments)
            else:
                items = attachments if isinstance(attachments, list) else []
            normalized: list[Any] = []
            for item in items:
                if isinstance(item, str) and item.strip():
                    normalized.append(item.strip())
                    continue
                if isinstance(item, dict):
                    aid = item.get("$artifact")
                    if not (isinstance(aid, str) and aid.strip()):
                        aid = item.get("artifact_id")
                    if isinstance(aid, str) and aid.strip():
                        normalized.append(dict(item))
            if normalized:
                vars["context"]["attachments"] = normalized

        run_id = self.runtime.start(
            workflow=self.workflow,
            vars=vars,
            actor_id=self._ensure_actor_id(),
            session_id=self._ensure_session_id(),
        )
        self._current_run_id = run_id
        return run_id

    def step(self) -> RunState:
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start() first.")
        state = self.runtime.tick(workflow=self.workflow, run_id=self._current_run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            self._sync_session_caches_from_state(state)
        return state


def create_memact_agent(
    *,
    # None = resolve from AbstractCore config global defaults (set via
    # `abstractcore --config`); packaged fallback pair applies with a loud
    # #FALLBACK warning when nothing is configured (B-F8, 2026-07-13).
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[List[Callable[..., Any]]] = None,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    max_iterations: int = 20,
    max_history_messages: int = -1,
    max_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    run_store: Optional[Any] = None,
    ledger_store: Optional[Any] = None,
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> MemActAgent:
    """Factory: create a MemActAgent with a local AbstractCore-backed runtime."""

    from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime

    if tools is None:
        from ..tools import ALL_TOOLS

        tools = list(ALL_TOOLS)

    from .defaults import resolve_provider_model

    provider, model = resolve_provider_model(provider, model)

    runtime = create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        tool_executor=MappingToolExecutor.from_tools(list(tools)),
        run_store=run_store,
        ledger_store=ledger_store,
    )
    return MemActAgent(
        runtime=runtime,
        tools=tools,
        on_step=on_step,
        max_iterations=max_iterations,
        max_history_messages=max_history_messages,
        max_tokens=max_tokens,
        max_output_tokens=max_output_tokens,
        actor_id=actor_id,
        session_id=session_id,
    )
