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
from ..adapters.memact_runtime import create_memact_workflow
from ..logic.builtins import (
    ASK_USER_TOOL,
    COMPACT_MEMORY_TOOL,
    INSPECT_VARS_TOOL,
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
        max_iterations: int = 25,
        max_history_messages: int = -1,
        max_tokens: Optional[int] = None,
        plan_mode: bool = False,
        review_mode: bool = False,
        review_max_rounds: int = 1,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self._max_iterations = int(max_iterations)
        if self._max_iterations < 1:
            self._max_iterations = 1
        self._max_history_messages = int(max_history_messages)
        if self._max_history_messages != -1 and self._max_history_messages < 1:
            self._max_history_messages = 1
        self._max_tokens = max_tokens
        self._plan_mode = bool(plan_mode)
        self._review_mode = bool(review_mode)
        self._review_max_rounds = int(review_max_rounds)
        if self._review_max_rounds < 0:
            self._review_max_rounds = 0

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
            RECALL_MEMORY_TOOL,
            INSPECT_VARS_TOOL,
            REMEMBER_TOOL,
            REMEMBER_NOTE_TOOL,
            COMPACT_MEMORY_TOOL,
            *tool_defs,
        ]
        logic = MemActLogic(
            tools=tool_defs,
            max_history_messages=self._max_history_messages,
            max_tokens=self._max_tokens,
        )
        self.logic = logic
        return create_memact_workflow(logic=logic, on_step=self.on_step)

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
        plan_mode: Optional[bool] = None,
        review_mode: Optional[bool] = None,
        review_max_rounds: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
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
        if not isinstance(limits.get("max_tokens"), int) or int(limits.get("max_tokens") or 0) <= 0:
            limits["max_tokens"] = 32768

        eff_plan_mode = self._plan_mode if plan_mode is None else bool(plan_mode)
        eff_review_mode = self._review_mode if review_mode is None else bool(review_mode)
        eff_review_max_rounds = self._review_max_rounds if review_max_rounds is None else int(review_max_rounds)
        if eff_review_max_rounds < 0:
            eff_review_max_rounds = 0

        runtime_ns: Dict[str, Any] = {
            "inbox": [],
            "plan_mode": eff_plan_mode,
            "review_mode": eff_review_mode,
            "review_max_rounds": eff_review_max_rounds,
        }
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
    provider: str = "ollama",
    model: str = "qwen3:1.7b-q4_K_M",
    tools: Optional[List[Callable[..., Any]]] = None,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    max_iterations: int = 25,
    max_history_messages: int = -1,
    max_tokens: Optional[int] = None,
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

    runtime = create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        tools=MappingToolExecutor.from_tools(tools),
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
        actor_id=actor_id,
        session_id=session_id,
    )
