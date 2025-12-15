"""CodeAct agent implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolDefinition
from abstractruntime import RunState, Runtime, WorkflowSpec

from .base import BaseAgent
from ..adapters.codeact_runtime import create_codeact_workflow
from ..logic.builtins import ASK_USER_TOOL
from ..logic.codeact import CodeActLogic


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


class CodeActAgent(BaseAgent):
    """Agent that primarily acts by executing Python code snippets."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        tools: Optional[List[Callable[..., Any]]] = None,
        on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_iterations: int = 20,
        max_history_messages: int = 12,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self._max_iterations = int(max_iterations)
        if self._max_iterations < 1:
            self._max_iterations = 1
        self._max_history_messages = int(max_history_messages)
        if self._max_history_messages < 1:
            self._max_history_messages = 1

        self.logic: Optional[CodeActLogic] = None
        super().__init__(
            runtime=runtime,
            tools=tools,
            on_step=on_step,
            actor_id=actor_id,
            session_id=session_id,
        )

    def _create_workflow(self) -> WorkflowSpec:
        tool_defs = _tool_definitions_from_callables(self.tools)
        tool_defs = [ASK_USER_TOOL, *tool_defs]
        logic = CodeActLogic(tools=tool_defs, max_history_messages=self._max_history_messages)
        self.logic = logic
        return create_codeact_workflow(logic=logic, on_step=self.on_step)

    def start(self, task: str) -> str:
        task = str(task or "").strip()
        if not task:
            raise ValueError("task must be a non-empty string")

        vars: Dict[str, Any] = {
            "context": {"task": task, "messages": _copy_messages(self.session_messages)},
            "scratchpad": {"iteration": 0, "max_iterations": int(self._max_iterations)},
            "_runtime": {"inbox": []},
            "_temp": {},
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
        return self.runtime.tick(workflow=self.workflow, run_id=self._current_run_id, max_steps=1)


def create_codeact_agent(
    *,
    provider: str = "ollama",
    model: str = "qwen3:1.7b-q4_K_M",
    tools: Optional[List[Callable[..., Any]]] = None,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    max_iterations: int = 20,
    max_history_messages: int = 12,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    run_store: Optional[Any] = None,
    ledger_store: Optional[Any] = None,
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> CodeActAgent:
    """Factory: create a CodeActAgent with a local AbstractCore-backed runtime."""

    from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime

    if tools is None:
        from ..tools.code_execution import execute_python

        tools = [execute_python]

    runtime = create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        run_store=run_store,
        ledger_store=ledger_store,
        tool_executor=MappingToolExecutor.from_tools(list(tools)),
    )

    return CodeActAgent(
        runtime=runtime,
        tools=list(tools),
        on_step=on_step,
        max_iterations=max_iterations,
        max_history_messages=max_history_messages,
        actor_id=actor_id,
        session_id=session_id,
    )


__all__ = ["CodeActAgent", "create_codeact_workflow", "create_codeact_agent"]

