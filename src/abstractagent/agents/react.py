"""ReAct agent implementation.

This module wires:
- Pure ReAct reasoning logic (abstractagent.logic)
- To an AbstractRuntime workflow (abstractagent.adapters)

The public API is intentionally stable:
- ReactAgent
- create_react_workflow
- create_react_agent
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolDefinition
from abstractruntime import RunState, RunStatus, Runtime, WorkflowSpec

from .base import BaseAgent
from ..adapters.loop_hooks import LoopHooks
from ..adapters.react_runtime import create_react_workflow
from ..logic.builtins import (
    ASK_USER_TOOL,
    COMPACT_MEMORY_TOOL,
    DELEGATE_AGENT_TOOL,
    INSPECT_VARS_TOOL,
    OPEN_ATTACHMENT_TOOL,
    RECALL_MEMORY_TOOL,
    REMEMBER_TOOL,
    REMEMBER_NOTE_TOOL,
    UPDATE_PLAN_TOOL,
)
from ..logic.react import ReActLogic


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


class ReactAgent(BaseAgent):
    """Reason-Act-Observe agent with tool calling."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        tools: Optional[List[Callable[..., Any]]] = None,
        on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        hooks: Optional[LoopHooks] = None,
        max_iterations: int = 20,
        max_history_messages: int = -1,
        # In ReAct, max_tokens IS the output-token cap (kept for compat);
        # max_output_tokens is the honest spelling of the same knob — when both
        # are given, max_output_tokens wins (fable5 B-F9, 2026-07-13).
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        plan_mode: bool = False,
        # DEFAULT ON (re-flipped 2026-07-13): the 2026-07-09 opt-in flip named its own
        # re-flip condition — "re-default to on once review failures degrade to
        # accept-with-#FALLBACK instead of killing the run" — and backlog 0027 shipped
        # exactly that containment (a failed verifier call accepts the held answer with
        # a loud #FALLBACK marker; never a dead run). CodeAct already defaulted True;
        # abstractcode re-flipped its CLI default on the same condition (c1207). Price:
        # one verifier call per candidate final answer (~seconds local / ~half a cent
        # API). Pass review_mode=False to opt out.
        review_mode: bool = True,
        review_max_rounds: int = 3,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.hooks = hooks
        self._max_iterations = int(max_iterations)
        if self._max_iterations < 1:
            self._max_iterations = 1
        self._max_history_messages = int(max_history_messages)
        # -1 means unlimited (send all messages), otherwise must be >= 1
        if self._max_history_messages != -1 and self._max_history_messages < 1:
            self._max_history_messages = 1
        self._max_tokens = max_output_tokens if max_output_tokens is not None else max_tokens
        self._plan_mode = bool(plan_mode)
        self._review_mode = bool(review_mode)
        self._review_max_rounds = int(review_max_rounds)
        if self._review_max_rounds < 0:
            self._review_max_rounds = 0

        self.logic: Optional[ReActLogic] = None
        super().__init__(
            runtime=runtime,
            tools=tools,
            on_step=on_step,
            actor_id=actor_id,
            session_id=session_id,
        )

    def _create_workflow(self) -> WorkflowSpec:
        tool_defs = _tool_definitions_from_callables(self.tools)
        # Built-in ask_user is a schema-only tool (handled via ASK_USER effect in the adapter).
        tool_defs = [
            ASK_USER_TOOL,
            OPEN_ATTACHMENT_TOOL,
            RECALL_MEMORY_TOOL,
            INSPECT_VARS_TOOL,
            REMEMBER_TOOL,
            REMEMBER_NOTE_TOOL,
            COMPACT_MEMORY_TOOL,
            DELEGATE_AGENT_TOOL,
            UPDATE_PLAN_TOOL,
            *tool_defs,
        ]

        logic = ReActLogic(tools=tool_defs)
        self.logic = logic
        return create_react_workflow(logic=logic, on_step=self.on_step, hooks=self.hooks)

    def start(
        self,
        task: str,
        *,
        plan_mode: Optional[bool] = None,
        review_mode: Optional[bool] = None,
        review_max_rounds: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        attachments: Optional[List[Any]] = None,
        # Named system-prompt slots (docs/skills-attachment.md): reachable from
        # the facade so README users don't need raw run vars (design adversary
        # P1 2026-07-13). Both must be byte-stable for the run (cache prefix).
        skills_block: Optional[str] = None,
        system_prompt_extra: Optional[str] = None,
    ) -> str:
        task = str(task or "").strip()
        if not task:
            raise ValueError("task must be a non-empty string")

        eff_plan_mode = self._plan_mode if plan_mode is None else bool(plan_mode)
        eff_review_mode = self._review_mode if review_mode is None else bool(review_mode)
        eff_review_max_rounds = self._review_max_rounds if review_max_rounds is None else int(review_max_rounds)
        if eff_review_max_rounds < 0:
            eff_review_max_rounds = 0

        # Base limits come from the Runtime config so model capabilities (max context)
        # are respected by default, unless explicitly overridden by the agent/session.
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
        # Message-size guards for LLM-visible context (character-level).
        # These are applied when building the provider payload; the durable run history
        # still preserves full message text.
        # Disabled by default (-1): enable by setting a positive character budget.
        limits.setdefault("max_message_chars", -1)
        limits.setdefault("max_tool_message_chars", -1)
        limits["estimated_tokens_used"] = 0
        # ReAct output-token capping is controlled via `_limits.max_output_tokens`.
        # Policy: unset by default (None) to avoid artificial truncation.
        try:
            max_output_tokens_override = int(self._max_tokens) if self._max_tokens is not None else None
        except Exception:
            max_output_tokens_override = None
        if isinstance(max_output_tokens_override, int) and max_output_tokens_override > 0:
            limits["max_output_tokens"] = max_output_tokens_override
        else:
            limits.setdefault("max_output_tokens", None)

        runtime_ns: Dict[str, Any] = {
            "inbox": [],
            "plan_mode": eff_plan_mode,
            "review_mode": eff_review_mode,
            "review_max_rounds": eff_review_max_rounds,
        }
        if isinstance(skills_block, str) and skills_block.strip():
            runtime_ns["skills_block"] = skills_block
        if isinstance(system_prompt_extra, str) and system_prompt_extra.strip():
            runtime_ns["system_prompt_extra"] = system_prompt_extra
        # Explicit output caps must SURVIVE the adapter's init_node, which
        # deliberately nulls config-derived _limits caps (full-context policy).
        # `_runtime.max_output_tokens` is the surviving channel the adapter's
        # `_max_output_tokens` helper reads as fallback (fable5 B-F9/0029 #9,
        # 2026-07-13: `ReactAgent(max_tokens=4096)` was silently dropped).
        if isinstance(max_output_tokens_override, int) and max_output_tokens_override > 0:
            runtime_ns["max_output_tokens"] = max_output_tokens_override

        vars: Dict[str, Any] = {
            "context": {"task": task, "messages": _copy_messages(self.session_messages)},
            "scratchpad": {"iteration": 0, "max_iterations": int(self._max_iterations)},
            "_runtime": runtime_ns,
            "_temp": {},
            # Canonical _limits namespace for runtime awareness
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
        if temperature is not None:
            try:
                vars["_runtime"]["temperature"] = float(temperature)
            except Exception:
                pass
        if seed is not None:
            try:
                vars["_runtime"]["seed"] = int(seed)
            except Exception:
                pass
        if isinstance(allowed_tools, list):
            normalized = [str(t).strip() for t in allowed_tools if isinstance(t, str) and t.strip()]
            vars["_runtime"]["allowed_tools"] = normalized

        run_id = self.runtime.start(
            workflow=self.workflow,
            vars=vars,
            actor_id=self._ensure_actor_id(),
            session_id=self._ensure_session_id(),
        )
        self._current_run_id = run_id
        return run_id

    def get_limit_status(self) -> Dict[str, Any]:
        """Get current limit status for the active run.

        Returns a structured dict with information about iterations, tokens,
        and history limits, including whether warning thresholds are reached.

        Returns:
            Dict with "iterations", "tokens", and "history" status info,
            or empty dict if no active run.
        """
        if self._current_run_id is None:
            return {}
        return self.runtime.get_limit_status(self._current_run_id)

    def update_limits(self, **updates: Any) -> None:
        """Update limits mid-session.

        Only allowed limit keys are updated; unknown keys are ignored.
        Allowed keys (runtime contract, fable5 B-F11 2026-07-13): max_iterations,
        max_tokens, max_output_tokens, max_input_tokens, max_history_messages,
        warn_iterations_pct, warn_tokens_pct, estimated_tokens_used,
        current_iteration.

        Honesty note: ReAct sends the full transcript every cycle by policy,
        so updating max_history_messages mid-run has no effect on this loop.
        Naming trap (B-F9): in THIS method `max_tokens` is the runtime's
        context ACCOUNTING ceiling (`_limits.max_tokens`) — not the output cap
        the constructor's `max_tokens` parameter maps to; use
        `max_output_tokens` here for output capping.

        Args:
            **updates: Limit key-value pairs to update

        Raises:
            RuntimeError: If no active run
        """
        if self._current_run_id is None:
            raise RuntimeError("No active run. Call start() first.")
        self.runtime.update_limits(self._current_run_id, updates)

    def step(self) -> RunState:
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start() first.")
        state = self.runtime.tick(workflow=self.workflow, run_id=self._current_run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            self._sync_session_caches_from_state(state)
        return state


def create_react_agent(
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
    plan_mode: bool = False,
    review_mode: bool = True,  # default on since 2026-07-13; see ReactAgent.__init__ (0027 containment shipped)
    review_max_rounds: int = 3,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    run_store: Optional[Any] = None,
    ledger_store: Optional[Any] = None,
    actor_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> ReactAgent:
    """Factory: create a ReactAgent with a local AbstractCore-backed runtime."""

    from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime

    if tools is None:
        from ..tools import ALL_TOOLS as _DEFAULT_TOOLS

        tools = list(_DEFAULT_TOOLS)

    from .defaults import resolve_provider_model

    provider, model = resolve_provider_model(provider, model)

    runtime = create_local_runtime(
        provider=provider,
        model=model,
        llm_kwargs=llm_kwargs,
        run_store=run_store,
        ledger_store=ledger_store,
        tool_executor=MappingToolExecutor.from_tools(list(tools)),
    )

    return ReactAgent(
        runtime=runtime,
        tools=list(tools),
        on_step=on_step,
        max_iterations=max_iterations,
        max_history_messages=max_history_messages,
        max_tokens=max_tokens,
        max_output_tokens=max_output_tokens,
        plan_mode=plan_mode,
        review_mode=review_mode,
        review_max_rounds=review_max_rounds,
        actor_id=actor_id,
        session_id=session_id,
    )


__all__ = [
    "ReactAgent",
    "create_react_workflow",
    "create_react_agent",
]
