"""Base agent class with common functionality.

All agent types (ReAct, CodeAct, etc.) inherit from BaseAgent to get:
- Runtime access (ledger, run store, cancel)
- State management (save/load/attach)
- Async message injection
- Common lifecycle methods
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from abstractruntime import Runtime, RunState, RunStatus, WorkflowSpec


class BaseAgent(ABC):
    """Abstract base class for all agent types.
    
    Provides common functionality that all agents need:
    - Runtime integration
    - State persistence
    - Cancellation
    - Ledger access
    - Async message injection
    
    Subclasses must implement:
    - _create_workflow(): Return the WorkflowSpec for this agent type
    - start(): Initialize and start a run
    - step(): Execute one step
    """
    
    def __init__(
        self,
        runtime: Runtime,
        tools: Optional[List[Callable]] = None,
        on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        *,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the agent.
        
        Args:
            runtime: AbstractRuntime instance for durable execution
            tools: List of tool functions decorated with @tool
            on_step: Optional callback for step visibility (step_name, data)
        """
        self.runtime = runtime
        self.tools = tools or []
        self.on_step = on_step
        self.workflow = self._create_workflow()
        self._current_run_id: Optional[str] = None
        self.actor_id: Optional[str] = actor_id
        self.session_id: Optional[str] = session_id
        self.session_messages: List[Dict[str, Any]] = []

    def _sync_session_caches_from_state(self, state: Optional[RunState]) -> None:
        if state is None or not hasattr(state, "vars") or not isinstance(state.vars, dict):
            return

        messages: Optional[List[Dict[str, Any]]] = None
        output = getattr(state, "output", None)
        if isinstance(output, dict) and isinstance(output.get("messages"), list):
            messages = [dict(m) for m in output["messages"] if isinstance(m, dict)]
        else:
            ctx = state.vars.get("context")
            if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list):
                messages = [dict(m) for m in ctx["messages"] if isinstance(m, dict)]

        if messages is not None:
            self.session_messages = list(messages)

    def _ensure_actor_id(self) -> str:
        if self.actor_id:
            return self.actor_id

        from abstractruntime.identity.fingerprint import ActorFingerprint
        import uuid

        fp = ActorFingerprint.from_metadata(
            kind="agent",
            display_name=self.__class__.__name__,
            metadata={"nonce": uuid.uuid4().hex},
        )
        self.actor_id = fp.actor_id
        return self.actor_id

    def _ensure_session_id(self) -> str:
        if self.session_id:
            return self.session_id
        import uuid

        self.session_id = f"sess_{uuid.uuid4().hex}"
        return self.session_id

    @abstractmethod
    def _create_workflow(self) -> WorkflowSpec:
        """Create the workflow specification for this agent type.
        
        Returns:
            WorkflowSpec defining the agent's execution graph
        """
        pass
    
    @abstractmethod
    def start(self, task: str) -> str:
        """Start a new agent run with a task.
        
        Args:
            task: The task description for the agent
            
        Returns:
            The run_id for this execution
        """
        pass
    
    @abstractmethod
    def step(self) -> RunState:
        """Execute one step of the agent.
        
        Returns:
            Current RunState after the step
        """
        pass
    
    # -------------------------------------------------------------------------
    # Common methods (inherited by all agent types)
    # -------------------------------------------------------------------------
    
    def run_to_completion(self) -> RunState:
        """Run the agent until completion or waiting.
        
        Returns:
            Final RunState (COMPLETED, FAILED, or WAITING)
        """
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start() first.")
        
        state = self.step()
        while state.status == RunStatus.RUNNING:
            state = self.step()
        
        return state
    
    def get_state(self) -> Optional[RunState]:
        """Get current agent state.
        
        Returns:
            Current RunState or None if no active run
        """
        if not self._current_run_id:
            return None
        return self.runtime.get_state(self._current_run_id)

    def get_context(self) -> Dict[str, Any]:
        """Get the agent's current context namespace (runtime-owned persisted state)."""
        state = self.get_state()
        ctx = state.vars.get("context") if state and hasattr(state, "vars") else None
        return dict(ctx) if isinstance(ctx, dict) else {}

    def get_scratchpad(self) -> Dict[str, Any]:
        """Get the agent's current scratchpad namespace (agent-owned schema, runtime-owned storage)."""
        state = self.get_state()
        scratch = state.vars.get("scratchpad") if state and hasattr(state, "vars") else None
        return dict(scratch) if isinstance(scratch, dict) else {}

    def get_node_traces(self) -> Dict[str, Any]:
        """Get runtime-owned per-node traces for the current run (passthrough to Runtime)."""
        if not self._current_run_id:
            return {}
        getter = getattr(self.runtime, "get_node_traces", None)
        if callable(getter):
            return getter(self._current_run_id)
        state = self.get_state()
        runtime_ns = state.vars.get("_runtime") if state and hasattr(state, "vars") else None
        traces = runtime_ns.get("node_traces") if isinstance(runtime_ns, dict) else None
        return dict(traces) if isinstance(traces, dict) else {}

    def get_node_trace(self, node_id: str) -> Dict[str, Any]:
        """Get a single runtime-owned node trace for the current run."""
        if not self._current_run_id:
            return {"node_id": node_id, "steps": []}
        getter = getattr(self.runtime, "get_node_trace", None)
        if callable(getter):
            return getter(self._current_run_id, node_id)
        traces = self.get_node_traces()
        trace = traces.get(node_id)
        if isinstance(trace, dict):
            return trace
        return {"node_id": node_id, "steps": []}
    
    def is_waiting(self) -> bool:
        """Check if agent is waiting for input.
        
        Returns:
            True if agent is in WAITING status
        """
        state = self.get_state()
        return state is not None and state.status == RunStatus.WAITING
    
    def is_running(self) -> bool:
        """Check if agent is actively running.
        
        Returns:
            True if agent is in RUNNING status
        """
        state = self.get_state()
        return state is not None and state.status == RunStatus.RUNNING
    
    def is_complete(self) -> bool:
        """Check if agent has completed.
        
        Returns:
            True if agent is in COMPLETED status
        """
        state = self.get_state()
        return state is not None and state.status == RunStatus.COMPLETED
    
    def get_pending_question(self) -> Optional[Dict[str, Any]]:
        """Get pending question if agent is waiting for user input.
        
        Returns:
            Dict with prompt, choices, allow_free_text, wait_key
            or None if not waiting
        """
        state = self.get_state()
        if not state or state.status != RunStatus.WAITING or not state.waiting:
            return None
        
        return {
            "prompt": state.waiting.prompt,
            "choices": state.waiting.choices,
            "allow_free_text": state.waiting.allow_free_text,
            "wait_key": state.waiting.wait_key,
        }
    
    def resume(self, response: str) -> RunState:
        """Resume agent with user response.
        
        Args:
            response: User's answer to the pending question
            
        Returns:
            Updated RunState after resuming
        """
        if not self._current_run_id:
            raise RuntimeError("No active run.")
        
        state = self.get_state()
        if not state or state.status != RunStatus.WAITING:
            raise RuntimeError("Agent is not waiting for input.")
        
        wait_key = state.waiting.wait_key if state.waiting else None
        
        state2 = self.runtime.resume(
            workflow=self.workflow,
            run_id=self._current_run_id,
            wait_key=wait_key,
            payload={"response": response},
        )
        if state2.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            self._sync_session_caches_from_state(state2)
        return state2
    
    def attach(self, run_id: str) -> RunState:
        """Attach to an existing run for resume.
        
        Args:
            run_id: ID of the run to attach to
            
        Returns:
            Current RunState
        """
        state = self.runtime.get_state(run_id)
        if state.workflow_id != self.workflow.workflow_id:
            raise ValueError(
                f"Run workflow_id mismatch: run has '{state.workflow_id}', "
                f"agent expects '{self.workflow.workflow_id}'."
            )

        if self.actor_id and state.actor_id and self.actor_id != state.actor_id:
            raise ValueError(
                f"Run actor_id mismatch: run has '{state.actor_id}', agent expects '{self.actor_id}'."
            )
        if self.actor_id is None and state.actor_id:
            self.actor_id = state.actor_id

        state_session_id = getattr(state, "session_id", None)
        if self.session_id and state_session_id and self.session_id != state_session_id:
            raise ValueError(
                f"Run session_id mismatch: run has '{state_session_id}', agent expects '{self.session_id}'."
            )
        if self.session_id is None and state_session_id:
            self.session_id = state_session_id

        self._current_run_id = run_id
        self._sync_session_caches_from_state(state)
        return state
    
    def save_state(self, filepath: str) -> None:
        """Save a run reference to file for later resume.

        The full durable state is owned and persisted by AbstractRuntime's RunStore.
        This method only stores the identifiers needed to re-attach on restart.
        
        Args:
            filepath: Path to save state file
        """
        import json
        from pathlib import Path
        from abstractruntime.storage.in_memory import InMemoryRunStore
        
        if not self._current_run_id:
            raise RuntimeError("No active run to save.")

        if isinstance(self.runtime.run_store, InMemoryRunStore):
            raise RuntimeError(
                "save_state requires a persistent RunStore (e.g. JsonFileRunStore); "
                "the current runtime uses InMemoryRunStore which cannot resume across restarts."
            )

        data = {
            "run_id": self._current_run_id,
            "workflow_id": self.workflow.workflow_id,
            "actor_id": self.actor_id,
            "session_id": self._ensure_session_id(),
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    def load_state(self, filepath: str) -> Optional[RunState]:
        """Load run reference from file and attach to it.

        The full durable state is loaded from AbstractRuntime's RunStore.
        
        Args:
            filepath: Path to state file
            
        Returns:
            RunState if found and valid, None otherwise
        """
        import json
        from pathlib import Path
        
        path = Path(filepath)
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        run_id = data.get("run_id")
        if not run_id:
            return None

        workflow_id = data.get("workflow_id")
        if workflow_id and workflow_id != self.workflow.workflow_id:
            raise ValueError(
                f"Saved workflow_id mismatch: file has '{workflow_id}', "
                f"agent expects '{self.workflow.workflow_id}'."
            )

        actor_id = data.get("actor_id")
        if actor_id and self.actor_id and actor_id != self.actor_id:
            raise ValueError(
                f"Saved actor_id mismatch: file has '{actor_id}', agent expects '{self.actor_id}'."
            )
        if actor_id and self.actor_id is None:
            self.actor_id = actor_id

        session_id = data.get("session_id")
        if session_id and self.session_id and session_id != self.session_id:
            raise ValueError(
                f"Saved session_id mismatch: file has '{session_id}', agent expects '{self.session_id}'."
            )
        if session_id and self.session_id is None:
            self.session_id = session_id

        try:
            return self.attach(str(run_id))
        except KeyError as e:
            raise RuntimeError(
                f"Saved run_id '{run_id}' was not found in the configured RunStore. "
                "If you deleted/moved the store directory, this run cannot be resumed."
            ) from e
    
    def clear_state(self, filepath: str) -> None:
        """Remove state file after completion.
        
        Args:
            filepath: Path to state file
        """
        from pathlib import Path
        Path(filepath).unlink(missing_ok=True)
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._current_run_id
    
    def cancel(self, reason: Optional[str] = None) -> RunState:
        """Cancel the current run.
        
        Args:
            reason: Optional cancellation reason
            
        Returns:
            Updated RunState with CANCELLED status
        """
        if not self._current_run_id:
            raise RuntimeError("No active run to cancel.")
        
        return self.runtime.cancel_run(self._current_run_id, reason=reason)
    
    def get_ledger(self) -> list:
        """Get ledger entries for the current run.
        
        Returns:
            List of ledger entries (dicts with effect details)
        """
        if not self._current_run_id:
            return []
        
        return self.runtime.get_ledger(self._current_run_id)
    
    def inject_message(self, message: str) -> None:
        """Inject a message into the agent's inbox for next iteration.
        
        The agent will see this message on its next reasoning step.
        Useful for providing guidance or additional context while running.
        
        Args:
            message: Message to inject
        """
        if not self._current_run_id:
            raise RuntimeError("No active run.")
        
        state = self.runtime.get_state(self._current_run_id)
        runtime_ns = state.vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            runtime_ns = {}
            state.vars["_runtime"] = runtime_ns

        inbox = runtime_ns.get("inbox")
        if not isinstance(inbox, list):
            legacy = state.vars.get("_inbox")
            inbox = legacy if isinstance(legacy, list) else []
            runtime_ns["inbox"] = inbox

        inbox.append(
            {
                "type": "user_guidance",
                "content": message,
                "timestamp": self.runtime._ctx.now_iso() if hasattr(self.runtime._ctx, "now_iso") else None,
            }
        )
        state.vars.pop("_inbox", None)
        self.runtime._run_store.save(state)
    
    def get_output(self) -> Optional[Dict[str, Any]]:
        """Get the output from a completed run.
        
        Returns:
            Output dict if completed, None otherwise
        """
        state = self.get_state()
        if state and state.status == RunStatus.COMPLETED:
            return state.output
        return None
    
    def get_error(self) -> Optional[str]:
        """Get the error from a failed run.
        
        Returns:
            Error string if failed, None otherwise
        """
        state = self.get_state()
        if state and state.status == RunStatus.FAILED:
            return state.error
        return None
