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
        
        return self.runtime.resume(
            workflow=self.workflow,
            run_id=self._current_run_id,
            wait_key=wait_key,
            payload={"response": response},
        )
    
    def attach(self, run_id: str) -> RunState:
        """Attach to an existing run for resume.
        
        Args:
            run_id: ID of the run to attach to
            
        Returns:
            Current RunState
        """
        state = self.runtime.get_state(run_id)
        self._current_run_id = run_id
        return state
    
    def save_state(self, filepath: str) -> None:
        """Save current run_id to file for later resume.
        
        Args:
            filepath: Path to save state file
        """
        import json
        from pathlib import Path
        
        if not self._current_run_id:
            raise RuntimeError("No active run to save.")
        
        state = self.get_state()
        data = {
            "run_id": self._current_run_id,
            "agent_type": self.__class__.__name__,
            "status": state.status.value if state else "unknown",
            "current_node": state.current_node if state else None,
            "task": state.vars.get("task") if state else None,
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    def load_state(self, filepath: str) -> Optional[RunState]:
        """Load run_id from file and attach to it.
        
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
        
        try:
            data = json.loads(path.read_text())
            run_id = data.get("run_id")
            if run_id:
                return self.attach(run_id)
        except (json.JSONDecodeError, KeyError):
            pass
        
        return None
    
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
        inbox = state.vars.get("_inbox", [])
        inbox.append({
            "type": "user_guidance",
            "content": message,
            "timestamp": self.runtime._ctx.now_iso() if hasattr(self.runtime._ctx, 'now_iso') else None,
        })
        state.vars["_inbox"] = inbox
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
