"""Local subprocess sandbox (development-only).

This is intentionally minimal: it captures stdout/stderr and enforces a timeout
ONLY when the caller asks for one (ADR-0027: no low default timeouts).
Stronger isolation (docker/e2b/wasm) can be added later.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Optional

from .interface import ExecutionResult


class LocalSandbox:
    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        python_executable: Optional[str] = None,
        command_sandbox: Any = None,
    ):
        """`command_sandbox`: an AbstractCore `abstractcore.tools.sandbox.Sandbox` (round 12). When
        given, the interpreter runs inside it (OS sandbox bound to the run's workspaces, the host's
        scrubbed environment, a private TMPDIR); the caller refuses before when it `refuses`."""
        self._cwd = cwd or os.getcwd()
        self._python = python_executable or sys.executable
        self._command_sandbox = command_sandbox

    def reset(self) -> None:
        # Stateless sandbox (new subprocess per call).
        return None

    def execute(self, code: str, *, timeout_s: Optional[float] = None) -> ExecutionResult:
        """Run `code` in a subprocess. `timeout_s` None/<=0 = NO timeout.

        ADR-0027: a 10s default used to sit on this signature and silently
        killed legitimate long computations (reported to the model as a plain
        error, indistinguishable from a code bug). A timeout here is an
        EXPLICIT caller safeguard now, never a hidden performance knob.
        """
        started = time.monotonic()
        #[WARNING:TIMEOUT] caller-supplied only; None/<=0 = unlimited (ADR-0027)
        eff_timeout: Optional[float] = None
        if timeout_s is not None:
            try:
                t = float(timeout_s)
            except (TypeError, ValueError):
                t = 0.0
            eff_timeout = t if t > 0 else None
        argv = [self._python, "-c", code]
        env = None
        box = self._command_sandbox
        if box is not None:
            if box.kind == "unsandboxed":
                env = box.env_for()
            else:
                argv, env = box.wrap(argv, self._cwd)
        try:
            completed = subprocess.run(
                argv,
                cwd=self._cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=eff_timeout,
            )
            duration_ms = (time.monotonic() - started) * 1000.0
            return ExecutionResult(
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                exit_code=int(completed.returncode),
                duration_ms=duration_ms,
                error=None,
            )
        except subprocess.TimeoutExpired as e:
            duration_ms = (time.monotonic() - started) * 1000.0
            return ExecutionResult(
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                exit_code=124,
                duration_ms=duration_ms,
                error=f"#[WARNING:TIMEOUT] abstractagent.sandbox.local: timed out after {eff_timeout}s (caller-supplied timeout_s)",
            )
        except Exception as e:
            duration_ms = (time.monotonic() - started) * 1000.0
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                duration_ms=duration_ms,
                error=str(e),
            )

