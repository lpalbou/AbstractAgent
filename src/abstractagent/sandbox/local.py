"""Local subprocess sandbox (development-only).

This is intentionally minimal: it enforces a timeout and captures stdout/stderr.
Stronger isolation (docker/e2b/wasm) can be added later.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

from .interface import ExecutionResult


class LocalSandbox:
    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        python_executable: Optional[str] = None,
    ):
        self._cwd = cwd or os.getcwd()
        self._python = python_executable or sys.executable

    def reset(self) -> None:
        # Stateless sandbox (new subprocess per call).
        return None

    def execute(self, code: str, *, timeout_s: float = 10.0) -> ExecutionResult:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [self._python, "-c", code],
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
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
                error=f"Timeout after {timeout_s}s",
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

