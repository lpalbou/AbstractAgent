"""Sandbox interfaces for CodeAct-style agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    error: Optional[str] = None


class Sandbox(Protocol):
    # ADR-0027: None/<=0 = no timeout. A 10.0 default used to live on this
    # protocol and every implementation inherited the silent kill.
    def execute(self, code: str, *, timeout_s: Optional[float] = None) -> ExecutionResult: ...

    def reset(self) -> None: ...

