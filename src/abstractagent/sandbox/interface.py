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
    def execute(self, code: str, *, timeout_s: float = 10.0) -> ExecutionResult: ...

    def reset(self) -> None: ...

