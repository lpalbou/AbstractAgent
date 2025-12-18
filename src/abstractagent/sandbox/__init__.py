"""Sandbox implementations used by agents."""

from .interface import ExecutionResult, Sandbox
from .local import LocalSandbox

__all__ = ["ExecutionResult", "Sandbox", "LocalSandbox"]

