"""Pure logic types shared across agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from abstractcore.tools import ToolDefinition


@dataclass(frozen=True)
class LLMRequest:
    """What to ask the LLM (portable; runtime-agnostic)."""

    prompt: str
    tools: List[ToolDefinition]
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class AskUserAction:
    question: str
    choices: Optional[List[str]] = None


@dataclass(frozen=True)
class FinalAnswer:
    content: str

