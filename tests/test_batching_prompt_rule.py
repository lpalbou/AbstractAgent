"""Pins for the narrowed tool-batching prompt rule (operator ruling 2026-07-27).

The old blanket rule ("do NOT batch side-effectful tools") made the model
serialize everything: measured 79.5% singleton batches, edit_file batched
0/149. The real hazard is SAME-target batching only (two edits to one file
shift each other's line coordinates); the runtime executes a batch in order,
so different-target writes are safe in one turn. The narrowed rule plus the
one-call multi-hunk diff pointer replaced the blanket text in all three loop
prompts. Prompt text IS the mechanism here — these pins exist so the rule
can never silently revert (the old text was removed with zero test delta).
"""

from __future__ import annotations

import pytest

from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic

pytestmark = pytest.mark.basic


def _system_prompt(logic_cls) -> str:
    logic = logic_cls(tools=[])
    request = logic.build_request(task="t", messages=[{"role": "user", "content": "t"}])
    return str(request.system_prompt or "")


@pytest.mark.parametrize("logic_cls", [ReActLogic, CodeActLogic, MemActLogic], ids=["react", "codeact", "memact"])
def test_same_target_ban_and_execute_ban_present(logic_cls) -> None:
    prompt = _system_prompt(logic_cls)
    # The narrowed hazard: same-target batching is forbidden, by name.
    assert "never batch two calls that touch the SAME target" in prompt
    # The operator's "execute never batches" must bind to the real tool name.
    assert "execute_command" in prompt
    # The compliant alternative is taught next to the ban.
    assert "One call is not a batch" in prompt


@pytest.mark.parametrize("logic_cls", [ReActLogic, CodeActLogic, MemActLogic], ids=["react", "codeact", "memact"])
def test_blanket_rule_is_gone(logic_cls) -> None:
    prompt = _system_prompt(logic_cls)
    assert "do NOT batch side-effectful tools" not in prompt
    assert "avoid batching side-effectful tools" not in prompt
    # The old streak generator ("refine via multiple smaller edits") is gone.
    assert "multiple smaller edits" not in prompt


def test_codeact_bans_its_own_executor_by_name() -> None:
    prompt = _system_prompt(CodeActLogic)
    assert "execute_python" in prompt


@pytest.mark.parametrize("logic_cls", [ReActLogic, CodeActLogic, MemActLogic], ids=["react", "codeact", "memact"])
def test_nearby_same_file_reads_prefer_one_wider_range(logic_cls) -> None:
    prompt = _system_prompt(logic_cls)
    assert "prefer ONE call with a wider range" in prompt
