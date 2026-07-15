"""Drift test: the declared emit-step inventory matches the adapter sources.

B-F12 (2026-07-13): the hooks layer promises totality ("never a filter"), so
the raw step names ARE consumer surface — an emit added without declaring it
in `emit_inventory.py` (or removed while still declared) is a silent contract
change. This test extracts every `emit("...")` literal from each adapter
source and diffs it against the declared inventory, both directions.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from abstractagent.adapters.emit_inventory import ADAPTER_STEP_INVENTORY

_ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "src" / "abstractagent" / "adapters"

_EMIT_RE = re.compile(r'emit\(\s*"([a-z_]+)"')


@pytest.mark.parametrize("adapter", sorted(ADAPTER_STEP_INVENTORY))
def test_declared_inventory_matches_emit_call_sites(adapter: str) -> None:
    source = (_ADAPTERS_DIR / f"{adapter}_runtime.py").read_text(encoding="utf-8")
    actual = set(_EMIT_RE.findall(source))
    declared = set(ADAPTER_STEP_INVENTORY[adapter])

    undeclared = actual - declared
    stale = declared - actual
    assert not undeclared, (
        f"{adapter}: emits not declared in emit_inventory.py (declare them + hooks.md): {sorted(undeclared)}"
    )
    assert not stale, (
        f"{adapter}: declared steps with no emit call site (remove or restore): {sorted(stale)}"
    )


def test_emit_calls_use_literal_step_names() -> None:
    """The drift test is only sound if every emit uses a literal first arg —
    a variable step name would be invisible to the regex. The only allowed
    non-literal occurrence is the `def emit(step...)` helper itself."""
    for adapter in sorted(ADAPTER_STEP_INVENTORY):
        source = (_ADAPTERS_DIR / f"{adapter}_runtime.py").read_text(encoding="utf-8")
        nonliteral = [
            m.group(1)
            for m in re.finditer(r'emit\(\s*(?!")([a-zA-Z_][a-zA-Z_0-9]*)', source)
            if m.group(1) != "step"
        ]
        assert not nonliteral, f"{adapter}: emit() with non-literal step name: {nonliteral}"
