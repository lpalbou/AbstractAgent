"""Unattended-run recipe (fable5 A audit, promoted 2026-07-13).

Both delegate sites and unattended hosts (work doors, schedulers, resident
agents) need the same two moves: exclude ``ask_user`` from the allowlist (an
unattended run that asks a question deadlocks until a human notices) and set
the no-questions directive. The delegate branches ship these moves inline;
hosts kept rediscovering them by reading adapter source. This module is the
packaged recipe — one import instead of folklore.

Usage (raw-workflow host)::

    from abstractagent.agents.unattended import unattended_runtime_overrides

    overrides = unattended_runtime_overrides(base_allowlist=my_tools)
    vars["_runtime"].update(overrides)

Usage (facade)::

    agent.start(task, allowed_tools=unattended_allowlist(my_tools))

The directive rides ``system_prompt_extra`` — byte-stable for the run (cache
contract), composed at a fixed post-base position by all three adapters.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# Kept in sync with the delegate branches' inline directive (same intent, host
# wording): the delegated-sub-agent framing is replaced by the unattended one.
UNATTENDED_DIRECTIVE = (
    "You are running unattended (no human is watching this run).\n"
    "- Do not ask the user questions; if blocked, state assumptions and proceed.\n"
    "- Prefer completing the task over waiting for clarification.\n"
    "- Report what you did and any assumptions in the final answer.\n"
)

#: Tools that BLOCK an unattended run waiting on a human. `ask_user` parks the
#: run on a wait; excluding it is the structural half of the recipe (the
#: directive is the behavioral half — both are needed: a directive alone still
#: leaves the tool callable).
BLOCKING_TOOLS = frozenset({"ask_user"})


def unattended_allowlist(base_allowlist: Optional[Iterable[str]] = None) -> Optional[List[str]]:
    """Base allowlist minus blocking tools; None passes through (meaning
    "runtime default set") — callers who want the default set minus ask_user
    must materialize their tool names first, because this helper never guesses
    the registry."""
    if base_allowlist is None:
        return None
    return [str(t) for t in base_allowlist if str(t) not in BLOCKING_TOOLS]


def unattended_runtime_overrides(
    base_allowlist: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """`_runtime` overrides for an unattended run.

    Returns ``system_prompt_extra`` (always) and ``allowed_tools`` (only when a
    base allowlist was given — never invents one). Merge into the run's
    ``_runtime`` namespace before start.
    """
    overrides: Dict[str, Any] = {"system_prompt_extra": UNATTENDED_DIRECTIVE}
    allow = unattended_allowlist(base_allowlist)
    if allow is not None:
        overrides["allowed_tools"] = allow
    return overrides
