"""Per-adapter emit-step inventories (fable5 B-F12, promoted 2026-07-13).

The hooks layer promises "the hook set is total, never a filter" — but a
capture host building on that promise had no inventory of the raw step names:
~24 steps were documented nowhere, and consumers discovered asymmetries (ReAct
emits no ``init``; ReAct says ``parse_retry_empty`` where CodeAct says
``parse_retry_empty_response``, renamed) by surprise. These constants are the declared
inventory; ``tests/test_emit_inventory_drift.py`` diffs them against the
actual ``emit("...")`` call sites in each adapter source, so a new emit that
isn't declared here (or a removed one still listed) fails the suite.

Stability contract (see docs/hooks.md):

- CANONICAL event names (the ``DEFAULT_EVENT_MAP`` in ``loop_hooks``) are
  frozen.
- RAW step names below are stable-with-announced-renames: renaming one is a
  consumer-visible change and belongs to a contract revision (backlog 0028),
  never a drive-by edit.
- The founding-publication rename (2026-07-13): CodeAct's
  ``parse_retry_empty_response`` became ``parse_retry_empty`` (one semantic,
  one name, zero consumers existed) — see ``RENAMED_STEPS`` for the alias map.
- Error prose in payloads is NOT a contract; key on step names and payload
  keys (``#FALLBACK``/``#TRUNCATION`` markers are stable).
"""

from __future__ import annotations

# Steps every adapter emits (same name, same rough semantic; payload shapes
# may differ per loop — e.g. `parse` payloads are loop-specific).
COMMON_STEPS = frozenset(
    {
        "act",
        "act_blocked",
        "allowlist_pruned",
        "ask_user",
        "context_warning",
        "delegate_agent",
        "delegate_agent_budget_fallback",
        "delegate_agent_substrate",
        "delegate_agent_substrate_skew",
        "done",
        "hook_steer_discarded",
        "inbox_drained",
        "max_iterations",
        "memory_compact",
        "memory_note",
        "memory_query",
        "memory_tag",
        "observe",
        "parse",
        "reason",
        "user_response",
        "vars_query",
    }
)

# NOTE: ReAct deliberately emits no `init` (CodeAct/MemAct do) — a consumer
# asymmetry that predates the inventory; recorded here rather than papered over.
REACT_STEPS = COMMON_STEPS | frozenset(
    {
        # 0028 multi-emit fix (2026-07-14): `max_iterations_reached` announces
        # budget exhaustion ONCE at first entry of the conclusion node;
        # `max_iterations` (canonical turn_end) fires once, at the completion
        # branch where the turn actually ends.
        "max_iterations_reached",
        "parse_final",
        "parse_repeat_tool_calls",
        "parse_retry_empty",
        "parse_retry_plan_only",
        "parse_retry_truncated",
        "parse_tool_calls",
        "review",
        "review_request",
        "review_skipped",
        "review_tool_calls",
        "update_plan",
    }
)

CODEACT_STEPS = COMMON_STEPS | frozenset(
    {
        "init",
        "parse_retry_empty",
        "plan",
        "plan_request",
        "review",
        "review_request",
        "review_skipped",
        "review_tool_calls",
    }
)

MEMACT_STEPS = COMMON_STEPS | frozenset(
    {
        "init",
        "compose",
        "compose_query",
        "finalize",
        "finalize_request",
        "finalize_skipped",
        "finalize_used_draft",
    }
)

ADAPTER_STEP_INVENTORY = {
    "react": REACT_STEPS,
    "codeact": CODEACT_STEPS,
    "memact": MEMACT_STEPS,
}


#: Old raw step name -> current name. Renames are contract revisions; the map
#: is the migration record consumers can consult (no double-emit — the rename
#: landed in the same wave that FIRST published the inventory, before any
#: consumer existed to key on the old name).
RENAMED_STEPS = {
    "parse_retry_empty_response": "parse_retry_empty",
}
