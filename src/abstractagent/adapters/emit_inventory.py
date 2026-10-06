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
        # 0026 follow-ups (2026-07-15): `init` and `parse_tool_calls` became
        # common — ReAct gained the run-init emit (the "ReAct emits no init"
        # asymmetry recorded at founding publication is CLOSED), CodeAct and
        # MemAct gained the tool-batch commit signal (canonical
        # `tool_proposed` now fires on all three loops). `inbox_undelivered`
        # is new: a true terminal reached with undrained durable-inbox
        # guidance says so loudly ({count, chars}) instead of completing
        # over it silently; the entries stay in the run's durable vars.
        "inbox_undelivered",
        "init",
        "inbox_drained",
        "max_iterations",
        "memory_compact",
        "memory_note",
        "memory_query",
        "memory_tag",
        "observe",
        "parse",
        "parse_tool_calls",
        "reason",
        "user_response",
        "vars_query",
    }
)

REACT_STEPS = COMMON_STEPS | frozenset(
    {
        # 0028 multi-emit fix (2026-07-14): `max_iterations_reached` announces
        # budget exhaustion ONCE at first entry of the conclusion node;
        # `max_iterations` (canonical turn_end) fires once, at the completion
        # branch where the turn actually ends.
        "max_iterations_reached",
        "parse_final",
        # Read-orchestration advice — payload {cycle, path, mode, enforcement}.
        # `enforcement: "advise"` means the model was told how to read this
        # file better AND its read still executed. Before 2026-08-22 this hint
        # DROPPED the batch and cost a task iteration for zero observations;
        # run 9ea71c55 lost 3 of 20 iterations to it. Emitted since the guard
        # landed but never declared — the drift test caught it once the
        # payload gained `enforcement`.
        "parse_read_orchestration_hint",
        "parse_repeat_tool_calls",
        "parse_retry_empty",
        "parse_retry_plan_only",
        "parse_retry_truncated",
        # Tool-call format repair (2026-10-03, shipped round 14): AbstractCore
        # reported `metadata.tool_call_error` (code invalid_tool_syntax |
        # unavailable_tool) and accepted ZERO calls; the model is asked once
        # more with the tool roster — payload {cycle, attempt, code}. At most
        # two per turn; the third raises (the run fails with the sentence).
        "parse_retry_tool_format",
        # 0017 work half (2026-07-21), nudge-then-stop (2026-08-21): fires on
        # every detection of a consecutive-identical or A-B-A-B tool-batch
        # streak; payload {kind: repeat|oscillation, span, key, answered,
        # cycle, hits, action: nudged|repeat_after_nudge|stopped}. Only
        # `stopped` ends the turn.
        "stuck_streak",
        # First-failure diagnosis (2026-08-21): a failed tool call was
        # explained back to the model with its own arguments, the verbatim
        # error, the likely cause and a concrete alternative — payload
        # {count, classes, tools}. `tool_failure_hint_error` reports that the
        # hint builder itself raised (the hint is skipped, the run continues;
        # never silent).
        "tool_failure_hint",
        "tool_failure_hint_error",
        # Operator conclude (2026-08-21): `POST /commands {type:"conclude"}`
        # reached this run and the loop is going to its conclusion path
        # instead of spending another iteration — payload {cycle, has_note}.
        # The turn ends with `stop_reason.code = "operator_conclude"`, which
        # is neither a failure nor a budget stop.
        "conclude_requested",
        # Sight lane (c3969 shape A / c4089 ruling, 2026-07-21): a successful
        # tool result DECLARED media refs on its output dict — payload
        # {tool, count, call_id}. Fires at CAPTURE time; refs that survive
        # the burst bound ride the next reason/conclude call's `media`
        # one-shot (trims are announced by media_dropped, so the pair is the
        # honest record of what actually rode).
        "media_captured",
        # The pending-media burst bound trimmed oldest refs (most-recent
        # wins) — payload {dropped, kept, reason, dropped_keys}; never silent.
        "media_dropped",
        "review",
        "review_request",
        "review_skipped",
        "review_tool_calls",
        "update_plan",
    }
)

CODEACT_STEPS = COMMON_STEPS | frozenset(
    {
        "parse_retry_empty",
        # Read-orchestration advice — payload {path, mode, enforcement}.
        # `enforcement: "advise"` means the model was told how to read this
        # file better AND its read still executed; the loop must never spend
        # a task iteration refusing a side-effect-free read (2026-08-22, run
        # 9ea71c55). The react adapter emits the same name with a `cycle`.
        # It was emitted here since the guard landed but never declared —
        # the drift test caught it once the payload changed.
        "parse_read_orchestration_hint",
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
