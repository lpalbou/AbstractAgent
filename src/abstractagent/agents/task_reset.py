"""Task reset for queued work (0030 promoted 2026-07-14, operator green light).

The work-door shape this serves: a door (gateway work door, scheduler, fleet
bridge) runs tasks against an agent loop and wants the NEXT task to start
clean without discarding what should persist. v1 stays ONE RUN PER TASK —
this is a pure VARS TRANSFORM the door applies between runs (take the
completed run's vars, reset them, seed `runtime.start(vars=...)`), never a
mid-run mutation (the single-writer contract: only the tick thread mutates a
live run's vars).

What resets (per-task state):
- `context.task` -> the new task;
- `scratchpad` -> fresh (iteration counters, cycles, plan, review state,
  context-warning latches — all per-task);
- `_temp` -> empty (per-step scratch);
- `_limits.current_iteration` -> 0 (the budget itself persists unless the
  door overrides it).

What persists (door-owned state):
- `_runtime` in CONTENT (deep-copied) — grants (allowed_tools), prompt slots
  (skills_block / system_prompt_extra), substrate (provider/model/temperature/
  seed), palette, cache identity. The door changes these deliberately, never
  by reset. EXCEPTION: the steering `inbox` is cleared — undelivered guidance
  was addressed to the PREVIOUS task's context, and draining it into the next
  task would amend the wrong task (the same reason hook-layer steering is
  discarded at terminal).
- `context.messages` by default (session continuity across tasks — the
  transcript is the shared memory). `carry_messages=False` starts an empty
  transcript for context-isolation between tasks.

Consecutive-user note: a handoff note plus the init node's task append yields
two adjacent user messages; ReAct merges adjacent user content at the payload
boundary, CodeAct/MemAct pass them through (providers accept both).

The loop's own init node completes the rest on start (iteration seeding, the
new task appended as the trailing user message).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def reset_react_task(
    vars: Dict[str, Any],
    task: str,
    *,
    carry_messages: bool = True,
    handoff_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a NEW vars dict ready to seed the next run with `task`.

    Works for all three loops (they share the context/scratchpad/_temp/_limits
    namespace shapes); named for ReAct because the work door runs the ReAct
    loop. The input dict is not mutated (the completed run's vars remain an
    honest record).

    `handoff_note`: optional context from the door (e.g. a summary of the
    previous task's outcome) appended as a user message with
    `metadata.kind="task_handoff"` — durable, so it survives restarts and
    stays out of the cached prefix's stable head only when messages change
    anyway (a new task always changes the tail).
    """
    new_task = str(task or "").strip()
    if not new_task:
        raise ValueError("reset_react_task requires a non-empty task")

    src: Dict[str, Any] = dict(vars or {})
    out: Dict[str, Any] = dict(src)

    context = dict(src.get("context") or {})
    context["task"] = new_task
    if carry_messages:
        # Copy the message DICTS too, not just the list (wave-E adversary
        # P2-1, 2026-07-14): with runtime.start()'s shallow vars copy and an
        # in-memory store, sharing message dicts meant the next run's live
        # mutations rewrote the completed run's stored record — "the input is
        # not mutated" must hold for the structures a run actually mutates.
        context["messages"] = [dict(m) if isinstance(m, dict) else m for m in (context.get("messages") or [])]
    else:
        context["messages"] = []
    if isinstance(handoff_note, str) and handoff_note.strip():
        from datetime import datetime, timezone

        context["messages"].append(
            {
                "role": "user",
                "content": f"[Task handoff]\n{handoff_note.strip()}",
                # Timestamped like every loop-authored message (MemAct's
                # time-range archival treats timestamp-less content as
                # non-matching).
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"kind": "task_handoff"},
            }
        )
    out["context"] = context

    # Door-owned _runtime persists in CONTENT but is deep-copied (small,
    # JSON-shaped) so the next run's mutations — and the door's own deliberate
    # grant/slot/substrate edits on the returned dict — can never rewrite the
    # completed run's record through an alias (wave-E P2-1; wave-F P3: the
    # one-level copy still shared inbox/allowed_tools lists and palette
    # profile dicts).
    runtime_ns = src.get("_runtime")
    if isinstance(runtime_ns, dict):
        import copy

        new_runtime = copy.deepcopy(runtime_ns)
        # The steering inbox is NOT door-owned state — it is undelivered
        # guidance addressed to the PREVIOUS task's context (wave-F P2:
        # guidance landing during the final cycle's act/observe window would
        # drain into task N+1 as an operator amendment OF THE WRONG TASK).
        # Hook-layer steering is already discarded at terminal for the same
        # reason; the durable inbox gets the same boundary here.
        new_runtime["inbox"] = []
        out["_runtime"] = new_runtime

    # Per-task state resets wholesale. The iteration BUDGET persists: normally
    # in _limits; when a legacy caller carried it only in scratchpad, carry it
    # forward rather than silently widening to the default (wave-E P3).
    new_scratchpad: Dict[str, Any] = {"iteration": 0}
    limits = dict(src.get("_limits") or {})
    old_scratchpad = src.get("scratchpad") or {}
    if "max_iterations" not in limits and isinstance(old_scratchpad, dict) and "max_iterations" in old_scratchpad:
        new_scratchpad["max_iterations"] = old_scratchpad["max_iterations"]
    out["scratchpad"] = new_scratchpad
    out["_temp"] = {}

    limits["current_iteration"] = 0
    # Token accounting is per-task like the warning latches it drives.
    limits["estimated_tokens_used"] = 0
    out["_limits"] = limits

    return out
