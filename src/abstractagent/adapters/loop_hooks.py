"""First-class loop hooks for the agent adapters (listen + steer + capture).

Maintainer directive (2026-07-12): hooks become a first-class citizen —
(a) LISTEN: observe the agent's actions; (b) STEER: steer the agent DURING
its own actions; (c) CAPTURE: consume every hook programmatically (automated
reprompting, review triggers, memory processes) without parsing prose.

This module is the LOOP half of that contract, shared by the ReAct/CodeAct/
MemAct adapters (one source — the tool_allowlist.py pattern):

- LISTEN rides the adapters' existing emit points (`on_step` vocabulary —
  reason/parse_tool_calls/observe/done/... ). Raw step names are mapped to a
  small CANONICAL event vocabulary (cycle_start, tool_proposed,
  tool_executed, turn_end, ...) via a PLUGGABLE event map, so the room's
  taxonomy can rename events without a second adapter wave. Unmapped steps
  pass through under their raw name — the hook set is total, never a filter.
  Handlers receive a COPY of the emit payload: a handler cannot mutate what
  the loop or `on_step` sees (the listen surface is read-only by mechanism,
  not convention).
- STEER: a handler NEVER mutates loop state directly. A handler may return
  an injection; the dispatcher queues it PER RUN (events carry the run id;
  a queue keyed to nothing would leak steering across runs — the delegate
  child executes the same workflow product), and the adapter folds that
  run's queued injections into the DURABLE guidance inbox
  (`_runtime.inbox`) at the next reason boundary — the same channel
  inject_guidance rides, one mechanism, one drain point. Delivery is
  AT-MOST-ONCE from host memory: an injection queued by a hook and not yet
  folded dies with the process (the folded result is durable; the queue is
  not). Hosts needing guaranteed delivery write to the run store's inbox
  directly (`BaseAgent.inject_message`).
- CAPTURE + containment: handler exceptions are CONTAINED — the run
  survives — and the failure surfaces as a `hook_error` FOLLOW-UP EVENT
  delivered to BOTH surfaces: the flat `on_step` stream and the handlers
  themselves (pure-listen notification: returns ignored, so follow-ups can
  never generate further follow-ups — loop-bounded by construction).

Cross-lane seam (stated for the fleet design): park/wake and cross-process
message delivery are runtime-substrate events (emit_event / events_inbox /
inject_guidance). They meet this layer where a delivered message lands in
`_runtime.inbox`: the adapter fires `message_drained` when the reason
boundary consumes it, which is the in-loop "message received" listen point.
"""

from __future__ import annotations

import copy
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

# Canonical event names for the raw emit steps shared by the three loops.
# PLUGGABLE: hosts may override/extend via LoopHooks(event_map=...); the
# defaults cover the taxonomy's core set. Unmapped steps pass through under
# their raw step name (total coverage, no silent drops).
DEFAULT_EVENT_MAP: Dict[str, str] = {
    "reason": "cycle_start",
    "parse_tool_calls": "tool_proposed",
    "observe": "tool_executed",
    "done": "turn_end",
    "max_iterations": "turn_end",
    "ask_user": "user_wait",
    "user_response": "user_response",
    "inbox_drained": "message_drained",
}

# Terminal steps annotate turn_end with WHY the turn ended.
_TURN_END_OUTCOME: Dict[str, str] = {
    "done": "final_answer",
    "max_iterations": "iteration_budget",
}


@dataclass(frozen=True)
class HookEvent:
    """One observable loop moment, structured for programmatic capture.

    `data` is a private COPY of the emit payload — mutating it affects
    nothing outside the handler.
    """

    name: str  # canonical event name (DEFAULT_EVENT_MAP or raw step)
    step: str  # the adapter's raw emit step (compat vocabulary)
    data: Dict[str, Any]  # copied emit payload (never None)
    run_id: str = ""  # the run this moment belongs to ("" only off-loop)
    agent: str = ""  # workflow id when known
    iteration: Optional[int] = None  # loop iteration when the payload carries one


# A handler may return None or "" (pure listen), a str (guidance injection),
# or {"inject": "..."}. Anything else is reported via hook_error — loudly,
# never silently dropped.
HookHandler = Callable[[HookEvent], Any]


@dataclass
class LoopHooks:
    """Host-side hook registry + per-run steer queues for one workflow product.

    One workflow product can serve MANY runs (re-registration, delegate
    children) — every queue and event is therefore keyed by run id. The
    adapters maintain the run context around node execution (`push_run` /
    `pop_run`, a thread-local stack so nested child runs never inherit the
    parent's identity).
    """

    handlers: List[HookHandler] = field(default_factory=list)
    event_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_EVENT_MAP))
    agent: str = ""
    # Handler TIME BUDGET (plan H-row "sync-on-tick DoS"): handlers run
    # synchronously on the tick thread, so a slow handler stalls the loop.
    # Calls over `slow_budget_s` earn a strike + a loud `hook_slow` event;
    # `max_slow_strikes` strikes DISABLE the handler for this LoopHooks
    # instance (loud `hook_error`). Hard preemption of a HUNG handler is
    # deliberately not provided (same exposure as a hanging on_step — a host
    # bug, not containable without thread abandonment).
    slow_budget_s: float = 1.0
    max_slow_strikes: int = 3
    # Leak backstop (0029 #14): max distinct runs with queued steering; the
    # oldest run's queue evicts loudly past this. Dict insertion order makes
    # the eviction FIFO by first-steer time.
    max_pending_runs: int = 64
    _pending: Dict[str, List[str]] = field(init=False, default_factory=dict)
    _strikes: Dict[int, int] = field(init=False, default_factory=dict)
    _disabled: set = field(init=False, default_factory=set)
    _local: threading.local = field(init=False, default_factory=threading.local, repr=False)

    def __post_init__(self) -> None:
        # Defensive copy: a host passing DEFAULT_EVENT_MAP by reference must
        # not be able to mutate the module constant through this instance.
        object.__setattr__(self, "event_map", dict(self.event_map))

    def add(self, handler: HookHandler) -> "LoopHooks":
        self.handlers.append(handler)
        return self

    # -- run context (adapter-maintained around node execution) ------------

    def push_run(self, run_id: str) -> None:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        stack.append(str(run_id or ""))

    def pop_run(self) -> None:
        stack = getattr(self._local, "stack", None)
        if stack:
            stack.pop()

    def current_run_id(self) -> str:
        stack = getattr(self._local, "stack", None)
        return stack[-1] if stack else ""

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, step: str, data: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Fire all handlers for one emit point; queue steering PER RUN.

        Returns follow-up emits ([(step, data), ...]) for the adapter's flat
        `on_step` stream. Follow-ups are ALSO notified to the handlers here
        (pure listen — returns ignored), so a hooks-only host still observes
        hook_error/hook_steer. Never raises.
        """
        run_id = self.current_run_id()
        payload = data if isinstance(data, dict) else {}
        try:
            event_data: Dict[str, Any] = copy.deepcopy(payload)
        except Exception:  # noqa: BLE001 - uncopyable payloads degrade to a shallow copy
            event_data = dict(payload)
        name = str(self.event_map.get(step, step))
        outcome = _TURN_END_OUTCOME.get(step)
        if outcome is not None:
            event_data.setdefault("outcome", outcome)
        raw_iter = event_data.get("iteration")
        event = HookEvent(
            name=name,
            step=str(step),
            data=event_data,
            run_id=run_id,
            agent=self.agent,
            iteration=raw_iter if isinstance(raw_iter, int) else None,
        )

        follow_ups: List[Tuple[str, Dict[str, Any]]] = []
        for handler in list(self.handlers):
            hid = id(handler)
            if hid in self._disabled:
                continue
            started = time.monotonic()
            try:
                result = handler(event)
            except Exception as e:  # noqa: BLE001 - a listener must never kill the run
                follow_ups.append(
                    (
                        "hook_error",
                        {
                            "event": name,
                            "run_id": run_id,
                            "handler": getattr(handler, "__name__", repr(handler)),
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                )
                continue
            finally:
                elapsed = time.monotonic() - started
                if elapsed > float(self.slow_budget_s):
                    strikes = self._strikes.get(hid, 0) + 1
                    self._strikes[hid] = strikes
                    if strikes >= int(self.max_slow_strikes):
                        self._disabled.add(hid)
                        follow_ups.append(
                            (
                                "hook_error",
                                {
                                    "event": name,
                                    "run_id": run_id,
                                    "handler": getattr(handler, "__name__", repr(handler)),
                                    "error": (
                                        f"handler DISABLED after {strikes} calls over the "
                                        f"{self.slow_budget_s:.2f}s budget (last: {elapsed:.2f}s) — "
                                        "sync handlers run on the tick thread"
                                    ),
                                },
                            )
                        )
                    else:
                        follow_ups.append(
                            (
                                "hook_slow",
                                {
                                    "event": name,
                                    "run_id": run_id,
                                    "handler": getattr(handler, "__name__", repr(handler)),
                                    "elapsed_s": round(elapsed, 3),
                                    "budget_s": float(self.slow_budget_s),
                                    "strikes": strikes,
                                },
                            )
                        )
            if result is None:
                continue
            injection = self._injection_from(result)
            if injection is not None:
                if injection.strip():
                    if run_id:
                        self._pending.setdefault(run_id, []).append(injection)
                        follow_ups.append(
                            ("hook_steer", {"event": name, "run_id": run_id, "chars": len(injection)})
                        )
                        # Structural leak backstop (0029 #14): terminal NODES
                        # discard their run's queue, and the facades discard on
                        # FAILED/CANCELLED — but a host driving the runtime
                        # directly never calls either for a failed run, and its
                        # queue would rot here forever. Cap the number of runs
                        # tracked; evict the OLDEST run's queue loudly. Live
                        # runs drain at every reason boundary, so under any
                        # sane concurrency the evicted queue belongs to a dead
                        # run; a pathological >max_pending_runs live-run host
                        # loses steering LOUDLY, never memory silently.
                        if len(self._pending) > self.max_pending_runs:
                            oldest_run = next(iter(self._pending))
                            dropped = self._pending.pop(oldest_run, [])
                            print(
                                f"[loop_hooks] #FALLBACK pending-steering cap ({self.max_pending_runs} runs) "
                                f"reached: dropped {len(dropped)} undelivered injection(s) for run {oldest_run} "
                                "(likely a failure-terminated run whose host never discarded)",
                                file=sys.stderr,
                            )
                    else:
                        # No run context (off-loop dispatch): dropping silently
                        # would violate works-or-loud; report instead.
                        follow_ups.append(
                            (
                                "hook_error",
                                {
                                    "event": name,
                                    "run_id": "",
                                    "handler": getattr(handler, "__name__", repr(handler)),
                                    "error": "steering returned outside any run context; injection dropped",
                                },
                            )
                        )
                # Empty/whitespace-only returns are an explicit no-op (same
                # meaning as None), documented — not a silent drop of intent.
                continue
            follow_ups.append(
                (
                    "hook_error",
                    {
                        "event": name,
                        "run_id": run_id,
                        "handler": getattr(handler, "__name__", repr(handler)),
                        "error": f"unsupported hook action {type(result).__name__}: {result!r}"[:300],
                    },
                )
            )

        # Deliver follow-ups to the handlers as pure-listen notifications so
        # the hooks-only configuration observes its own failures/steers.
        # Returns are IGNORED (no follow-ups of follow-ups: loop-bounded);
        # a handler raising on its own error notification is contained to
        # stderr (reporting the reporter has to terminate somewhere).
        for fu_step, fu_data in follow_ups:
            fu_event = HookEvent(
                name=str(self.event_map.get(fu_step, fu_step)),
                step=fu_step,
                data=dict(fu_data),
                run_id=run_id,
                agent=self.agent,
            )
            for handler in list(self.handlers):
                if id(handler) in self._disabled:
                    continue  # benched means benched — notifications included
                try:
                    handler(fu_event)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[loop_hooks] handler raised on {fu_step} notification: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
        return follow_ups

    @staticmethod
    def _injection_from(result: Any) -> Optional[str]:
        """The guidance text a handler's return carries, or None when the
        return is not an injection action (caller reports it loudly)."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and set(result.keys()) == {"inject"} and isinstance(result["inject"], str):
            return result["inject"]
        return None

    # -- steer drain (adapter calls at the reason boundary) ---------------

    def drain_pending_injections(self, run_id: str) -> List[str]:
        """Take the named run's queued injections (oldest first). The adapter
        folds them into that run's durable `_runtime.inbox` before building
        the reason payload — hook steering and gateway inject_guidance share
        one drain point, and a run can never drain another run's steering."""
        queue = self._pending.pop(str(run_id or ""), None)
        return list(queue) if queue else []

    def discard_run(self, run_id: str) -> int:
        """Drop any undelivered steering for a finished run (returns count).
        Called by terminal nodes so a dead run's queue can neither leak into
        a later run nor grow host memory."""
        queue = self._pending.pop(str(run_id or ""), None)
        return len(queue) if queue else 0
