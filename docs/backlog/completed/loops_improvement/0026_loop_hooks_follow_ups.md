# Completed: loop-hooks follow-ups from the adversarial review (hooks wave)

## Metadata
- Created: 2026-07-12
- Status: Completed
- Completed: 2026-07-15
- Proposal ID: (hooks-wave follow-ups; pre-dates the A-D set)

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The 2026-07-12 loop-hooks ship (adapters/loop_hooks.py + wiring in all three
adapters) closed the adversaries' P1s (cross-run steer leakage, follow-up
invisibility, slow-handler DoS). These are the named P2/P3 leftovers.

## The follow-ups
- Emit `parse_tool_calls` on CodeAct/MemAct so `tool_proposed` fires on all
  three loops (today ReAct-only).
- A real mutation pin against the `act` payload (handlers get copies; one test
  asserts payload copy depth on the hot path).
- Discard queued steering on run FAILURE/cancel, not only at clean terminals
  (today only done/max_iterations discard).
- A ReAct run-init emit (siblings emit an init step; ReAct starts at reason).
- The silent conclude-phase drain: injections arriving during the forced
  conclusion cycle are folded but can no longer influence — either surface
  or discard loudly.

## Promotion criteria
Next hooks build wave, or the first external hook consumer hitting one.

## Non-goals
New hook vocabulary; async handler execution (rejected in design — sync +
budget is the contract).

## Completion report
- Date: 2026-07-15
- Outcome, per follow-up:
  - **parse_tool_calls on CodeAct/MemAct**: both sibling parse nodes emit the
    same raw step + `{"count": N}` payload at the tool-batch commit point —
    canonical `tool_proposed` now fires on all three loops. CodeAct's
    fenced-code path deliberately does NOT fire it (no tool batch; the
    documented `parse.has_code` signal carries that path). `parse_tool_calls`
    moved to `COMMON_STEPS` in the inventory.
  - **Act-payload mutation pin**: the existing pin was found VACUOUS — it
    keyed `tool_calls` off `tool_proposed`, whose payload only carries
    `count`, so its mutation never touched anything. Rewritten as a vandal
    handler corrupting EVERY mutable value on EVERY event; execution and the
    durable transcript must stay pristine (the `act` emit's `args` shares the
    live `arguments` dict with the TOOL_CALLS effect payload — the exact leak
    the dispatch deepcopy exists for).
  - **Steering discard on FAILURE/cancel**: executed earlier as 0029 #14
    (facade discard + `max_pending_runs` structural backstop); noted here for
    the record.
  - **ReAct run-init emit**: `init` (payload `task`) now fires on all three
    loops once at workflow entry; the founding "ReAct emits no init"
    asymmetry is CLOSED (composed visit turns re-enter at `reason`, so it
    stays a RUN moment). `init` moved to `COMMON_STEPS`.
  - **Conclude-phase drain honesty**: the silent half was durable-inbox
    guidance landing AFTER the loop's last drain (e.g. `inject_guidance`
    while the final/conclusion LLM call is in flight) — the run completed
    over it with no signal. All three loops' TRUE terminals now emit
    `inbox_undelivered` `{count, chars}` when `_runtime.inbox` is non-empty,
    WITHOUT consuming the entries (the durable record keeps what never got
    delivered). Guidance drained at the conclusion boundary still influences
    (rides the conclusion prompt as "Host guidance:") — both halves pinned on
    one run. Composition handoffs (`final_next_node`) deliberately do not
    fire it: the continuing run drains at its next reason boundary
    (pinned negative).
- Validation: `tests/test_loop_hooks_0026_follow_ups.py` (5 tests: 3-loop
  init+tool_proposed parity, 3-loop terminal undelivered honesty, ReAct
  conclusion both-halves, composition-handoff negative) + the rewritten
  mutation pin in `tests/test_loop_hooks.py`. Emit inventory + drift test
  updated (`init`/`parse_tool_calls`/`inbox_undelivered` common). Full suite
  233 passed / 2 skipped.
- Follow-ups revealed: none. (0028 remains the open hook-taxonomy item;
  nothing here changed its gate.)
