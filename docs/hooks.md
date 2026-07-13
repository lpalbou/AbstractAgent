# Loop hooks (listen + steer + capture)

First-class hooks on the ReAct/CodeAct/MemAct loops. One shared contract:
`src/abstractagent/adapters/loop_hooks.py`, exported from the package root.

```python
from abstractagent import LoopHooks, HookEvent, ReactAgent

def my_handler(event: HookEvent):
    # LISTEN / CAPTURE: structured, copied payload — safe to inspect anywhere.
    print(event.run_id, event.name, event.step, event.data)
    # STEER: return an injection; it folds into the DURABLE guidance inbox
    # and reaches the model at the next reason boundary.
    if event.name == "tool_executed" and event.data.get("tool") == "web_search":
        return {"inject": "Prefer primary sources over blogs."}

agent = ReactAgent(runtime=runtime, hooks=LoopHooks().add(my_handler))
```

`hooks=` is accepted by `ReactAgent`, `CodeActAgent`, `MemActAgent`, and the
three workflow factories (`create_react_workflow`, `create_codeact_workflow`,
`create_memact_workflow`). `on_step` continues to work unchanged; both
surfaces can be wired at once.

## Events

Canonical names come from `DEFAULT_EVENT_MAP` (pluggable per
`LoopHooks(event_map=...)`):

| canonical | raw step(s) | fires | loops |
|---|---|---|---|
| `cycle_start` | `reason` | each reasoning cycle begins | all three |
| `tool_proposed` | `parse_tool_calls` | model proposed a tool batch | ReAct (CodeAct/MemAct emit the raw `parse` step instead) |
| `tool_executed` | `observe` | one tool result observed | all three |
| `turn_end` | `done`, `max_iterations` | turn finished; `data.outcome` = `final_answer` \| `iteration_budget` | all three |
| `user_wait` | `ask_user` | loop asked the user | all three |
| `user_response` | `user_response` | user's answer resumed the loop | all three |
| `message_drained` | `inbox_drained` | delivered guidance consumed at the reason boundary (the in-loop "message received" point) | all three |

On ReAct, `turn_end` from `done` also carries `data.handed_off` (true when a
composed host — e.g. an entity visit — continues the run past this turn's
final answer). Treat `data.get("handed_off")` as ReAct-specific.

Unmapped steps pass through under their raw names — the hook set is total,
never a filter. Useful raw steps include `parse_final`, `act`, `act_blocked`,
`allowlist_pruned`, `delegate_agent`, the review/verifier steps, and the
hook layer's own status events below. Every event carries `run_id` and
`agent` (workflow id); `iteration` is set when the underlying payload carries
one (notably `cycle_start` and ReAct's `message_drained`).

### Hook-layer status events

The layer reports on itself through the same stream (raw names, no canonical
mapping):

| step | fires |
|---|---|
| `hook_steer` | a handler's injection was queued for this run |
| `hook_error` | a handler raised, returned an unsupported shape, or steered outside a run context — contained, run unaffected |
| `hook_slow` | a handler call exceeded `slow_budget_s` (a strike) |
| `hook_steer_discarded` | a gracefully completing run dropped undelivered steering (count included) |

## Contract

- **Handlers never mutate loop state.** `event.data` is a deep copy of the
  emit payload (degrading to a shallow copy only if a payload is not
  deep-copyable); the only write channel is the steering return, which rides
  the durable `_runtime.inbox` (same channel as `inject_guidance`) and is
  consumed at the next reason boundary.
- **Return contract (frozen):** `None` or `""` = pure listen; `str` or
  `{"inject": str}` = steer; anything else is reported as `hook_error`.
  Handlers receive only the `HookEvent` — no runtime, agent, or executor
  references.
- **Steering is per-run and at-most-once.** Queued injections are keyed by
  run id (one workflow product serves many runs — delegate children
  included). A gracefully completing run discards undelivered steering
  loudly (`hook_steer_discarded`). Limits to know: an injection queued but
  not yet folded dies with the process, and a run that fails or is cancelled
  does not reach the discard point — its queued steering is simply never
  delivered. Hosts needing guaranteed delivery write the run store's inbox
  directly (`BaseAgent.inject_message`).
- **Failures are contained and observable.** A raising handler never kills
  the run; the failure surfaces as `hook_error` on the flat `on_step` stream
  AND as a pure-listen notification to the handlers themselves (returns
  ignored — no follow-up loops).
- **Slow handlers are benched.** Handlers run synchronously on the tick
  thread. Calls over `slow_budget_s` (default 1s) earn `hook_slow` strikes;
  at `max_slow_strikes` (default 3) the handler is disabled for the
  instance — main dispatches and notifications both stop, other handlers are
  unaffected, and the disable is reported as `hook_error`. Hard preemption
  of a hung handler is not provided (same exposure as a hanging `on_step`).
- **`on_step` is unchanged.** The flat `(step, data)` stream stays as-is.

## Seam (fleet / daemon designs)

Park/wake and cross-process message delivery are runtime-substrate events
(`emit_event` / `events_inbox` / `Runtime.steer` / `inject_guidance`). They
meet this layer when a delivered message lands in `_runtime.inbox`: the loop
fires `message_drained` as it consumes it — subscribe there for "a message
reached the agent's reasoning". Note that the loop's own bounded retry
nudges ride the same inbox, so `message_drained` means "the reasoning cycle
absorbed queued guidance", not necessarily "an external message arrived".
