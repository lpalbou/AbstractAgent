# Agents

Related:
- [`docs/getting-started.md`](getting-started.md)
- [`docs/tools.md`](tools.md)
- [`docs/persistence.md`](persistence.md)
- [`docs/architecture.md`](architecture.md)

AbstractAgent ships three agent patterns implemented as **(logic → runtime workflow → API wrapper)**:

| Agent | Best for | Key behavior | Entry points |
|---|---|---|---|
| **ReAct** | Tool-first workflows | Loops until the model returns **no tool calls** | `abstractagent.agents.react.ReactAgent`, `create_react_agent()` |
| **CodeAct** | Python-centric tasks | Python via `execute_python`; fenced ` ```python ... ``` ` extracted on prompted-tools models | `abstractagent.agents.codeact.CodeActAgent`, `create_codeact_agent()` |
| **MemAct** (experimental) | Memory-enhanced sessions | Uses runtime-owned `active_memory` blocks | `abstractagent.agents.memact.MemActAgent`, `create_memact_agent()` |

Note: `MemActAgent` / `create_memact_agent` are **not** re-exported at the package top-level. Import them from
`abstractagent.agents.memact` (see [`docs/api.md`](api.md)).

## ReAct (`ReactAgent`)

Files:
- Workflow: `src/abstractagent/adapters/react_runtime.py` (`create_react_workflow`)
- Logic: `src/abstractagent/logic/react.py` (`ReActLogic`)
- API: `src/abstractagent/agents/react.py` (`ReactAgent`, `create_react_agent`)

Typical usage:

```python
from abstractagent import create_react_agent

agent = create_react_agent()
agent.start("Search for TODOs and summarize what needs fixing")
state = agent.run_to_completion()
print(state.output["answer"])
```

Notes (code reality):
- ReAct persists its **loop trace** under `vars["scratchpad"]["cycles"]` (not as assistant “thought” messages).
- ReAct disables runtime-level trimming and sends the full `context.messages` window by default.
- `ReactAgent(..., review_mode=...)` is honored and defaults **on**: the ReAct adapter runs a verifier round on final answers
  (`maybe_review`/`review` nodes), and a failed verifier call degrades to accepting the held answer with a
  loud `#FALLBACK` marker (never a dead run). Pass `review_mode=False` to opt out. `plan_mode` is stored
  under `vars["_runtime"]` but the ReAct adapter does not read it — plan management rides the
  always-available `update_plan` tool and the `check_plan` gate instead.
- CodeAct's fenced-code fallback is **capability-conditional**: the default is
  `not supports_native_tools` (the bit the runtime seeds per run from model capabilities) — native-tools
  models neither get taught the ```python fence nor have prose fences extracted (teaching a channel that
  competes with the trained tool_calls channel manufactured zero-tool fabrication); prompted models keep
  the fence taught + extracted. Explicit `_runtime.codeact_fenced_fallback` wins both ways; an absent
  capability bit fails safe (fence ON). One resolver (`CodeActLogic.fenced_fallback_enabled`) drives both
  the prompt line and the parser gate so they can never disagree.
- `create_react_agent(tools=None)` defaults to `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/agents/react.py`).

## CodeAct (`CodeActAgent`)

Files:
- Workflow: `src/abstractagent/adapters/codeact_runtime.py` (`create_codeact_workflow`)
- Logic: `src/abstractagent/logic/codeact.py` (`CodeActLogic`)
- API: `src/abstractagent/agents/codeact.py` (`CodeActAgent`, `create_codeact_agent`)

Behavior highlights:
- If the model emits a fenced Python block AND the fenced channel is enabled (`codeact_fenced_fallback` — default: enabled only on prompted-tools models), the adapter executes it via a `TOOL_CALLS` effect targeting `execute_python`. Routed runs: the capability bit describes the model it was DERIVED from, not necessarily the run's `_runtime.model`. The facade stamps `_runtime.tool_support_model` at start; when the effective model differs from the stamp, the resolver distrusts the bit and fails toward fence ON (a prompted model without the fence completes silently with code-as-prose — the worse failure). Hosts that route runs to different models without that stamp should pin `codeact_fenced_fallback` explicitly per run.
- Optional `plan_mode` and `review_mode` are implemented for CodeAct in `src/abstractagent/adapters/codeact_runtime.py`.
- `create_codeact_agent(tools=None)` defaults to `[execute_python]` (`src/abstractagent/agents/codeact.py`).

## MemAct (`MemActAgent`) — EXPERIMENTAL

**Status: experimental.** MemAct is kept because it is a genuinely distinct
architecture (Letta-style per-turn memory rewriting; the memory machinery
lives runtime-side and is separable), but it has one opt-in consumer and less
production mileage than ReAct/CodeAct. It is deliberately NOT re-exported at
the package top level — the deep import is the experimental label made
structural. Expect its surface to move faster than the siblings'.

Files:
- Workflow: `src/abstractagent/adapters/memact_runtime.py` (`create_memact_workflow`)
- Logic: `src/abstractagent/logic/memact.py` (`MemActLogic`)
- API: `src/abstractagent/agents/memact.py` (`MemActAgent`, `create_memact_agent`)

MemAct relies on the runtime’s active memory subsystem:
- the workflow ensures memory exists (`abstractruntime.memory.active_memory.ensure_memact_memory`)
- memory blocks are injected into the system prompt and updated via structured steps
- `create_memact_agent(tools=None)` defaults to `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/agents/memact.py`).
- MemAct has **no** plan or review nodes. `plan_mode`/`review_mode`/`review_max_rounds`
  are deprecated and ignored: non-default values emit a `DeprecationWarning`,
  legacy defaults are ignored silently, and the parameters will be removed in a
  future release. A MemAct verifier is a deliberate non-goal while
  MemAct stays experimental.

## Replies that announce tools without calling them (all three loops)

A reply with no tool call used to be the final answer. Two kinds of such replies are not answers, and
ReAct, CodeAct and MemAct now handle them the same way
(`src/abstractagent/adapters/announced_calls.py`):

- **Announced**: a short reply (under 300 characters of prose) whose LAST sentence announces work a tool would
  do: first-person intent plus a tool verb ("I have strong material. Let me verify a couple of key specifics
  before writing the digest.", "I'm going to grep …") or an action statement ("Checking the remaining two
  sources now.", "Proceeding to fetch …"). Qwen3.x on MLX often ends a step this way. Never treated as
  announcements: questions; replies that wait on the user ("Let me know if…", "… unless you say otherwise",
  "once you confirm …"); intent that talks rather than acts ("Let me summarize: …", "Next, I recommend …");
  refusals ("I will not run that…"); replies that start by delivering ("Done.", "Here is…"); structured answers;
  replies where the intent is not the last sentence.
- **Unrunnable**: tool-call markup that did not become a call: a call cut off mid-parameter
  (`metadata.unparsed_tool_call`, read first), a tool name that was not offered or calls drafted inside the
  thinking block that AbstractCore could not recover (its warnings; Core has no structured code for these yet,
  so a contract test generates them from Core itself), or bare markup in the reply or, when there is no visible
  reply, in the thinking block. This check only applies when there is no visible answer, or when the visible
  answer is shorter than 300 characters.

What happens:

1. **One re-prompt.** The assistant turn is kept as the provider returned its visible content, and the next
   user message QUOTES the failed reply verbatim (reasoning included, in `<think>…</think>`, inside a fenced
   block), then asks the model to call the tools now, in the required format, as its visible reply, or to answer
   directly. The quote is in the user message on purpose: Qwen3.5/3.6 chat templates strip `<think>` from
   assistant turns before the last user message, so a history turn alone reaches the model empty. The
   re-prompt call carries `_runtime_observability: {"reprompted": "<reason>", ...}` on its `LLM_CALL` effect
   payload, so the ledger records it. Emits `parse_reprompt` (`reason`: `announced_tool_use` |
   `unrunnable_tool_call`). Transcript messages are tagged `metadata.kind`: `reprompted_reply` and `reprompt`.
2. **Second failure.** If the re-prompted reply fails the same way, the step ends with a visible error
   (`parse_reprompt_failed`) and the reply is never published. ReAct goes to its conclusion path (a tool-free
   call that answers from what the run already has); CodeAct and MemAct go to their budget terminal and answer
   with the error text. All three set `output.stop_reason.code = "no_tool_call"` and `output.no_tool_call_stop`;
   ReAct also adds an `error` notice and a report line.
3. **No iteration left.** No re-prompt (`parse_reprompt_skipped`). ReAct's conclusion path answers (a conclusion
   reply that is itself an announcement is dropped, `conclusion_announcement_dropped`); CodeAct and MemAct end
   with the error and `stop_reason.code = "no_tool_call"`. Their budget terminal never publishes a turn that
   carried tool calls (pre-call narration such as "Searching.").

Long replies: ReAct's older followthrough check (`_looks_like_deferred_action`, any length) stays a separate
soft nudge (`parse_retry_plan_only` with `soft: true`). It never ends a step with an error, so a real digest
that says "Let me list the three risks…" is not turned into a failure.

Why quote verbatim: re-prompting with the reply as the runtime records it (content `""` when the calls sat in
the thinking block) made the model believe its tools had already run; with the raw reply it re-issued the same
calls as visible calls (framework mission XP, 2026-09-26).

Controls (independent):
- `_runtime.check_plan`: the announcement check. Default on in task lanes, off in visit lanes
  (`suppress_loop_tail`), where "I will read that entry again" is musing.
- `_runtime.check_unrunnable_calls`: the unrunnable check. Default on in every lane. Turning `check_plan` off
  does not re-enable publishing tool markup.

Known limit: announcement detection is English only (backlog 0033). A French, German or Spanish announcement
still ends the run as its answer; unrunnable markup is caught in any language.

Calls that AbstractCore **did** recover from the thinking block (`metadata.tool_calls_from_reasoning`, Core
2.16.0+) run normally. ReAct shows the count in the `parse` (`tool_calls_from_reasoning`) and
`parse_tool_calls` (`from_reasoning`) events, the cycle entry, the report, and an `info` notice
(`code: "tool_calls_from_reasoning"`). CodeAct and MemAct add `from_reasoning` to `parse_tool_calls`.

## Common API and output contract

All agents inherit `BaseAgent` (`src/abstractagent/agents/base.py`):
- `start(task, ...) -> run_id`
- `step() -> RunState`
- `run_to_completion() -> RunState`
- `attach(run_id)` / `save_state(path)` / `load_state(path)`
- `cancel()` / `get_ledger()` / `inject_message(...)`

On completion, the workflow returns `state.output` with (at minimum):
- `answer`: final assistant answer (string)
- `iterations`: how many loop iterations ran (int)
- `messages`: the durable conversation transcript (list of message dicts)

ReAct also returns:
- `report`: a deterministic “what happened” summary derived from the scratchpad
- `scratchpad`: the full scratchpad dict (including `cycles`)
