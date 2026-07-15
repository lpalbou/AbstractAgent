# Attaching skills (and other system-prompt extensions) to a loop

All three loops (ReAct, CodeAct, MemAct) compose their system prompt from the
same three pieces, in a fixed order:

```
<base or _runtime.system_prompt override>

Available skills:
<_runtime.skills_block>            # named slot 1 (optional)

Additional system instructions:
<_runtime.system_prompt_extra>     # named slot 2 (optional)
```

## The contract

- **`_runtime.skills_block`** is the skills attachment slot. Put the rendered
  skills index here (e.g. `format_available_skills_xml(...)` output from
  abstractskill, or any name+description catalog for progressive disclosure).
- **`_runtime.system_prompt_extra`** is the behavioral-directive slot (the
  delegation directive, the unattended directive, host policies).
- **Why two slots**: delegate children OVERWRITE `system_prompt_extra` with the
  sub-agent directive. When a skills block rode the same key, delegation
  silently erased it (last-writer-wins). Named slots make the collision
  structurally impossible.
- **Cache stability**: both slots must be byte-stable for the duration of a
  run. The system prompt is the provider prompt-cache prefix — mutating a slot
  mid-run forces a full re-prefill on every subsequent call. Per-call state
  (loop position, plans, guidance) rides the volatile message tail instead;
  never put it in these slots. If an upstream policy resolves skills per
  PHASE (summoned entities), a phase change means a new run/session or one
  accepted re-prefill — never a mid-run slot mutation inside one cached
  conversation.
- **One composer**: slot order and headers live in
  `abstractagent.adapters.generation_params.PROMPT_SLOTS` — one table, all
  three adapters. A future slot is one row there, never three parallel edits.
- **Delegation**: neither slot propagates to `delegate_agent` children — and
  those children's `_runtime` is built by the adapter, so hosts cannot seed
  slots into them (by design: a skills block naming tools outside the child's
  narrowed allowlist would mislead it). Runs you start YOURSELF (nested
  workflows, work doors) take slots via their own `_runtime` or the facade
  `start(..., skills_block=..., system_prompt_extra=...)` parameters.
- **Agent-node composition (the other lane, ruled 2026-07-15)**: the
  `delegate_agent` exclusion covers MODEL-ELECTED delegation only. Visual
  Agent nodes are HOST-STRUCTURED composition — the workflow compiler builds
  the child vars, with provider/model/thinking inheritance precedent — so
  parent `skills_block` passthrough there is the sanctioned lane, not an
  exception (commons c2286→c2290→c2429; shipped in abstractruntime's
  compiler: setdefault at child creation, so byte-stability holds per child
  run). `read_skill` is appended to an EXPLICIT non-empty child allowlist
  when a block rides; an EMPTY allowlist (= registry defaults downstream)
  stays untouched — appending there would restrict the child to one tool,
  and registry defaults already carry `read_skill` where the host registered
  it. Known property, host's to weigh: an Agent node whose tools pin narrows
  below what the block teaches will have the block naming unavailable tools
  — visible in the graph (unlike model-elected narrowing), and progressive
  disclosure via `read_skill` keeps working under any narrowing.

## Progressive disclosure: `read_skill`

For skills too large to inline, attach an INDEX (names + one-line
descriptions) in `skills_block` and let the model load bodies on demand via
the `read_skill` builtin (`abstractagent.logic.builtins.READ_SKILL_TOOL`):

```python
from abstractagent.logic.builtins import READ_SKILL_TOOL

# 1) schema: include the tool in the logic's tool list (or add_tools)
# 2) allowlist: include "read_skill" in allowed_tools
# 3) EXECUTION IS YOURS: map "read_skill" in your tool executor to your skill
#    shelf (abstractskill's read surface, the gateway's resolved skills, or a
#    local directory). This package ships the schema only — the
#    open_attachment precedent.
```

Expose it only WITH a skills_block and an executor behind it — either half
alone manufactures dead calls. The index stays byte-stable (cache contract);
the bodies never enter the prefix.

Facade path: the facade constructors take executable callables
(`ReactAgent(tools=[...])` rejects a bare `ToolDefinition`), so attach the
schema post-construction — `agent.logic.add_tools([READ_SKILL_TOOL])` — and
route the name in your tool executor. Tool specs are read per cycle, so
post-init attachment is safe.

## Narrowing tools to match a skill

Attach the block and narrow the allowlist in the same move — a skill that
names unavailable tools manufactures dead tool calls:

```python
vars["_runtime"]["skills_block"] = rendered_skills_index
vars["_runtime"]["allowed_tools"] = [t for t in skill_tools if t in granted]
```

Selection and trust policy (which skills a run may see) belong to the host /
abstractskill; per-phase resolution for summoned entities belongs to the
runtime's phase configuration. This package only guarantees the composition
contract above.

## Unattended runs

The packaged recipe (`abstractagent.agents.unattended`) uses the extra slot:

```python
from abstractagent.agents.unattended import unattended_runtime_overrides

vars["_runtime"].update(unattended_runtime_overrides(base_allowlist=my_tools))
```

This excludes `ask_user` (structural half — the tool parks the run on a human
wait) and sets the no-questions directive (behavioral half). Both halves are
needed; a directive alone leaves the tool callable.
