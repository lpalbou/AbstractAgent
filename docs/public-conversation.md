# Public conversation

ReAct hosts can opt into short public updates while an agent continues working.
The agent writes these messages in its ordinary continuation after reading tool
results. Publishing an update adds a durable event, with no additional inference
request. It does add output tokens when the model chooses to communicate.

Enable the feature before starting the run:

```python
run_id = agent.start("Help plan an outing.", conversation={"enabled": True})
```

Raw-workflow hosts set `vars["_runtime"]["conversation"] = {"enabled": True}`
before the first reasoning request. The value must be the boolean `True`.
The setting is frozen for the turn. Composed entity visits use the same ReAct
adapter and reset that snapshot with `reset_react_turn` at their existing bridge.
Hosts retain their existing entity message/authority boundaries.

The default is off. Disabled runs retain their original system prompt and tool
schemas. Enabled runs use a stable instruction and append-only transcript;
the growing list of updates is not injected into the system prompt. Cache reuse
depends on the backend and should be measured separately from request stability.

## Public updates

The model may start its explicit public `content` with one block:

```xml
<public_update>{"text":"The museum lists parking nearby; I still need to check the hours.","source_call_ids":["read-1"]}</public_update>
```

Its normal tool calls or final answer accompany that block. A normal final answer
requires no block. Updates should communicate a meaningful discovery, a reason
affecting a choice, or uncertainty in the person's language. No update is needed
for routine tool completion.

Only a leading block in explicit public content is eligible. Private reasoning,
tool results, quoted examples and code-fenced examples are never mined for updates.
Text is limited to 600 characters, the block to 4,096 characters and source IDs to
16. Source IDs must identify results already observed. Malformed, oversized,
unknown-source, cancelled or truncated updates are rejected rather than clipped;
`public_update_rejected` reports a code without repeating the rejected text.
That code is retained in the cycle and a durable event for technical replay.
The usual tool-call and final-answer contracts still apply.

Accepted updates produce an `EMIT_EVENT` effect named `public_update`, containing:

```json
{
  "id": "run-id:public_update:2",
  "text": "The museum lists parking nearby; I still need to check the hours.",
  "run_id": "run-id",
  "session_id": "session-id",
  "iteration": 2,
  "source_call_ids": ["read-1"]
}
```

The update also lives in `scratchpad.cycles[*].public_update` and appears through
[loop hooks](hooks.md). The ID uses the absolute cycle position, so it stays unique
when a composed visit resets its per-turn iteration. Read completed durable events
for replay. Hosts can attach a reply to the update ID and use their established
message/steering interface; publishing a message grants no execution authority.

Source identity establishes lineage, not truth or entailment. A web lead may still
need verification, and a command printing a claim does not establish that claim.
Updates do not expose private reasoning or replace the original source records.

## Questions and silence

When opted in, the `ask_user` schema additionally accepts `importance`, `timeout_s`
and `fallback`. The required `question` string and optional `choices` stay the same.
The adapter forwards the optional policy as `ASK_USER.payload.question`:

```json
{
  "importance": "clarification",
  "timeout_s": 180,
  "fallback": "Keep both options and compare their parking."
}
```

An `essential` question waits indefinitely and has no timeout or fallback.
A timed `clarification` waits 120–300 seconds and requires a previously declared,
nonempty fallback. The Runtime question handler validates and enforces that policy;
hosts must use a Runtime implementation supporting conversational questions.
Omitting the policy preserves the existing indefinite `ASK_USER` behavior.

On expiration, the Runtime returns `timed_out: true`, `fallback`, and
`question: {id, outcome: "expired"}`, without a `response`. ReAct emits
`question_expired` and appends an explicit host instruction explaining that the
person did not answer or approve anything. A genuine answer retains its question
identity and is recorded as user input. Silence never supplies approval.

See [API](api.md), [hooks](hooks.md) and [persistence](persistence.md).
