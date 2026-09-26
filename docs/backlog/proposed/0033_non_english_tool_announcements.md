# Proposed: Detect tool announcements in languages other than English

## Metadata
- Created: 2026-09-26
- Status: Proposed
- Completed: N/A

## Context

Since the mission AGX fix, a short reply that ENDS on an announcement of tool work ("Let me verify ...",
"Checking the remaining two sources now.") is re-prompted once instead of being published as the answer
(`src/abstractagent/adapters/announced_calls.py`, `looks_like_tool_announcement`).

## Problem (documented limit)

The detector is English vocabulary. Review 28 (framework `untracked/missions-2026-09-25/REVIEW/28-agent-reprompt.md`,
D7) measured 0 of 5 non-English announcements detected ("Je vais vérifier …", "Ich werde …", "Voy a buscar …",
"Permettez-moi …"). The operator's models answer in the user's language, so a French run that stops on
"Je vais vérifier le prix du pétrole." still completes with that sentence as its answer. Unrunnable tool markup is
language-independent and is caught in every language.

## Proposed Direction

Detect by STRUCTURE rather than vocabulary: a short reply that follows tool results, ends a step with no call,
and delivers no content (no numbers/claims beyond the task restatement), with per-language intent lists only as a
secondary signal. Measure false positives on a multilingual final-answer set before enabling.

## Promotion gate

A multilingual evaluation set (announcements + genuine finals) with 0 false positives.
