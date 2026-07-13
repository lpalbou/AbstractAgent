"""Shared allowlist prune visibility for the agent adapters.

Works-or-loud (agency-caps invariant, 2026-07-11): a host/door granting a tool
the adapter has no definition for must be able to SEE that the grant did not
take. Deny-safe was already true everywhere (unknown names are never offered
nor executable); this module makes the drop visible, identically across the
ReAct/CodeAct/MemAct adapters (one source — the diary_type-clamp drift lesson).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def note_pruned_grants(
    runtime_ns: Dict[str, Any],
    raw: Any,
    normalized: List[str],
) -> Optional[Dict[str, Any]]:
    """Record grant entries normalization DROPPED; return the emit payload.

    Writes `runtime_ns["allowlist_pruned"] = {"dropped", "requested"[, "invalid"]}`
    (a durable run var) when the grant lost names — unknown/unregistered tools
    land in `dropped`; non-name garbage (None, numbers, blank strings) lands in
    `invalid` as reprs (a grant of `[None]` is otherwise a silent FULL deny).

    The note is the record of the LAST prune event and is self-describing
    rather than auto-cleared: normalization stores the pruned list back in
    place, so a later steady-state read cannot distinguish "same grant, already
    pruned" from "host wrote a new clean grant" — a reader whose current grant
    differs from the note's kept set (requested minus dropped) knows the note
    is stale.

    Returns the event payload exactly once per prune event (None on the
    steady-state re-reads that follow), so callers emit without per-cycle noise.
    """
    requested: List[str] = []
    invalid: List[str] = []
    items = raw if isinstance(raw, (list, tuple)) else ([raw] if isinstance(raw, str) else [])
    seen: set[str] = set()
    for t in items:
        if isinstance(t, str) and t.strip():
            name = t.strip()
            if name not in seen:
                seen.add(name)
                requested.append(name)
        else:
            invalid.append(repr(t))
    kept = set(normalized)
    dropped = [name for name in requested if name not in kept]
    if not dropped and not invalid:
        return None
    note: Dict[str, Any] = {"dropped": dropped, "requested": requested}
    if invalid:
        note["invalid"] = invalid
    if runtime_ns.get("allowlist_pruned") == note:
        return None
    runtime_ns["allowlist_pruned"] = note
    return {"dropped": list(dropped), "invalid": list(invalid), "kept": list(normalized)}
