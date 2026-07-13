# Recurrent: backlog hygiene

## Metadata
- Created: 2026-07-12
- Status: Recurrent
- Completed: N/A (runs repeatedly)

## Purpose
Keep filenames, global numbering, overview counts, and lifecycle states
truthful.

## Run conditions
After any item add/move/close; otherwise opportunistically.

## Checklist
- [ ] Every item file matches `NNNN_slug.md`; numbers globally unique.
- [ ] overview.md counts and ledgers match the directories.
- [ ] Stale links after moves fixed.
- [ ] Items whose code-reality drifted are patched or flagged.

## Expected output
A consistent overview.md; renames/flags for non-compliant files.

## Non-goals
Rewriting item history; deleting items (move to deprecated/ instead).
