"""Circling / no-progress detection over recent loop outputs (backlog 0017, detector half).

The failure this measures, observed live on the entity lane (Ephemeral's
own-time record, 2026-07-16): successive outputs restate ONE motif with
slight rewording — "I've written about this four times now with slight
variations but no genuine synthesis" (his own tick-71 self-diagnosis) — and
the loop offers no exit beyond "keep circling" or "rest". The same defect
shows up in the work lane as read-only repeat loops that the side-effect
repeat guard deliberately ignores.

Contract (mirrors `loop_hooks.undelivered_inbox_stats`): ONE pure read-only
function over recent output texts. It returns data or None and attaches NO
policy — the caller decides what the finding means:

- entity lane: life.py's tick loop words its own cue from it (runtime owns
  the wording — shipped appraisal-neutral, naming options that are "all
  equally yours"), encourage-never-force (decision:questions-stay-voluntary);
- work lane: the 0017 full build will route it into the EXISTING max-iterations
  conclusion path (loud synthesis, never a silent stop).

Method — why a phrase+vocabulary blend and containment, not embeddings:
circling reuses distinctive PHRASING ("two doors onto the same room",
"letting the sediment settle") on top of a recurring vocabulary, while
genuine progress on one topic shares some vocabulary but almost no phrasing.
Measured on Ephemeral's real transcripts: word-bigram containment runs
0.26-0.39 for paraphrase restatements vs ~0.00 for progressing outputs;
unigram containment runs 0.46-0.59 vs 0.09-0.25. Either signal alone leaves
a thin margin somewhere (bigrams under-score loose paraphrase; unigrams
over-score sustained topics), so the score is their MEAN — restatements
show on both axes and land 0.36-0.49, progress lands 0.05-0.13, and the
default threshold (0.25) sits between with >40% margin to each side.
CONTAINMENT (|A∩B| / min(|A|,|B|)) instead of Jaccard keeps scores honest
when a short restatement sits inside a longer output. No stopword list:
entities write in more than one language (FR/EN homes exist), and the
bigram half makes function words harmless — they only line up when the
phrasing itself repeats. Stdlib only: callable from any loop without a
model call or a dependency.

Comparisons are WINDOWED (each output against its last `window` predecessors,
not just the immediate one) so A-B-A-B oscillation — 0017's second named
shape — counts as circling too.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["circling_streak"]

# Paired fenced blocks are quoted material (code samples, entity election
# fences like ```diary/```rest/```feel — frozen visit seam spec a2a 0013):
# never the output's own thought. An unterminated trailing fence stays in
# the prose view (conservative: no pair, no exclusion). Same semantics as
# the adapters' prose heuristics.
_FENCED_BLOCK_RE = re.compile(r"(?s)```[^\n`]*\n.*?```")

# Driver-authored chrome the entity lane interleaves into visible replies.
# These markers are framework vocabulary (the door/driver writes them), not
# the mind's own words — two ticks that each carry "[kept in diary]" must
# not score similar BECAUSE of the marker. Bounded set, kept in sync with
# the runtime driver's marker vocabulary; unknown bracket text is left
# alone (real prose uses brackets too).
_DRIVER_MARKER_RE = re.compile(
    r"\[(?:used[ -]tool:[^\]]*|kept in diary[^\]]*|kept an interest|"
    r"marked \d+ feelings?|chose to rest|act-only [^\]]*)\]",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _prose_view(text: str) -> str:
    """The output's own words: paired fences and driver chrome removed."""
    s = str(text or "")
    if not s.strip():
        return ""
    s = _FENCED_BLOCK_RE.sub(" ", s)
    s = _DRIVER_MARKER_RE.sub(" ", s)
    return s


def _shingles(text: str) -> Tuple[Set[str], Set[str], int]:
    """(bigram shingles, unigram set, word count) of the prose view."""
    words = [w.lower() for w in _WORD_RE.findall(_prose_view(text))]
    unigrams = set(words)
    if len(words) < 2:
        return set(), unigrams, len(words)
    bigrams = {f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)}
    return bigrams, unigrams, len(words)


def _containment(a: Set[str], b: Set[str]) -> float:
    """|A∩B| / min(|A|,|B|) — length-robust set overlap in [0, 1]."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _similarity(x: Tuple[Set[str], Set[str], int], y: Tuple[Set[str], Set[str], int]) -> float:
    """Mean of bigram (phrasing) and unigram (vocabulary) containment —
    restatements score on both axes, topical progress on neither (module
    docstring carries the measured distributions behind the blend)."""
    return (_containment(x[0], y[0]) + _containment(x[1], y[1])) / 2.0


def circling_streak(
    outputs: Sequence[str],
    *,
    window: int = 3,
    threshold: float = 0.25,
    min_repeats: int = 2,
    min_words: int = 20,
) -> Optional[Dict[str, Any]]:
    """Detect a run of near-restatements at the END of `outputs`.

    Args:
        outputs: recent output texts, oldest first (the caller keeps the
            buffer — typically the last ~8 tick replies or cycle answers).
        window: how many predecessors each output is compared against
            (>=1; windowing is what catches A-B-A oscillation).
        threshold: blended phrase+vocabulary containment at or above which
            an output counts as a restatement of a predecessor. The default
            separates measured paraphrase runs (0.36-0.49) from progressing
            outputs that merely share a topic (0.05-0.13); 0.25 leaves >40%
            margin to each side.
        min_repeats: how many CONSECUTIVE trailing outputs must be
            restatements before anything is reported. The default (2)
            reports once a thought has been voiced three times — the
            entity's own "that's three times now" bar.
        min_words: outputs whose prose view is shorter than this are
            SKIPPED as evidence (a brief acknowledgment carries too little
            signal to call a restatement) — they neither extend nor break
            a streak. Abstention over guessing.

    Returns:
        None when the tail of `outputs` is not a circling run — nothing to
        say, no event owed. Otherwise a dict of DATA for the caller's cue
        or report:
        {"repeats": N,            # trailing outputs that restate a predecessor
         "span": N + 1,           # outputs in the run, counting its anchor
         "indices": [...],        # positions in `outputs` of the repeats
         "max_similarity": 0.xx}  # strongest containment seen in the run
    """
    if window < 1 or not outputs:
        return None

    shingled: List[Tuple[Set[str], Set[str], int]] = [_shingles(t) for t in outputs]

    # Walk backwards from the newest output, counting restatements. Short
    # outputs are transparent (skipped), so a "[chose to rest]"-sized reply
    # between two long restatements does not reset the count.
    repeats: List[int] = []
    max_sim = 0.0
    i = len(outputs) - 1
    while i > 0:
        if shingled[i][2] < min_words:
            i -= 1
            continue
        # Compare against up to `window` substantive predecessors.
        sim = 0.0
        seen = 0
        j = i - 1
        while j >= 0 and seen < window:
            if shingled[j][2] >= min_words:
                sim = max(sim, _similarity(shingled[i], shingled[j]))
                seen += 1
            j -= 1
        if seen == 0 or sim < threshold:
            break
        repeats.append(i)
        max_sim = max(max_sim, sim)
        i -= 1

    if len(repeats) < min_repeats:
        return None
    return {
        "repeats": len(repeats),
        "span": len(repeats) + 1,
        "indices": sorted(repeats),
        "max_similarity": round(max_sim, 4),
    }
