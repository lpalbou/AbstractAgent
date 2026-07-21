# Adversary 1 — credibility audit of untracked/agentic-loops-report.md (2026-07-16)

## Verdict summary
The report should NOT be trusted as evidence; marginally useful as a topic list. Six real
research threads named correctly (Reflexion, Self-Refine, ReWOO, ToT, Meta-Reasoner, the
infinite-loops paper); essentially everything load-bearing on top is fabricated.

## 1a. The 10 "loops"
1. Reflexion — DISTORTED. Paper real (arXiv 2303.11366, Shinn et al., NeurIPS 2023). Repo
   fabricated (`stanford-nlp/Reflexion` 404; real: `noahshinn/reflexion`). "Fixed-depth (2-step)"
   wrong: trial-based iterative retry (up to ~12 trials).
2. Self-Refine — DISTORTED. Paper real (2303.17651, Madaan et al.). Repo fabricated
   (`allenai/self-refine` 404; real: `madaan/self-refine`). "Persistent draft cache" invented.
3. ReWOO — DISTORTED. Paper real (2305.18323, Xu et al.). Repo fabricated
   (`google-research/ReWOO` 404; real: `billxbf/ReWOO`). Misses the actual mechanism
   (Planner→Worker→Solver with #E1..#En variable substitution); invents a "vector DB".
4. Meta-Reasoner — DISTORTED. Paper real (2502.19918 "Dynamic Guidance for Optimized
   Inference-time Reasoning"). Repo + "meta-ai" org fabricated. Real mechanism: progress-report
   summarization + contextual multi-armed bandit picking strategies (backtrack/restart/switch)
   ONLINE during inference — no episodic KV memory, no offline meta-policy.
5. AutoGPT 2.0 — DISTORTED→FABRICATED. Repo real but NO "AutoGPT 2.0" release exists (latest:
   AutoGPT Platform v0.6.66, low-code block platform). Name appears only in SEO content;
   citation (toolify.ai) is a content farm, now 403.
6. Toolformer — DISTORTED. Real paper is arXiv 2302.04761 (Feb 2023, Meta). Cited ID 2205.06366
   is a condensed-matter PHYSICS paper (magnonic Wiedemann-Franz law). Repo fabricated.
   Internal contradiction: "self-supervised tool-use token insertion" AND "no weight update" —
   Toolformer IS a fine-tuning method, training-time, not "on the fly". Not an agentic loop.
7. Tree-of-Thoughts — DISTORTED. Paper real (2305.10601, Yao et al.). Repo fabricated
   (`deepmind/tree-of-thoughts` 404; real: `princeton-nlp/tree-of-thought-llm`).
8. Self-Verification — DISTORTED. arXiv 2405.06682 exists but is "Self-Reflection in LLM
   Agents: Effects on Problem-Solving Performance" (Renze & Guven) — retry-after-reflection on
   MCQ exams, not generate→verify, not titled "Self-Verification for LLMs". Repo fabricated
   (real code: matthewrenze/self-reflection). A real Self-Verification paper exists (Weng et
   al. 2212.09561) but is not what is cited.
9. Event-driven loops — DISTORTED. BoundaryML podcast real; repo fabricated; taxonomy
   attributes invented. A podcast discussion, not a benchmarked loop.
10. AI MEGALoop — FABRICATED as a research object. LinkedIn self-published series; no paper,
    no benchmark, no code (repo 404). Fabrication by elevation.
— "MemAct (2023)" in the exec summary — FABRICATED. No published "MemAct (2023)" loop exists.
  The only published MemAct is "Memory-as-Action" (arXiv 2510.12635, October 2025) — RL context
  curation, unrelated lineage. MemAct is OUR OWN package's loop name; the generator echoed
  project vocabulary back as world history. Most diagnostic fabrication in the report:
  project context leaked into "research findings".

PATTERN: all seven fabricated repos assign real papers to prestige orgs (stanford-nlp, allenai,
google-research, meta-ai, deepmind, openai) — hallucinated citation completion signature.

## 1b. Benchmark table (§2) — every number checked against the real papers
- Reflexion HotpotQA "+7" — FABRICATED (real: +20%; ReAct HotpotQA EM ~27-35%, not 62%).
- Reflexion ALFWorld 48% (+3) — FABRICATED (real: +22 absolute, 130/134 ≈ 97% — its headline).
- Reflexion WebShop 44% (+2) — FABRICATED (paper reports Reflexion FAILED to improve WebShop).
- Self-Refine GSM8K "+6", MATH "+7" — FABRICATED (real: +0.2% math with GPT-4, 0 with GPT-3.5;
  MATH never evaluated; the paper discusses why math self-refinement barely works).
- ReWOO ALFWorld "+12" — FABRICATED (ReWOO never evaluated ALFWorld; the paper names exactly
  this reactive class as where plan-ahead is impractical). Its REAL headline (5x token
  efficiency, +4% HotpotQA) is OMITTED from the table.
- ToT MATH "+9" — FABRICATED (ToT evaluated Game of 24: 4%→74%, Creative Writing, Crosswords;
  never MATH).
- Meta-Reasoner AgentBench "+7" — FABRICATED (real eval: Game-of-24/TheoremQA/SciBench,
  9-12% over SOTA, 28-35% inference-time reduction; not AgentBench).
- "AgentBench (Avg.)" as percentages — FABRICATED (AgentBench overall is a weighted ratio,
  GPT-4=4.01, not percent; none of these methods on its leaderboard).
- AutoGPT 2.0 WebShop "+16" — FABRICATED (no such evaluation exists).
- Toolformer row — FABRICATED (6.7B GPT-J fine-tune; these benchmarks not used; magnitudes
  impossible).
- Provenance footnote (src_008 "AgentBench leaderboard") — FABRICATED (git-stars.org is a
  repo-ranking site, unreachable/403; no such leaderboard).
CONCLUSION: every single number in §2 is untraceable; three cases INVERT the cited paper's own
published finding.

## 1c. Sources: of 18, 6 arXiv IDs resolve to real papers (one to the wrong FIELD); no
quantitative claim survives contact with its own citations. src_006 ("When Agents Do Not
Stop", arXiv 2607.01641, Hou et al. 2026-07-02) is the one genuinely current on-topic primary
source; its use (step caps, loop detection) is legitimate. src_010 (Claude Code harness
patterns) is cited but never used.

## 2. Surviving ideas ranked by real evidence strength
1. ReWOO-style plan-ahead with variable substitution — token economics, not accuracy.
   5x token reduction on HotpotQA at +4%; ~64% average token cut across six benchmarks.
   Mechanism: stop re-feeding the whole transcript between tool calls when the plan doesn't
   depend on observations. Boundary (paper's own): fails in reactive/unknown-state
   environments. Hybrid forms (plan k ahead, re-plan on surprise) are what production adopted.
2. Reflexion verbal self-correction across retries — +22 ALFWorld (130/134), +20 HotpotQA,
   91% pass@1 HumanEval — STRICTLY conditional on a reliable external failure signal;
   WebShop negative result shows it does nothing when failure isn't detectable.
   Mechanism: convert an outcome signal into a targeted prompt delta instead of a blind retry.
3. ToT deliberative branching — 4%→74% Game of 24. Needs a state evaluator; expensive;
   largely superseded by RL-trained reasoning models that internalize search.
4. Self-Refine iterative refinement — ~20% avg across 7 tasks BUT ≈0 on math/logic without
   external ground truth (paper's own analysis). Engineering translation: gate refinement on a
   checkable signal (tests, linters); don't run on faith.
5. Generate-then-verify — real but modest at cited scope; stronger lineage: Weng et al.
   2212.09561, Chain-of-Verification. House caveat: our fleet measurement = in-loop verifier
   +103% tokens for zero delta when an external check exists; adopt only where no external
   verifier exists.
6. Dynamic reasoning-depth control (Meta-Reasoner) — 9-12% accuracy + 28-35% inference-time
   reduction on math/puzzles with a per-domain bandit; unproven on tool-use loops. The
   summary-fed supervisor idea is interesting for long loops; bandit machinery research-grade.
7. Event-driven/async observation handling — architecture pattern, no accuracy evidence; we
   already run it (event-inbox resident, durable mailbox + cursor drain).
8. AutoGPT-style hierarchical decomposition — weakest survivor; evidence-backed versions are
   plan-and-execute variants, not AutoGPT.

Checklist triage: sound-generic #1/#4/#6 (we exceed #6 with ledgers); #5 sound with a real
citation (step caps / loop detection, src_006's actual topic); #2 "use LangChain
StructuredTool" = stack ad; #3 "attach FAISS/Chroma" = no retrieval policy; §3.3 ecosystem
claims mostly false.

## 3. What a real July-2026 survey should have covered (missing entirely)
- Plan-and-Execute / Plan-and-Solve (2305.04091; LangGraph) — the practical planner/executor split.
- LLMCompiler (2312.04511) — parallel tool DAG with dependency-resolved fan-out; ReWOO's successor.
- LATS (2310.04406) — MCTS over ReAct trajectories unifying search+reflection+env feedback.
- Process reward models / verifier-guided search (2305.20050; Best-of-N with PRMs) — the
  strongest verification line.
- RL-trained reasoning models (o1/o3/R1-class, RLVR, thinking budgets) — the biggest 2024-26
  shift; deliberation moved INSIDE the model, reframing when explicit ToT/reflection pays.
  A July-2026 survey silent on this is disqualifying.
- Multi-agent patterns: debate (2305.14325), AutoGen, MetaGPT, CAMEL.
- Context compaction / memory management (MemGPT/Letta, Claude Code auto-compaction, the REAL
  MemAct 2510.12635).
- Computer-use loops (Anthropic computer use, OpenAI Operator/CUA).
- CodeAct (2402.01030) + SWE-agent ACI (2405.15793) — named in the exec summary, never analyzed.
- Durable execution / crash-resumable loops — first-order production concern in 2026.
- Prompt-cache-aware loop design (append-only transcripts, stable prefixes).
- Industry harness taxonomies (Anthropic "Building effective agents"; Claude Code patterns).

## 4. Trust verdict
Not as evidence. Use as a keyword list to independently re-research; read the six verified
papers + arXiv 2607.01641 in the original; never cite the report; numbers radioactive.
