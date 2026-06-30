# Token Waste in Frontier Claude Coding Agents — Research Summary

*A research summary of where the dominant token waste lies in frontier Claude agentic coding. It starts from the original question — how often "incorrect tool calls" and "tool-call loops" appear in real session traces — defines those failure modes, then shows why tool-call loops are no longer the main problem in frontier Claude, and finally establishes that the real, measurable waste has shifted to **redundant file re-reads / context accumulation.** Every paragraph cites its source with clickable links.*

---

## 1. The original focus, and why it shifted

This investigation began with a narrow question: how many failures of two types appear in the real session traces of frontier models and agents —

- **incorrect tool calls**
- **tool-call loops**

The evidence reviewed below moves the focus away from those two categories. In short: tool-call loops have been **engineered down to a low-single-digit tail event** in frontier Claude, and incorrect/failed calls are common but cheap and recoverable. The **dominant token waste** in current Claude agentic coding is **context accumulation driven by redundant re-reading (and re-editing) of the same files**, which inflates *input* tokens — and input tokens, not reasoning/output tokens, are what dominate agentic cost ([Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750)).

The target is therefore redefined: away from "identical consecutive tool calls" and toward **redundant file-access / context bloat** — the measurable, high-value signal in the traces.

> **Headline finding:** *The repetition intuition is correct, but the location is not. The waste is not frozen doom-loops — it is the same files being read repeatedly and re-billed into a growing context.*

---

## 2. Definitions: what each failure mode is

### 2.1 Tool failures vs. incorrect tool calls

- **Tool failure** = a *valid* call that fails at execution time. Agent-Diff's definition: "valid tool calls that fail during execution: Bash syntax errors, runtime exceptions (NameError, ImportError), logic bugs, or environment misconceptions." *Source: [Agent-Diff, arXiv:2602.11224](https://arxiv.org/abs/2602.11224).*
- **Incorrect tool call** = the agent *builds the call wrong*, regardless of whether the tool would have worked. Captured by the BFCL error categories: `instruction_alignment_failure` (no parseable JSON), `wrong_func_count`, `wrong_func_format`, `hallucinated_func_name` (a function not in the tool list), `wrong_func_name` (wrong tool picked), `missing_required_parameter`. *Source: [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html).*

### 2.2 What a tool-call loop is

A **tool-call loop** is repeated, non-progressing invocation: calling the same tool with identical arguments, re-reading the same files, retrying the same failing test/edit, or oscillating between two states. Synonyms in the literature: **step repetition** (MAST), **recursive tool invocation**, **"doom loop,"** and degenerate **"Ralph loops."** The Ralph loop itself (Geoffrey Huntley, 2025) is a deliberate `while true` pattern that re-runs an agent with fresh context against a persistent on-disk workspace; it works when bounded but becomes a "token burn machine" when it isn't. Both OpenAI's Codex CLI (`/goal`) and Claude Code (`/loop`) have since shipped first-class versions of this pattern with built-in token budgets and termination checks. *Sources: [MAST / "Why Do Multi-Agent LLM Systems Fail?", arXiv:2503.13657](https://arxiv.org/abs/2503.13657); [REST-API tool taxonomy, arXiv:2504.15546](https://arxiv.org/abs/2504.15546); Ralph loop — Geoffrey Huntley (2025) via [Braintrust, "Debugging Ralph Wiggum"](https://www.braintrust.dev/blog/ralph-wiggum-debugging).*

**Why coding agents loop:**

- **Identical inputs each turn.** "The model is doing exactly what you'd expect given identical inputs every turn." If the conversation doesn't reflect prior failure, the model repeats it — tool-use loops are "not a model problem so much as a harness problem." *Source: [DEV Community, "How to Fix Tool-Use Loops in Autonomous Coding Agents"](https://dev.to/alanwest/how-to-fix-tool-use-loops-in-autonomous-coding-agents-540e).*
- **Uninformative error messages** (e.g., "Bad request," "Missing body") give the agent nothing to adapt on. *Source: [REST-API tool taxonomy, arXiv:2504.15546](https://arxiv.org/abs/2504.15546).*
- **Tool output that re-triggers the tool.** The opencode incident: every `TodoWrite` call returned the full todo state, prompting another `TodoWrite` — infinite loop, ~$250 burned overnight, zero useful code. Fix: "stop reflecting full state back to the model on every tool call. Return minimal confirmation instead." *Source: opencode incident writeup, via [DEV Community](https://dev.to/alanwest/how-to-fix-tool-use-loops-in-autonomous-coding-agents-540e).*
- **Lossy compaction / context loss** drops the detail that would tell the agent it already tried X. *Sources: [LangChain, "Context Engineering for Agents"](https://www.langchain.com/blog/context-engineering-for-agents); [Anthropic, "Effective context engineering"](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).*
- **Failure to recognize termination** (MAST mode 1.5, ~12.4% of multi-agent failures). *Source: [MAST, arXiv:2503.13657](https://arxiv.org/abs/2503.13657).*
- **Session-poisoning bugs.** A malformed tool-call JSON left in message history can cause every subsequent turn to fail with HTTP 400 forever until the session is cleared (documented in kimi-cli and LiteLLM/Bedrock). *Source: GitHub issues — kimi-cli, LiteLLM/Bedrock; [community/discussions #185278](https://github.com/orgs/community/discussions/185278).*

### 2.3 Do loops/failures *reduce* token usage? No — they multiply it

- **Quadratic re-billing.** Naive agent loops "compound token costs at O(N²) because LLM APIs bill for the entire conversation history on every call." A 20-step loop generating 1,000 tokens/step produces **~210,000 cumulative input tokens, not 20,000** — the N(N+1)/2 triangular series. A 10-turn loop can send ~50× the tokens of a single linear call; "loop length, not model choice, is usually what determines cost and latency." *Source: [Augment Code, "AI Agent Loop Token Costs"](https://www.augmentcode.com/guides/ai-agent-loop-token-cost-context-constraints).*
- **Tool responses dominate tokens.** A 30-day instrumentation study found "67% of the tokens my agent consumed were completely waste," driven by tool-output flooding, repetitive system prompts, and uncompressed history. Braintrust notes a single production trace can hit a million tokens, and "a retrying tool call, oversized retrieval step, or sub-agent loop can dominate total token usage even when the top-level trace looks normal." *Sources: 30-day instrumentation blog study; [Braintrust](https://www.braintrust.dev/blog/ralph-wiggum-debugging).*
- **Diagnostic (Steve Kinney):** watch whether token consumption grows **linearly** (healthy — each turn adds a roughly constant amount) or **quadratically** (the full conversation is re-sent each turn without compaction — this blows the budget fast). *Source: [Steve Kinney, "The Anatomy of an Agent Loop"](https://stevekinney.com/writing/agent-loops).*

### 2.4 Which is worse for token usage: tool failures or tool-call loops?

**Loops are far worse, and the reason is structural.** The two failures have fundamentally different cost shapes.

- A **single failed/incorrect call has bounded cost**: one error message, one wasted round-trip. Recovery exceeds **85%** when the error is structured (compiler/test output) and drops to **~17%** only when the error is ambiguous. *Source: "Beyond Binary Correctness" Figma-to-code transcript study.*
- A **loop has unbounded, compounding cost**: it isn't N wasted calls — it's the **triangular sum of everything before it, re-billed every iteration.** *Sources: [Augment Code](https://www.augmentcode.com/guides/ai-agent-loop-token-cost-context-constraints); [Steve Kinney](https://stevekinney.com/writing/agent-loops).*
- **A loop is usually just a tool failure the agent couldn't recover from, repeated.** They aren't independent categories — the loop is the *tail risk* of the failure. *Source: synthesis across [MAST, arXiv:2503.13657](https://arxiv.org/abs/2503.13657) and the failure-taxonomy literature.*
- **Per incident, loops win by orders of magnitude.** ClawsBench documented GLM-5 running a 137-step loop of identical failing calls (`message({"command": "message"})`, each returning "Action send requires a target"), with zero argument adaptation. A single failure costs cents; a runaway loop costs dollars to hundreds of dollars. *Source: [ClawsBench, arXiv:2604.05172](https://arxiv.org/abs/2604.05172); [Braintrust Ralph trace](https://www.braintrust.dev/blog/ralph-wiggum-debugging).*
- **Second-order damage:** loops poison the context window (low-signal repetition → model gets "dumber" → loops more). Failures don't have this runaway property unless they accumulate into loops. *Source: [LangChain, "Context Engineering for Agents"](https://www.langchain.com/blog/context-engineering-for-agents).*

**Verdict:** failures are *frequent but cheap and recoverable*; loops are *rare but catastrophically expensive and self-reinforcing.* Every mature harness puts a hard ceiling on loops (wall-clock timeout ~300s, a token/cost budget e.g. ~$2.00/run, loop detection via fingerprinting). *Source: [Steve Kinney, "The Anatomy of an Agent Loop"](https://stevekinney.com/writing/agent-loops).*

---

## 3. Why tool-call loops are not the problem anymore

Pulling every loop-prevalence study together: pathological repetitive tool-call loops are now **rare in frontier Claude** and are largely a **weaker/open-model phenomenon.** They persist as a **tail risk** on long-horizon work and in individual production sessions, but they are not the main cost center.

- **SWE-smith — the cleanest comparative baseline.** Running SWE-agent with Claude 3.7 Sonnet vs. a fine-tuned open model on SWE-bench Verified: **>25% of the open SWE-agent-LM-32B trajectories had a repetitive sequence ≥ length 10, versus < 4% for Claude 3.7 Sonnet**, and a length-10 repetition carried an **89% failure probability** — the model keeps issuing similar commands until the cost or turn limit terminates the run. *Source: [SWE-smith (Yang et al.), arXiv:2504.21798](https://arxiv.org/abs/2504.21798).*
- **Wink — the strongest, most current per-model number for newer Claude.** Meta analyzed **42,920 real production trajectories** from its internal VS Code coding agent over five weeks. Overall "infinite loop" prevalence = **5.21%** across all models; total misbehavior ≈ 29%. By model: **Claude Sonnet 4.5 = 2.18%**, **Claude Opus 4.5 = 3.59%.** Loop definition: "the agent invoking the same or similar tool calls three or more times in a row, repeated code edits to the same file, or engaging in verbose reasoning without advancing toward a solution." Notably, loops **increased** from Sonnet 4.5 → Opus 4.5 (statistically significant, p < 0.00001) — "newer model = fewer loops" is not always true. *Source: [Wink (Meta), arXiv:2602.17037](https://arxiv.org/abs/2602.17037).*
- **Scale's SWE-bench Pro trajectory analysis — what Claude fails on when it fails.** Context overflow accounted for **35.6% of Sonnet 4 failures** and semantic-understanding failures for **35.9% of Opus 4.1 failures**, while **tool-use inefficiency (42%) was mostly a smaller-model problem.** Frontier Claude failures cluster on context and comprehension, not tool-call mechanics. *Source: Scale AI SEAL trajectory analysis, via [morphllm.com SWE-bench Pro writeup](https://www.morphllm.com/swe-bench-pro).*
- **ClawsBench — loops migrate to the tail but don't vanish.** Evaluating Claude Sonnet 4.6 and Opus 4.6 on Claude Code (among others), **Claude Opus 4.6 produced the dataset's longest trajectory — 179 steps, with steps 152–178 all empty Terminal calls** — and "these loops occur across all capability tiers." So the single worst degenerate loop in the benchmark belonged to a frontier Claude model. ClawsBench also showed incorrect tool calls spike without scaffolding: agents without skill specs produced **1,000+ "unrecognized subcommand" errors** by inventing CLI syntax. *Source: [ClawsBench, arXiv:2604.05172](https://arxiv.org/abs/2604.05172).*
- **LongCLI-Bench — the regime where loops still bite.** On long-horizon CLI tasks even top **Claude-Opus-4.6 hits only 16.7% pass**, and "repetitive loops from weak strategic adaptation" is one of three dominant failure causes: the agent hits an execution failure, proposes a superficial patch, reruns the same command, observes the same error, and repeats until the step limit is exhausted. *Source: [LongCLI-Bench, arXiv:2602.14337](https://arxiv.org/abs/2602.14337).*
- **Anthropic builds the harness to suppress both modes.** Claude Code restricts tool responses to **25,000 tokens by default**, error responses are prompt-engineered to be actionable (the "structured error message → high recovery rate" lever), specific loop bugs are patched (e.g., an OAuth 401 retry loop), and per-tool OpenTelemetry spans (`claude_code.tool`) let teams measure their own loop/failure rates. *Source: [Anthropic, "Writing tools for AI agents"](https://www.anthropic.com/engineering/writing-tools-for-agents).*

**One number to treat skeptically:** a community reverse-engineering project on Claude Code cites ~250K wasted API calls/day attributed to anti-distillation and frustration-detection mechanisms — but that's an unverified third-party estimate of a *different* phenomenon (defensive calls, not failure loops), so it should not be leaned on. *Source: GitHub reverse-engineering project (unverified).*

**Bottom line for loops:** strict loops sit at **~2–4% for frontier Claude** ([Wink](https://arxiv.org/abs/2602.17037)) versus SWE-smith's **< 4% for Claude 3.7** — a low-single-digit tail risk, with long-horizon tasks as the regime where they bite. Investigating strict loops as a primary token-waste target for frontier Claude on bounded tasks is low-value; they are worth only a cheap guardrail. *Sources: [SWE-smith, arXiv:2504.21798](https://arxiv.org/abs/2504.21798); [Wink, arXiv:2602.17037](https://arxiv.org/abs/2602.17037); [LongCLI-Bench, arXiv:2602.14337](https://arxiv.org/abs/2602.14337); [ClawsBench, arXiv:2604.05172](https://arxiv.org/abs/2604.05172).*

---

## 4. The shift: where the token waste actually is (redundant re-reads / context accumulation)

### 4.1 The key 2026 paper — "How Do AI Agents Spend Your Money?"

The most rigorous source is the first systematic study of token consumption in agentic coding, analyzing OpenHands trajectories from eight frontier LLMs on SWE-bench Verified — including **Claude Sonnet 3.7, 4, and 4.5.** *Source for all five findings below: [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750).*

1. **Input tokens, not reasoning/output, dominate cost.** Agentic coding consumes **~3,500× more tokens than single-round reasoning** and ~1,200× more than chat — driven by exponential **input**-token growth as the same context is fed in repeatedly, producing a far higher input/output ratio even with caching enabled. *The waste is on the input/context side.*
2. **For Claude Sonnet 4.5 specifically**, decomposed by token type across five phases (Setup, Explore, Fix, Validate, Closeout): **cache-read input tokens are the largest category by a wide margin in every phase**, reflecting cumulative reuse of prior context; output tokens are high only in Setup. Output tokens are priced ~80× higher per token than cache reads, yet the sheer accumulated *volume* of re-read context outweighs them.
3. **The behavioral driver of expensive failed runs is redundancy:** higher-cost runs show sharply increased repeated file viewing and modification — "redundant back-and-forth file access and re-editing that inflates context length and token usage without proportional progress." Higher-cost models such as **Claude Sonnet 4**, Qwen3-Coder-480B, and Kimi-K2 perform more file actions, with **~50% of them repeats on the same file.** Concretely: **Claude Sonnet 4 ≈ 14.2 file views (7.17 repeated)** vs. **GPT-5 ≈ 2.4 views (0.65 repeated).** *This is the repetition signal — but it is "re-read the same file again," not a frozen identical-call doom loop.*
4. **Claude-specific efficiency gap:** Kimi-K2 and Claude Sonnet 4.5 consume **>1.5M more tokens than GPT-5** on the same tasks, even on the easy subset all models solve — model temperament, not task difficulty.
5. **More tokens ≠ more accuracy.** Accuracy peaks at intermediate cost and **degrades** at high cost — excess spend is unproductive exploration, not deeper reasoning. Corroborated independently: a clear negative correlation between output-token count and accuracy in tool-augmented agents. *Corroboration: ["Beyond the Final Answer", arXiv:2510.02837](https://arxiv.org/abs/2510.02837).*

### 4.2 Verdict

**Classic repetitive loops are not the biggest token waste in frontier Claude — redundant re-reading/re-editing of the same files is.** Investigating "loops" is worthwhile only if the target is redefined as **redundant file-access + context bloat.** *Sources: [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750); [SWE-Pruner, arXiv:2601.16746](https://arxiv.org/abs/2601.16746).*

---

## 5. The crucial distinction: loops vs. re-reads (NOT the same thing)

**A re-read is a *superset* phenomenon that overlaps with, but is not equal to, a repetitive tool-call loop.**

- **Repetitive tool-call loop (strict):** the **same tool + same arguments** repeated, non-progressing. Match key = the **full signature** `(tool_name, normalized_args)`. Pathological by definition. Rare at the frontier.
- **Redundant file re-read:** the **same file path** read more than once, **regardless of line range/arguments**. Match key = the **file-path field only**.

**Example — same file, different ranges:**

```
read(path="auth.py", lines=1-50)
read(path="auth.py", lines=200-250)
```

- A **loop detector** (full-signature match) sees two *different* calls → does **not** flag.
- A **re-read detector** (path-only match) sees `auth.py` twice → **flags it**.

**Containment is one-directional:** every strict identical re-read *is* a re-read, but most re-reads are *not* strict loops (the args differ). A detector built on "repetitive tool calls" therefore **misses most of the re-reading** — precisely the diffuse, common waste that matters most.

**Three sub-cases of re-reads (sub-classify after path-grouping):**

1. **Identical re-read** — same file, same range, no new reason → this *is* a repetitive call. Pathological. Rare.
2. **Different-slice re-read** — same file, different range → often legitimate new context. Not a loop, but inflates context.
3. **Re-read after context loss / compaction** — content was evicted, so the agent re-reads to recover it. Driven by memory loss, not by being "stuck." **This is the big token-waste case** and looks like a loop in aggregate without being one mechanically.

**Why it matters for cost:** a strict loop wastes tokens by re-billing a frozen useless action until a limit kills the run (catastrophic per incident, rare). Repeated file access wastes tokens by steadily **re-injecting the same file contents into a growing, re-billed context** (the quadratic history problem) — diffuse, common, the actual dominant cost center.

**The single most useful field to log:** *was this file's content already present in the live context window at read time?* That separates wasteful re-reads (cases 1 & 3) from legitimate new exploration (case 2). *Source: analytical framing derived from this investigation; the "expired / redundant / useless" sub-classification follows [AgentDiet, arXiv:2509.23586](https://arxiv.org/abs/2509.23586).*

---

## 6. How to detect re-reads (methodology)

A re-read is detected by examining the **tool-call log**, but **not** by searching for *repetitive* (identical) calls. The match keys on **one field — the file path** — not the whole call.

| Detector | Match key | Finds |
|---|---|---|
| Repetitive tool call (loop) | full `(tool_name, normalized_args)` | "stuck" identical calls |
| Re-read same file | file-path field only (ignore line range) | all repeated access to a file |

**Validated path-level method** *(Sources: [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750); [Code-Cleanliness, arXiv:2605.20049](https://arxiv.org/abs/2605.20049); [ASSERT-KTH reproducible-trajectories](https://github.com/ASSERT-KTH/reproducible-trajectories)):*

1. Parse the trajectory's structured JSON / tool log.
2. Group every read-type action — the `Read` tool **plus** bash `cat` / `head` / `tail` / `sed` / `awk` / `grep` — by its **file-path argument** (canonicalize relative/absolute paths first).
3. Count files whose path appears **≥ 2×**. Report: (a) **revisitation rate** = fraction of read files read ≥2×; (b) **mean re-reads per revisited file**; (c) **re-read token share**.
4. To separate *genuine redundancy* from *forced re-reads*, check whether the file's content was still in the live context at read time (no intervening Edit, no compaction/eviction). Use AgentDiet's three-way taxonomy (useless / redundant / expired) and a TTL/compaction-aware heuristic. *Source: [AgentDiet, arXiv:2509.23586](https://arxiv.org/abs/2509.23586).*

---

## 7. Prevalence of redundant re-reads in Claude

**Strict loops vs. path-level re-reads differ by ~an order of magnitude:**

| Metric | Frontier-Claude figure | Source |
|---|---|---|
| Strict identical-action loops | **2.18% (Sonnet 4.5), 3.59% (Opus 4.5)** | [Wink, arXiv:2602.17037](https://arxiv.org/abs/2602.17037) |
| Path-level repeated file actions | **~50% of file actions** (Sonnet 4) | [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750) |
| Read-token share of total budget | **76.1%** (Sonnet 4.5, Mini-SWE-Agent) | [SWE-Pruner, arXiv:2601.16746](https://arxiv.org/abs/2601.16746) |
| File revisitation reduction on cleaner code | **−34%** (Sonnet 4.6) + 7–8% fewer tokens | [Code-Cleanliness, arXiv:2605.20049](https://arxiv.org/abs/2605.20049) |

**Per-model file-VIEW counts (overall / repeated, ≈repeat share)** *— Source: [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750), Figure 7:*

- Claude Sonnet 3.7: 6.86 / 3.52 (~51%)
- Claude Sonnet 4: 14.20 / 7.17 (~50%)
- Claude Sonnet 4.5: 11.24 / 5.80 (~52%)
- (Contrast) GPT-5: 2.38 / 0.65 (~27%); GPT-5.2: 3.18 / 1.10 (~35%)

**Key reframing:** strict doom-loops are a **rare reliability tail (~2–4%)**; redundant path-level re-reads are a **pervasive efficiency tax (~50% of file actions, up to 76% of tokens)**. Conflating the two would massively overstate "looping." *Sources: [Wink, arXiv:2602.17037](https://arxiv.org/abs/2602.17037); [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750); [SWE-Pruner, arXiv:2601.16746](https://arxiv.org/abs/2601.16746).*

---

## 8. Datasets & tools for analyzing Claude traces

- **[SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) (HuggingFace):** 5,017 trajectories from SWE-agent + Claude 3.7 Sonnet. Easy `load_dataset`. *Success-biased* (curated for fine-tuning), so weaker for failure/loop hunting.
- **[SWE-bench/experiments](https://github.com/swe-bench/experiments) (GitHub + AWS S3):** predictions, execution logs, `.traj` reasoning traces for many submissions — **includes failed runs + multiple Claude variants.** Best source for loop/failure hunting.
- **[benchflow/ClawsBench](https://huggingface.co/datasets/benchflow/ClawsBench) (HuggingFace, [arXiv:2604.05172](https://arxiv.org/abs/2604.05172)):** Claude Opus 4.6 / Sonnet 4.6 (also GPT-5.4, Gemini 3.1, GLM-5), ATIF schema — but a *productivity-agent* benchmark (Gmail/Calendar/Docs/Drive/Slack), not a code repo.
- **[ASSERT-KTH reproducible-trajectories](https://github.com/ASSERT-KTH/reproducible-trajectories) (GitHub):** turnkey path-level re-read extractor for Claude Code `.jsonl` traces (Read + bash cat/head/tail/sed/awk), reports full vs. partial reads.
- **Mitigation code:** SWE-Pruner ([`Ayanami1314/swe-pruner`](https://github.com/Ayanami1314/swe-pruner)), FastContext ([`microsoft/fastcontext`](https://github.com/microsoft/fastcontext)), AgentDiet, and the `read-once` Claude Code hook ([DEV Community](https://dev.to/boucle2026/read-once-a-claude-code-hook-that-stops-redundant-file-reads-4bjk)).
- **Format heterogeneity caveat:** there is no common trajectory format (JSON/YAML/Markdown), so agent-specific parsers are required; key off OBSERVATION markers and the file-path argument.

**Biggest open gap:** no published path-level re-read prevalence for **Opus 4.5/4.6** or absolute re-read rates for **Sonnet 4.6** coding traces (only the single Code-Cleanliness *delta*). The newest models are largely unstudied on this metric. *Sources: gap assessment across [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750) (stops at Sonnet 4.5), [Code-Cleanliness, arXiv:2605.20049](https://arxiv.org/abs/2605.20049) (Sonnet 4.6 delta only), [ClawsBench, arXiv:2604.05172](https://arxiv.org/abs/2604.05172) (4.6 but productivity domain).*

---

## 9. Recommended directions for further investigation

1. **Reframe the target** from consecutive-identical-call loops (rare, already studied on Claude 3.7/4) to **redundant file-access ratio + context-growth curve** — the signal that correlates with both cost and failure and remains under-measured on current Claude (4.5/4.6, Opus 4.6). *Source: [Bai et al., arXiv:2604.22750](https://arxiv.org/abs/2604.22750).*
2. **Build two metrics side by side:** (a) strict-loop detector (full-signature, run ≥10 for SWE-smith parity / ≥3 for doom-loop), (b) path-level re-read detector (path-only match). Always report both — never collapse them into one "loop rate." *Sources: [SWE-smith, arXiv:2504.21798](https://arxiv.org/abs/2504.21798); [Wink, arXiv:2602.17037](https://arxiv.org/abs/2602.17037).*
3. **Add the "already-in-context at read time?" flag** to separate wasteful re-reads (identical-range or post-eviction) from legitimate new exploration (different range). *Source: [AgentDiet, arXiv:2509.23586](https://arxiv.org/abs/2509.23586).*
4. **Run it on real Claude data:** SWE-smith-trajectories (3.7) + SWE-bench/experiments Claude submissions (4 / 4.5) for per-model *absolute* re-read rates the literature reports only in aggregate. For 4.6 / Opus 4.6, use ClawsBench when its trajectory files release (mind the productivity-task domain). *Sources: [SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories); [SWE-bench/experiments](https://github.com/swe-bench/experiments); [ClawsBench, arXiv:2604.05172](https://arxiv.org/abs/2604.05172).*
5. **Mitigate and re-measure:** `read-once` hook (Claude Code) or exploration-offload subagent (FastContext) / context-pruning (SWE-Pruner, AgentDiet). Target the demonstrated **21–60% token reductions at flat pass rate.** *Sources: [read-once hook](https://dev.to/boucle2026/read-once-a-claude-code-hook-that-stops-redundant-file-reads-4bjk); [FastContext, arXiv:2606.14066](https://arxiv.org/abs/2606.14066); [SWE-Pruner, arXiv:2601.16746](https://arxiv.org/abs/2601.16746); [AgentDiet, arXiv:2509.23586](https://arxiv.org/abs/2509.23586).*
6. **Keep loop detection as a cheap guardrail only** (length-N fingerprint + hard budget) for long-horizon/autonomous runs and weaker models — not as a primary research target for frontier Claude on bounded tasks. *Sources: [Steve Kinney](https://stevekinney.com/writing/agent-loops); [SWE-smith, arXiv:2504.21798](https://arxiv.org/abs/2504.21798).*
7. **Secondary high-value target:** tool/MCP **definition overhead as a fraction of total context** — often the single biggest line item and the easiest to cut. *Sources: [AgentMarketCap, "MCP's Context Bloat Crisis"](https://agentmarketcap.ai/blog/2026/04/08/mcp-context-bloat-enterprise-scale-tool-definitions-agent-context-budget); [Anthropic tool-search](https://www.anthropic.com/engineering/writing-tools-for-agents).*

---

## 10. Papers

**Core / token waste & re-reads**
- Bai et al., *How Do AI Agents Spend Your Money?* — [arXiv:2604.22750](https://arxiv.org/abs/2604.22750) (the central paper)
- *SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents* — [arXiv:2601.16746](https://arxiv.org/abs/2601.16746) (76.1% read-token share, Sonnet 4.5)
- Trivedi & Schmitt (SonarSource), *Does Code Cleanliness Affect Coding Agents?* — [arXiv:2605.20049](https://arxiv.org/abs/2605.20049) (−34% revisitation, Sonnet 4.6)
- *AgentDiet: Reducing Cost of LLM Agents with Trajectory Reduction* — [arXiv:2509.23586](https://arxiv.org/abs/2509.23586) (useless/redundant/expired taxonomy)
- *FastContext: Training Efficient Repository Explorer for Coding Agents* — [arXiv:2606.14066](https://arxiv.org/abs/2606.14066) (exploration offload, up to −60% tokens)

**Loops & failure prevalence**
- Yang et al., *SWE-smith* — [arXiv:2504.21798](https://arxiv.org/abs/2504.21798) (the < 4% Claude 3.7 baseline; length-10 = 89% failure)
- Meta, *Wink* — [arXiv:2602.17037](https://arxiv.org/abs/2602.17037) (2.18% Sonnet 4.5 / 3.59% Opus 4.5 strict loops; 42,920 production traces)
- *LongCLI-Bench* — [arXiv:2602.14337](https://arxiv.org/abs/2602.14337) (long-horizon loops; Opus 4.6 at 16.7%)
- *ClawsBench* — [arXiv:2604.05172](https://arxiv.org/abs/2604.05172) (Opus 4.6 179-step empty-terminal loop; Sonnet 4.6)
- *SWE-Compass* — [arXiv:2511.05459](https://arxiv.org/abs/2511.05459) (INF failure category; Gemini 8.3%, Claude lower)
- *How Coding Agents Fail Their Users* — [arXiv:2605.29442](https://arxiv.org/abs/2605.29442) (20,574 real sessions)
- *Beyond Resolution Rates* — [arXiv:2604.02547](https://arxiv.org/abs/2604.02547) (9,374 trajectories, structure > length)
- *Beyond Final Code* — [arXiv:2503.12374](https://arxiv.org/abs/2503.12374) (error-frequency vs. resolution curve)

**Definitions & taxonomies**
- *Agent-Diff* — [arXiv:2602.11224](https://arxiv.org/abs/2602.11224) (runtime tool-failure definition)
- MAST, *Why Do Multi-Agent LLM Systems Fail?* — [arXiv:2503.13657](https://arxiv.org/abs/2503.13657) (step repetition, termination)
- REST-API tool taxonomy — [arXiv:2504.15546](https://arxiv.org/abs/2504.15546) (recursive invocation)
- *Beyond the Final Answer* — [arXiv:2510.02837](https://arxiv.org/abs/2510.02837) (output tokens vs. accuracy)
- Berkeley Function Calling Leaderboard (BFCL) — [gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/leaderboard.html) (incorrect-call categories)

**Overthinking (context, mostly weaker models)**
- *OckBench* — [arXiv:2511.05722](https://arxiv.org/abs/2511.05722) ("Overthinking Tax")
- *Stop Overthinking* survey — [arXiv:2503.16419](https://arxiv.org/abs/2503.16419)

**Vendor / blog & dataset sources**
- Augment Code, *AI Agent Loop Token Costs* — [augmentcode.com](https://www.augmentcode.com/guides/ai-agent-loop-token-cost-context-constraints)
- Steve Kinney, *The Anatomy of an Agent Loop* — [stevekinney.com](https://stevekinney.com/writing/agent-loops)
- DEV Community, *How to Fix Tool-Use Loops in Autonomous Coding Agents* — [dev.to](https://dev.to/alanwest/how-to-fix-tool-use-loops-in-autonomous-coding-agents-540e)
- DEV Community, *read-once: A Claude Code Hook…* — [dev.to](https://dev.to/boucle2026/read-once-a-claude-code-hook-that-stops-redundant-file-reads-4bjk)
- Braintrust, *Debugging Ralph Wiggum* — [braintrust.dev](https://www.braintrust.dev/blog/ralph-wiggum-debugging)
- LangChain, *Context Engineering for Agents* — [langchain.com](https://www.langchain.com/blog/context-engineering-for-agents)
- Anthropic, *Effective context engineering for AI agents* — [anthropic.com](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Anthropic, *Writing tools for AI agents* — [anthropic.com](https://www.anthropic.com/engineering/writing-tools-for-agents)
- AgentMarketCap, *MCP's Context Bloat Crisis* — [agentmarketcap.ai](https://agentmarketcap.ai/blog/2026/04/08/mcp-context-bloat-enterprise-scale-tool-definitions-agent-context-budget)
- Morph, *SWE-bench Pro / Scale trajectory analysis* — [morphllm.com](https://www.morphllm.com/swe-bench-pro)
- Datasets/repos: [SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) · [SWE-bench/experiments](https://github.com/swe-bench/experiments) · [benchflow/ClawsBench](https://huggingface.co/datasets/benchflow/ClawsBench) · [ASSERT-KTH reproducible-trajectories](https://github.com/ASSERT-KTH/reproducible-trajectories)

---


*End of summary. Two conclusions to carry forward: (1) for frontier Claude, the dominant token waste is **redundant file re-reads / context accumulation**, not classic tool-call loops; (2) re-reads should be detected at the **file-path level**, not by matching whole tool calls — and each re-read should record whether the content was already in context, to separate genuine waste from forced re-reads.*
