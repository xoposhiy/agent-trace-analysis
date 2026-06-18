# Finding and Fixing Wasted Context in LLM Coding Agents

---

## Abstract

When a coding agent such as Claude Code runs, it keeps a growing record of everything it has done: every tool call, every file it read, and every step of its own reasoning. This record is sent back to the model on each new turn, so **most of the tokens an agent pays for are accumulated input context, not new output**. A large part of this context is *waste* — content that is useless, repeated, or out of date. This paper reviews recent research and engineering work on three connected questions: (1) How much context is waste, and can we remove it safely? (2) Can we automatically detect the parts of a session where the agent explores a lot but reaches dead-ends? (3) What can we do about it — for example, move exploration into **subagents**, guide the agent with a **navigation file** (AGENTS.md / CLAUDE.md), or **warn the user** when a session keeps missing the prompt cache? We bring together evidence from academic papers and from engineering reports by Anthropic, Manus, and others. We also describe a real tension: removing wasted context can break the prompt cache, which is itself a major source of cost. The paper ends with a practical design for a tool that detects bad usage patterns and warns the user.

---

## 1. Introduction: the cost lives in the context

A modern coding agent works in a loop. It reads files, runs commands, looks at the results, thinks, and then decides what to do next. Each of these steps is added to a long history, often called the *trajectory*. On every new turn, this whole history is sent to the model again. Because of this, the number of *input* tokens grows quickly, while the number of *output* tokens (the agent's actual answers) stays small. The Manus team, who build a production agent, report that in their system the ratio of input to output tokens is about 100 to 1 ([Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus), Ji, 2025). In other words, the cost of running an agent is mostly the cost of re-reading its own past.

Anthropic frames context as a *finite resource* that must be managed carefully, because a model's attention does not scale forever as the context gets longer ([Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents), Anthropic, 2025). When the history is full of low-value content, the model both pays more and can perform worse. This is the core problem this paper studies, and it leads to a simple research question from our project notes: *Can we detect the parts of a session that are mostly exploration and dead-ends, and then either remove that waste, move it elsewhere, or warn the user?*

---

## 2. How much context is waste? The AgentDiet result

The clearest evidence on this question comes from a paper called [Reducing Cost of LLM Agents with Trajectory Reduction](https://arxiv.org/abs/2509.23586) (Xiao et al., 2026; to appear in *Proc. ACM Softw. Eng.*, FSE 2026). The authors studied the trajectories of a top coding agent (Trae Agent) on the SWE-bench Verified benchmark and found that waste falls into three clear types:

- **Useless information** — content the agent never needs again, such as cache files in a directory listing or noisy build output (for example, repeated "Entering/Leaving directory" lines).
- **Redundant information** — the same content repeated, which happens most often when a file is edited several times and each full version is kept.
- **Expired information** — content that *was* correct but is now out of date, for example an old version of a file that has since been changed.

Their tool, **AgentDiet**, removes this waste during execution using a small "reflection" module: a cheaper model looks at older steps and rewrites them in a shorter form, while keeping the parts that still matter. To save effort, it uses a sliding window, so it only cleans steps that are old enough to be safe to compress (Xiao et al., 2026).

The headline numbers are the anchor of our project. According to the paper, **AgentDiet reduced input tokens by 39.9% to 59.7%, and total computational cost by 21.1% to 35.9%, while keeping the same task success rate** (Xiao et al., 2026). This is strong support for the claim in our notes that 40–60% of context can be removed with no loss in quality. The authors also note that, before their work, agent products such as Claude Code and Cursor only compressed context when the window was nearly full, treating it as a safety step rather than an efficiency goal (Xiao et al., 2026). So the room for improvement was real and mostly unexplored.

One caution from the same paper: full *summarization* of the history can backfire. Summaries cost extra model calls, and they can hide the failure signals an agent needs to learn from, which sometimes makes the trajectory *longer*, not shorter (Xiao et al., 2026). This is why simple, rule-based removal of clear waste is often the safer default.

---

## 3. Detecting exploration and dead-ends in a session trace

The next research question is whether we can *automatically spot* the parts of a session that are mostly fruitless exploration. Several recent studies suggest we can, by looking at the *shape* of the trajectory.

The most direct evidence comes from a study of offensive-security agents, [CyberExplorer](https://arxiv.org/abs/2602.08023) (2026). The authors compared successful and failed runs along two axes: how many sub-tasks were spawned ("breadth") and how many interaction rounds were used ("depth"). They found that successful runs form a tight cluster with limited spawning and moderate depth, while **dead-end runs spread out widely: they spawn many helpers and use many rounds, yet still fail** (CyberExplorer, 2026). In their words, failure cases do not stop early; they continue through long but ineffective exploration. This gives us a measurable fingerprint of a dead-end: high activity with no progress.

A similar pattern appears in software-engineering agents. The paper [Process-Centric Analysis of Agentic Software Systems](https://arxiv.org/abs/2512.02393) (2026) builds a graph of each agent's process. It reports that *resolved* issues tend to follow a clean path — locate the bug, patch it, then validate — while *unresolved* issues show chaotic, repetitive, or backtracking behavior (Process-Centric Analysis, 2026). Importantly, the authors also built a method to construct and check this graph *in real time*, so trajectory problems can be flagged while the agent is still running, not only afterwards.

Other work focuses on *why* agents get stuck. [When Agents go Astray: Course-Correcting SWE Agents with PRMs](https://arxiv.org/abs/2509.02360) (Gandhi et al., 2026) names the common inefficiencies plainly: redundant exploration, looping, and failing to stop once a solution is already found. The authors argue that most earlier work only diagnoses these problems *after* a run ends, which wastes the chance to save compute during the run; their tool instead scores the process step by step and corrects course as it happens (Gandhi et al., 2026). In a related line, research on deep-research agents ([RE-TRAC: Recursive Trajectory Compression for Deep Search Agents](https://arxiv.org/abs/2602.02486), 2026) found that in failed runs, the agent very often *plans* to explore a branch and then forgets to — in their analysis, up to 93% of failed trajectories contained such forgotten branches. This is a different but related signal of an unhealthy exploration pattern.

Finally, [Where LLM Agents Fail and How They Can Learn From Failures](https://www.researchgate.net/publication/396048725_Where_LLM_Agents_Fail_and_How_They_can_Learn_From_Failures) (the AgentDebug system, 2026) shows the practical payoff of finding the single "critical step" where a run went wrong: letting the agent retry from that point, rather than from the start, sharply raised success rates on the ALFWorld benchmark (for example, from 21 to 55 successes for one model). Together, these papers support a clear answer to our question: **yes, dead-end exploration has detectable signatures** — wide branching with no progress, repetitive or backtracking steps, looping, and forgotten plans.

---

## 4. Two fixes: subagents and navigation files

Once we can detect heavy, low-value exploration, the natural next step is to *move it out of the main session* or *prevent it in advance*. Our project notes suggest two ideas, and both are well supported.

### 4.1 Move exploration into subagents

A **subagent** is a separate agent, with its own fresh context window, that the main ("lead") agent creates to handle one sub-task. The key benefit is *context isolation*: the messy details of the sub-task stay inside the subagent and never pollute the main context. Anthropic describes exactly this design in [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (Anthropic, 2025). In their setup, a lead agent plans the work and spawns several subagents that run in parallel, each exploring one direction; each subagent then returns only its findings, not its whole search history.

Anthropic reports that this multi-agent design beat a single-agent version of Claude Opus 4 by about 90% on their internal research evaluation, but it also used roughly 15 times more tokens than a normal chat (Anthropic, 2025). This is why subagents are not free: they make sense when a task is valuable and can be split into independent parts. The same report is honest about the limits — it states that subagents work less well for tightly connected work such as most coding, because those tasks cannot be cleanly split and agents are still weak at coordinating in real time (Anthropic, 2025). Anthropic also describes early failures, such as agents spawning far too many subagents for simple questions, which echoes the "dead-end" spawning pattern from Section 3.

A companion guide, [When to use multi-agent systems (and when not to)](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) (Anthropic, 2026), gives a useful rule: context isolation helps most when a sub-task produces a lot of context (more than about 1,000 tokens) but most of that content is *not* needed by the main task. This matches our project idea well: the explorative, dead-end-heavy parts of a session are exactly the kind of work that creates a lot of throw-away context, so they are good candidates to push into a subagent that returns only a short, distilled summary.

### 4.2 Guide the agent with a navigation file (AGENTS.md / CLAUDE.md)

The second fix is to *prevent* wasteful exploration by giving the agent a map of the project up front. The open standard for this is [AGENTS.md](https://agents.md/), described as a "README for agents" — a predictable place to put build steps, test commands, conventions, and structure (agents.md, 2025). Claude Code's own version is **CLAUDE.md**. In December 2025 the standard moved under the Linux Foundation's Agentic AI Foundation, and by mid-2026 more than 60,000 public GitHub repositories included such a file ([AGENTS.md Guide (2026)](https://vibecoding.app/blog/agents-md-guide), 2026).

The intuition is strong: if the agent already knows where things are and how to run the tests, it should explore less and waste fewer tokens. Practical guides stress keeping these files *short*. The [Writing a good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md) guide (HumanLayer, 2025) explains why: models follow instructions best at the very start and very end of the context, and as the number of instructions grows, the model follows *all* of them less reliably — so a focused file beats a long one.

But there is an important and surprising counter-result. A careful study from ETH Zurich and LogicStar.ai, [Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?](https://arxiv.org/abs/2602.11988) (Gloaguen et al., 2026), tested four coding agents with no context file, an auto-generated file, and a human-written file. Across the benchmarks, they found that **context files tended to *lower* task success compared with no file, and raised inference cost by more than 20%** (Gloaguen et al., 2026). The reason connects directly to Section 3: the files *encouraged broader exploration* — more testing and more file traversal — which often added cost without adding success. The authors suggest that in *well-documented* repositories the file is mostly redundant, and redundancy is itself costly for agents.

The lesson is not "never use a navigation file." It is that a navigation file only helps when it fills a real gap, and that a bloated or auto-generated one can *cause* the very dead-end exploration we want to remove. So the navigation file is both a possible fix *and* a possible source of the problem, depending on how it is written.

---

## 5. Cache misses: a hidden and very large cost

There is a second kind of waste that is easy to miss: **paying full price to re-read context that could have been served from a cache.**

Anthropic's API supports *prompt caching*. It stores the front part of a request so that repeated calls with the same beginning are much cheaper and faster. According to the [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) documentation (Anthropic), reading from the cache costs only about 10% of the normal input price, while writing to it costs a little more than the base price; the cache matches the prompt as a *prefix*, in the fixed order tools → system → messages. The original launch announcement claims caching can cut costs by up to 90% and latency by up to 85% for long prompts ([Prompt caching with Claude](https://www.anthropic.com/news/prompt-caching), Anthropic, 2024).

The catch is the word *prefix*. Because the cache only matches from the start of the request, **anything that changes early in the context invalidates the cache for everything after it.** The Claude Code engineering post [Lessons from Building Claude Code: Prompt Caching Is Everything](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything) (Shihipar, 2026) lists the common causes of cache misses: putting a timestamp near the front, changing or reordering tool definitions, editing the system prompt mid-session, or switching models. The team treats a drop in cache hit rate as a production incident, because even a small miss rate can sharply raise cost and latency (Shihipar, 2026). The Manus team make the same point: a single different token early in the prompt can break the cache from that token onward (Ji, 2025).

**How common is this problem?** Quite common, and the impact is large. The security-agent team at ProjectDiscovery describe in [How We Cut LLM Costs by 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching) (2026) that their agent ran at only a 7% cache hit rate before they fixed it, and reached 84% after moving dynamic content out of the cached prefix — cutting overall cost by 59%. On the academic side, a workload study, [Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297) (Yuan et al., 2026), found that *with healthy caching*, agentic coding sessions reuse 84.6% to 99.5% of their input tokens. That high number is exactly why a *broken* cache is so expensive: the agent is forced to reprocess a huge, ever-growing prefix at full price. A real Claude Code bug report (GitHub issue #24147) shows the extreme case: re-reading a large CLAUDE.md on every message drove cache reads to consume almost all of a user's quota.

**The key tension.** This creates a direct conflict with Section 2. Removing wasted context (pruning) means *editing the history*. But editing the history *breaks the prefix cache*. So aggressive pruning can save input tokens while *also* destroying cache hits — and the cache may have been saving far more than the pruning gained. Anthropic's own context-editing documentation warns about this and advises only clearing enough content to make the cache loss worthwhile. A cache-aware research system, [TokenPilot: Cache-Efficient Context Management for LLM Agents](https://arxiv.org/html/2606.17016) (Xu et al., 2026), shows the payoff of respecting this rule: by stabilizing the prefix and evicting context carefully, it raised cache hit rates (for example from about 39% to 79% on one benchmark) and cut total cost from $7.24 to $2.79 on its tests.

The way to resolve the tension is the same fix from Section 4: **append, do not edit.** Add new guidance at the *end* of the context, keep the early prefix stable, and push exploration into subagents whose throw-away context never touches the parent's cached prefix. This way, context hygiene and cache hygiene point in the same direction.

---

## 6. A practical detector-and-warning system

The research above points to a concrete tool that fits our project goals. It would run quietly alongside the agent and watch the session telemetry. Each assistant turn from the Anthropic API reports three numbers in its `usage` object: `cache_read_input_tokens`, `cache_creation_input_tokens`, and `input_tokens`, and Claude Code already writes these to its per-session log file ([How to Monitor Claude Code Cache Statistics and Token Usage](https://docs.bswen.com/blog/2026-04-01-monitor-cache-stats/), BSWEN, 2026). From these, the tool could do three things:

1. **Warn about cache-miss-heavy sessions.** Compute the cache ratio per turn: `cache_read / (cache_read + cache_creation + input_tokens)`. In a healthy session this climbs toward ~90%. If cache reads stay flat while cache *creation* keeps growing, the cache is broken (BSWEN, 2026). The tool would then name the likely cause — an edited system prompt, a model switch, an idle timeout, or a CLAUDE.md that changes every turn — and suggest a fix.

2. **Flag dead-end exploration.** Using the signals from Section 3 — wide branching with no progress, repeated or backtracking steps, looping, and long activity without a solution — the tool could mark stretches of the session as "explorative with dead-ends" and suggest moving that work into a subagent (Sections 3 and 4.1).

3. **Audit the navigation file.** Following Gloaguen et al. (2026), the tool could warn when a CLAUDE.md / AGENTS.md is long, auto-generated, or largely duplicates the repository's own docs, since such files can *increase* exploration and cost rather than reduce them.

The first feature is the easiest to build and gives the clearest, most measurable value, so it is the natural place to start.

---

## 7. Conclusion

Most of what a coding agent pays for is its own accumulated context, and a large share of that context is waste. The [AgentDiet](https://arxiv.org/abs/2509.23586) study shows that 40–60% of input tokens can be removed without hurting quality. Recent trajectory-analysis work shows that the worst waste — wide, looping, dead-end exploration — has detectable signatures. Two fixes follow: route that exploration into **subagents** with isolated context, and guide the agent with a **navigation file** — while remembering the ETH Zurich finding that a bad navigation file can make things worse. Finally, a second hidden cost, **cache misses**, is common, expensive, and in tension with pruning; the safe path is to keep the early context stable, append rather than edit, and warn users when their usage pattern is quietly burning tokens. A lightweight monitor built on the agent's own telemetry could deliver all three benefits today.

---

## References

The paper titles below are clickable links.

1. Xiao, Y.-A., et al. (2026). [*Reducing Cost of LLM Agents with Trajectory Reduction*](https://arxiv.org/abs/2509.23586) (the **AgentDiet** paper). arXiv:2509.23586; to appear in *Proc. ACM Softw. Eng.* (FSE 2026). — Source for the 39.9–59.7% input-token reduction, 21.1–35.9% cost reduction, the three waste types (useless/redundant/expired), and the reflection-module method.

2. Anthropic (2025). [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). — Context as a finite resource; subagents return short distilled summaries.

3. Ji, Y. ("Peak") (2025). [*Context Engineering for AI Agents: Lessons from Building Manus*](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus). Manus. — ~100:1 input-to-output token ratio; single-token cache invalidation; append-only context.

4. CyberExplorer team (2026). [*CyberExplorer: Benchmarking LLM Offensive Security Capabilities...*](https://arxiv.org/abs/2602.08023). arXiv:2602.08023. — Dead-end runs show wide branching and long depth without success; successful runs form a tight cluster.

5. Process-Centric Analysis authors (2026). [*Process-Centric Analysis of Agentic Software Systems*](https://arxiv.org/abs/2512.02393). arXiv:2512.02393. — Resolved issues follow locate→patch→validate; unresolved ones are chaotic/repetitive/backtracking; real-time trajectory flagging.

6. Gandhi, S., et al. (2026). [*When Agents go Astray: Course-Correcting SWE Agents with PRMs*](https://arxiv.org/abs/2509.02360) (SWE-PRM). arXiv:2509.02360. — Redundant exploration, looping, and failure to terminate; inference-time process reward model for course-correction.

7. RE-TRAC authors (2026). [*RE-TRAC: Recursive Trajectory Compression for Deep Search Agents*](https://arxiv.org/abs/2602.02486). arXiv:2602.02486. — Up to 93% of failed deep-research trajectories contain planned-but-forgotten branches.

8. AgentDebug authors (2026). [*Where LLM Agents Fail and How They Can Learn From Failures*](https://www.researchgate.net/publication/396048725_Where_LLM_Agents_Fail_and_How_They_can_Learn_From_Failures). — Finding the single critical failure step and retrying from it sharply raises success on ALFWorld.

9. Anthropic (2025). [*How we built our multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system). — Orchestrator-worker pattern; ~90% lift over single agent; ~15× tokens; weaker fit for tightly coupled coding; over-spawning failure mode.

10. Anthropic (2026). [*When to use multi-agent systems (and when not to)*](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them). — Context isolation helps most when a sub-task makes >1,000 tokens of mostly-irrelevant content.

11. agents.md (2025). [*AGENTS.md — an open standard for coding agents*](https://agents.md/). — "README for agents"; nearest-file precedence; widely adopted standard.

12. vibecoding.app (2026). [*AGENTS.md Guide (2026): Copilot, Cursor & More*](https://vibecoding.app/blog/agents-md-guide). — 28+ tools and 60,000+ repos by mid-2026; Linux Foundation Agentic AI Foundation.

13. HumanLayer (2025). [*Writing a good CLAUDE.md*](https://www.humanlayer.dev/blog/writing-a-good-claude-md). — Models follow start/end instructions best; instruction-following degrades as instruction count grows; keep files short.

14. Gloaguen, T., Mündler, N., Müller, M., Raychev, V., & Vechev, M. (2026). [*Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?*](https://arxiv.org/abs/2602.11988). ETH Zurich & LogicStar.ai. arXiv:2602.11988. — Context files tended to lower success and raised cost by >20% by encouraging broader exploration.

15. Anthropic. [*Prompt caching* (Claude API documentation)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). — Prefix-match caching; ~10% read cost; tools→system→messages order; `usage` telemetry fields.

16. Anthropic (2024). [*Prompt caching with Claude*](https://www.anthropic.com/news/prompt-caching). — Up to 90% cost and 85% latency savings on long prompts.

17. Shihipar, T. (2026). [*Lessons from Building Claude Code: Prompt Caching Is Everything*](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything). Anthropic. — Causes of cache misses; treating cache breaks as incidents; mitigations.

18. ProjectDiscovery (2026). [*How We Cut LLM Costs by 59% With Prompt Caching*](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching). — Hit rate 7% → 84%; 59% cost reduction in production.

19. Yuan, Y., Nayak, A., Kundu, S., & Talati, N. (2026). [*Agentic AI Workload Characteristics*](https://arxiv.org/abs/2605.26297). arXiv:2605.26297. — Healthy caching reuses 84.6–99.5% of input tokens; cache thrashing under memory pressure.

20. Xu, B., et al. (2026). [*TokenPilot: Cache-Efficient Context Management for LLM Agents*](https://arxiv.org/html/2606.17016). arXiv:2606.17016 (work in progress). — Prefix stabilization raised hit rate (~39%→79%); cost cut from $7.24 to $2.79.

21. BSWEN (2026). [*How to Monitor Claude Code Cache Statistics and Token Usage*](https://docs.bswen.com/blog/2026-04-01-monitor-cache-stats/). — Cache-ratio formula and the "flat reads, growing creation" warning sign.

22. anthropics/claude-code (2026). [*GitHub issue #24147*](https://github.com/anthropics/claude-code/issues/24147). — Real case where re-reading a large CLAUDE.md consumed almost all cache-read quota.

---
