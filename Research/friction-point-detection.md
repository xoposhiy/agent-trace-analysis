# Friction Point Detection in Agent Traces — Research Notes

> **Project context:** Agent Trace Analysis — detecting, categorizing, and explaining problems in coding-agent sessions (Claude Code traces, SWE-chat, DeepSWE, etc.)

---

## 1. What Is a "Friction Point"?

Before we look at methods, it helps to be clear about what we are trying to find. In the research literature, people use different words for the same idea: *failures*, *errors*, *anomalies*, *inefficiencies*, *waste*. For this project, all of these fall into three simple groups:

- **Correctness friction** — the agent did something wrong. For example: it called a tool with made-up arguments, misread a file, or misunderstood what the user wanted.
- **Efficiency friction** — the agent did too much work. For example: it read the same file twice, kept old useless information in its context, or took a long detour to reach a simple result.
- **Interaction friction** — the human had to step in more than necessary. For example: the user had to correct the agent, repeat themselves, or ask the same thing again.

These three types map directly to the project's cost dimensions: money (efficiency friction wastes tokens), human attention (interaction friction steals human time), and agent runtime (both efficiency and correctness friction slow things down). Most detection methods in the literature target one of these types, but a good system should cover all three.

---

## 2. Taxonomies: What Can Go Wrong?

Before you can detect a problem, you need a clear list of what problems look like. Several recent papers have built these lists — called "taxonomies."

### 2.1 AgentErrorTaxonomy (Zhu et al., 2025) [1]

This taxonomy comes from a large study of more than 500 failed agent trajectories across ALFWorld, GAIA, and WebShop. It organizes failures into five parts, based on which part of the agent broke:

| Module | What fails here | Example |
|---|---|---|
| **Memory** | The agent recalls the wrong thing, or forgets something | It says a file contains X, but X was never there |
| **Reflection** | The agent judges its own progress incorrectly | It thinks a task is done when it is not |
| **Planning** | The agent makes a bad or inefficient plan | It solves the task in 10 steps when 3 would work |
| **Action** | The agent calls a tool with wrong parameters | It searches for a string it made up |
| **System** | An external tool fails or gives unexpected output | An API call returns an error |

The most important finding from this paper is that errors **spread**. An early planning mistake does not stay in the planning step — it quietly breaks the memory and action steps that come after it. This means you cannot just look at one moment in the trace and call it done. You need to find the *root cause*, the first thing that went wrong, and trace how it affected the rest. [1]

The same paper introduces **AgentDebug**, a framework that finds root-cause errors and tells the agent what went wrong. The detector uses **GPT-4.1** as the judge model — the paper tested several alternatives (Llama-3.3-70B, GPT-4o-mini, Qwen3-Next-80B) and GPT-4.1 was clearly best, achieving 42% step accuracy versus significantly lower scores for the others. The backbone agent models tested were **GPT-4o-mini, Qwen3-8B, and Qwen3-Next-80B** — all current models. This feedback helped agents improve their task success by up to 26%. [1]

### 2.2 AgentOps Taxonomy (Dong et al., 2024) [2]

This paper takes a different angle. Instead of asking "what types of errors exist?", it asks "what data do you need to collect in order to detect errors?" It comes from a DevOps perspective — thinking about how you would monitor an agent system in production.

The answer is: you need to log everything that moves through the agent's pipeline — inputs, outputs, tool call names and parameters, token counts, timestamps, and tool results. [2] The paper also points out an important challenge: when something goes wrong, it is often not clear whose fault it is. Was it the agent? The tool? The model provider? This "shared accountability" problem is something our project will need to handle. [2]

---

## 3. Detection Methods

Now that we know what we are looking for, here are the main ways researchers have tried to find it.

### 3.1 LLM-as-a-Judge

The most common approach is to give a language model the agent's trace and ask it to judge whether something went wrong. [3, 4]

The basic process is:
1. Compress the trace into a shorter, cleaner format.
2. Give this compressed trace to a judge model, along with a rubric (a checklist of what errors look like).
3. The judge outputs: did an error happen? What kind? Where?

This approach is used by AgentDebug [1] and several other studies [3]. The problem is that it does not work well out of the box — experiments consistently show that even frontier models score poorly when asked to find errors in raw traces [1, 3]. The main reasons it fails:

- **Bias toward position** — the judge tends to focus on parts of the trace that appear first or last, not the most important parts. [3]
- **Faithfulness problem** — if the agent's own explanation of what it did is wrong (it says "I searched for X" but actually searched for something else), the judge can be fooled. *Gaming the Judge* [4] shows this is a real risk: an agent can produce unfaithful chain-of-thought that passes an LLM-based check while having done the wrong thing.
- **Long context problem** — as the trace gets longer, the judge's accuracy drops. A full coding session can have many thousands of tokens. [2, 3]

The best way to make LLM judging work better is to not judge the whole trace at once. Instead, first use cheap methods to find the parts that look suspicious, then run the judge only on those small windows. [1] We also need to give the judge a clear rubric — not just "is there an error?" but "check each of these five things" (aligned with the taxonomy modules above).

### 3.2 Feedback-Based Detection

This is one of the most interesting methods for this project, and also one of the least explored in the literature.

The idea is simple: the human's own messages are a signal. When a user says "no, that's wrong" or asks the same question twice, that is a sign that the agent made a mistake. When the user accepts the result and moves on, that is a positive signal.

These human messages are called **weak labels** — they are not perfectly reliable (sometimes a user clarifies their intent, not because the agent failed, but because they were not clear the first time), but they contain real information about where things went wrong.

**AgentRewardBench** [5] tests this idea for web agents. The paper evaluates whether automatic judges can predict task success using exactly this kind of signal, and compares different ways of representing the trajectory for the judge.

Another related idea is **Process Reward Models (PRMs)** — models that score each individual step of the agent, not just the final result. [3] This is useful because it helps you find *where* in the trace the problem started, not just that something went wrong overall.

For practical use in this project, the approach would be:
- Parse the JSONL transcript and find "correction turns" — user messages that correct, redirect, or repeat something.
- Use these turns as weak labels to mark suspicious regions.
- The hard part: separating genuine corrections (the agent failed) from normal clarifications (the user was not precise enough at the start).

### 3.3 Specialized Anomaly Detection Models

AgentDebug [1] relies on general-purpose language models as judges. But there is growing evidence that **specialized small models, trained specifically for this task, work better**.

**Trajectory Guard** [6] takes a completely different approach. Instead of using a language model at all, it uses a neural architecture called a **Siamese Recurrent Autoencoder** — a model that learns what a "normal" trajectory looks like, and flags anything that deviates. It does not depend on any specific LLM, so it will not go "stale" as new models are released. It runs at 32 milliseconds per inference, which is 17–27 times faster than LLM-based judges, and achieves F1 scores of 0.88–0.94 on test sets across security audits (RAS-Eval) and multi-agent benchmarks (Who&When). [6]

This speed matters a lot for the project's long-term goal: surfacing hints in real time, while a session is in progress. An LLM judge call takes seconds; a specialized model call takes milliseconds.

Also worth noting — from the broader agent evaluation literature [3] — some simple statistics about how often problems occur:
- About 17% of agent failures involve the agent repeating the same step.
- About 14% involve a mismatch between the agent's reasoning (what it says it will do) and its actual action (what it actually does).

These are useful baselines for building simple heuristic detectors.

### 3.4 Heuristic / Rule-Based Detection

Not every friction point needs an AI to detect it. Some problems have clear patterns that can be found with simple rules. These heuristics serve as good pre-filters before running more expensive LLM judges. [1]

**Tool call problems:**
- *Parameter hallucination*: the agent calls a tool with an argument that does not appear anywhere earlier in the trace. You can check this by looking for the argument string in the previous turns. [1]
- *Wrong tool*: the agent calls a tool that does not exist or is not appropriate for the current situation. You can check this against the list of available tools. [2]

**Repetition patterns** [7]:
- Reading the same file twice with no change in between
- Running the same search query more than once
- A sequence like: edit → undo → same edit again

**Wasted information in the context (AgentDiet [7]):**

This paper identifies three types of waste that appear frequently in coding agent traces:

- **Useless information** — content that the agent loaded into context but never used. For example, a file was read, but nothing from it appeared in the next action.
- **Redundant information** — the same content appears multiple times in the growing context window.
- **Expired information** — content that was once useful but is no longer relevant. For example, an error message from a test run that has already been fixed.

AgentDiet was evaluated on **Claude Sonnet 4** and **Gemini 2.5 Pro** — the most up-to-date model lineup of any paper in this review. Both are current flagship models. The paper found that these three waste types are so common that removing them reduces input tokens by 40–60% without hurting performance on SWE-bench Verified and Multi-SWE-bench. [7]

One extra note relevant to Claude Code specifically: according to data from the OpenRouter platform (September 2025), 99% of all tokens in Claude agent sessions are *input* tokens accumulated in the context window. Only 1% are newly generated. [7] This makes waste detection directly relevant to the money cost dimension.

### 3.5 Planning-Level Detection

Planning errors are the most common category in the AgentDebug study [1] — especially "Inefficient Plan." These errors are dangerous because they happen early and affect everything that comes after.

Simple ways to detect planning problems [1]:

- **Goal-action alignment**: does each action the agent takes connect to a goal it stated earlier? An action with no clear reason is a warning sign.
- **Plan completeness**: did the agent write out a plan before starting work? Agents that jump in without planning tend to fail more.
- **Backtracking count**: how many times did the agent undo something it had already done? More backtracking usually means a worse initial plan.

These checks can be done as heuristics first, and then confirmed by an LLM judge if needed.

---

## 4. Preprocessing Traces

Before any detection method can run, you need to convert the raw JSONL trace into a cleaner format. Several papers discuss how to do this well.

**Fields to extract from Claude Code traces** (based on AgentOps [2] and AgentDiet [7]):

- Turn number, role (user or assistant), timestamp
- Token counts per turn (input and output separately)
- Tool calls: which tool, what parameters, what result, how long it took
- The agent's reasoning text (if extended thinking is available)
- Classification of each human message: is this a new task, a correction, a clarification, or an acceptance?

**Compact formats used in the literature:**

- **Summary per turn**: replace each long tool result with a one-sentence summary. AgentDiet's reflection module does this automatically. [7]
- **Action sequence**: strip all the reasoning text and keep only the action type and key parameters. Useful for finding repetition patterns quickly. [1]
- **State diff**: for coding agents, the change to the codebase between turns is often more informative than the full file content. [7]

The reason these compact formats matter: LLM judges get worse as traces get longer. [2, 3] A compressed representation lets you run a judge on a more manageable input, and also makes it faster and cheaper.

---

## 5. Datasets

| Dataset | Size | Task type | Labels | Source |
|---|---|---|---|---|
| **AgentErrorBench** [1] | ~500 failed rollouts | ALFWorld, GAIA, WebShop | Module-level error labels per step | https://github.com/ulab-uiuc/AgentDebug |
| **SWE-chat** | Large | Real coding sessions | Benchmark pass/fail | Our own target domain |
| **DeepSWE** | Benchmark runs | SWE-bench derivative | Pass/fail per task | Cross-checking detectors |
| **AgentRewardBench** [5] | Web sessions | Web navigation | Human success/failure labels | https://arxiv.org/abs/2504.08942 |

---

## 6. Model Coverage Across Papers

This section summarizes which models each paper tested, and flags any that are now outdated.

| Paper | Models tested | Outdated? |
|---|---|---|
| **AgentDebug** [1] | Judge: GPT-4.1. Agents: GPT-4o-mini, Qwen3-8B, Qwen3-Next-80B | No — all current |
| **AgentDiet** [7] | Claude Sonnet 4, Gemini 2.5 Pro | No — most up-to-date lineup in this review |
| **Trajectory Guard** [6] | No named LLM — custom neural architecture | N/A — not LLM-dependent |

All remaining papers use current models. AgentDiet in particular was updated in March 2026 and uses Claude Sonnet 4 and Gemini 2.5 Pro. Trajectory Guard's architecture is not tied to any specific LLM, so it will not go stale as new models are released.

---

## 7. How to Evaluate Detectors

Once we build a detector, how do we know it is working?

**Precision and recall** against human-labeled traces (AgentErrorBench [1]). The module-level labels give us a clear target: can our detector correctly identify which of the five modules (memory, reflection, planning, action, system) failed?

**Agreement with task outcomes** — for DeepSWE traces: if a task failed, did our detector find at least one friction point? This is not a perfect measure, but it is practical.

**Cost-weighted evaluation** — something we can add that is not in the literature yet: weight missed detections by how expensive the friction was. Missing a 10,000-token wasted detour should count more than missing a single redundant file read. The cost framing is inspired by AgentDiet's token-level cost accounting. [7]

**Counterfactual attribution** [8] — the most careful method: ask "if the agent had done something different at step X, would the final result have changed?" This is expensive to compute but gives the most reliable signal for finding which steps really mattered.

---

## 8. Gaps in the Literature

After reviewing the existing work, here are the areas where the literature does not have good answers yet — and where this project can contribute something new:

**1. Coding agent traces.** Most papers test on web navigation tasks (WebShop, WebArena) or household tasks (ALFWorld) [1, 3]. AgentErrorBench is the closest annotated dataset, but it is still not Claude Code traces. We need to check whether the existing taxonomies and methods transfer.

**2. Cost per friction event.** Papers detect that something went wrong, but almost none of them say how much it cost. [1, 3] Our project's cost dimensions (money, human attention, runtime) are novel and practically useful.

**3. Real-time detection.** Almost all existing work is post-hoc — you analyze the trace after the session ends. Trajectory Guard [6] is the only exception. The project's real-time hints vision is ahead of the literature.

**4. Human feedback as a signal.** The feedback-based detection approach (using the user's replies as weak labels) is almost completely unexplored. AgentRewardBench [5] is the only study that touches this. This is a real opportunity.

**5. Local deployment.** Every major observability tool — LangSmith, Phoenix, Arize, AgentOps [2] — is cloud-hosted. A small, local application that runs on your own machine and reads your own traces does not exist yet.

---

## 9. Proposed Detection Pipeline

Based on everything above, here is a reasonable starting architecture for the MVP phase. It follows the three-stage design (preprocess → heuristic/retrieval → LLM judge) described in the project README, and is grounded in approaches from AgentDebug [1], AgentDiet [7], and AgentOps [2]:

```
Raw JSONL trace
      │
      ▼
[Preprocessor]                         ← AgentOps [2], AgentDiet [7]
  - Extract: turns, tool calls, token counts, timestamps
  - Classify each human turn: task / correction / clarification / accept
  - Compute per-turn token cost and latency
      │
      ▼
[Heuristic pre-filter]                 ← AgentDebug [1], AgentDiet [7]
  - Repetition detector: same tool + same args?
  - Waste classifier: useless / redundant / expired content?
  - Correction counter: how many user correction turns?
  - Output: list of suspicious regions with a score and type
      │
      ▼
[LLM judge — windowed]                 ← AgentDebug [1], LLMs-as-Judges [3]
  - Give judge: compressed window + rubric from AgentErrorTaxonomy
  - Recommended judge: GPT-4.1 or Claude Sonnet 4
  - Output: error category, explanation, severity
      │
      ▼
[Cost estimator]                       ← AgentDiet [7]
  - Token waste, human turns wasted, latency added per friction event
      │
      ▼
[Structured report]
  - Per session: total friction events, cost breakdown
  - Per event: location in trace, category, explanation, estimated cost
```

For the real-time hints phase (Roadmap phase 4), the heuristic pre-filter and Trajectory Guard [6] can run on each turn as it arrives, replacing the LLM judge with a millisecond-latency model.

---

## References

[1] **Where LLM Agents Fail and How They Can Learn From Failures** (AgentDebug, AgentErrorTaxonomy, AgentErrorBench) — Zhu et al., 2025
https://arxiv.org/abs/2509.25370
GitHub: https://github.com/ulab-uiuc/AgentDebug

[2] **AgentOps: Enabling Observability of LLM Agents** — Dong et al., 2024
https://arxiv.org/abs/2411.05285

[3] **LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods** — Li et al., 2024
https://arxiv.org/abs/2412.05579

[4] **Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation** — 2026
https://arxiv.org/abs/2601.14691

[5] **AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories** — 2025
https://arxiv.org/abs/2504.08942

[6] **Trajectory Guard: A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI** — Advani et al., 2026
https://arxiv.org/abs/2601.00516

[7] **Reducing Cost of LLM Agents with Trajectory Reduction** (AgentDiet) — Xiao et al., 2025/2026
https://arxiv.org/abs/2509.23586

[8] **Counterfactual Credit Policy Optimization** — 2026
https://arxiv.org/abs/2603.21563

[9] **Survey on Evaluation of LLM-based Agents** — Yehudai et al., 2025
https://arxiv.org/abs/2503.16416