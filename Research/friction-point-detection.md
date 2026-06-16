# Friction Point Detection in Agent Traces

## 1. What Is a "Friction Point"?

Before we look at methods, it helps to be clear about what we are trying to find. In the research literature, people use different words for the same idea: *failures*, *errors*, *anomalies*, *inefficiencies*, *waste*. For this project, all of these fall into three simple groups:

- **Correctness friction** — the agent did something wrong. For example: it called a tool with made-up arguments, misread a file, or misunderstood what the user wanted.
- **Efficiency friction** — the agent did too much work. For example: it read the same file twice, kept old useless information in its context, or took a long detour to reach a simple result.
- **Interaction friction** — the human had to step in more than necessary. For example: the user had to correct the agent, repeat themselves, or ask the same thing again.

These three types map directly to the project's cost dimensions: money (efficiency friction wastes tokens), human attention (interaction friction steals human time), and agent runtime (both efficiency and correctness friction slow things down). Most detection methods in the literature target one of these types, but a good system should cover all three.

---

## 2. Taxonomies: What Can Go Wrong?

Before you can detect a problem, you need a clear list of what problems look like. Several recent papers have built these lists — called "taxonomies."

### 2.1 AgentErrorTaxonomy (Zhu et al., 2025)

This taxonomy comes from a large study of more than 500 failed agent trajectories in three different task environments. It organizes failures into five parts, based on which part of the agent broke:

| Module | What fails here | Example |
|---|---|---|
| **Memory** | The agent recalls the wrong thing, or forgets something | It says a file contains X, but X was never there |
| **Reflection** | The agent judges its own progress incorrectly | It thinks a task is done when it is not |
| **Planning** | The agent makes a bad or inefficient plan | It solves the task in 10 steps when 3 would work |
| **Action** | The agent calls a tool with wrong parameters | It searches for a string it made up |
| **System** | An external tool fails or gives unexpected output | An API call returns an error |

The most important finding from this paper is that errors **spread**. An early planning mistake does not stay in the planning step — it quietly breaks the memory and action steps that come after it. This means you cannot just look at one moment in the trace and call it done. You need to find the *root cause*, the first thing that went wrong, and trace how it affected the rest.

The same paper introduces **AgentDebug**, a framework that finds root-cause errors and tells the agent what went wrong. In experiments, this feedback helped agents improve their task success by up to 26%.

### 2.2 TRAIL Taxonomy (Deshpande et al., 2025)

TRAIL uses a simpler three-category system:

- **Reasoning errors** — the agent's thinking was wrong (hallucinations, wrong conclusions, logical gaps)
- **System execution errors** — something failed at the technical level (API problems, bad tool output format)
- **Planning and coordination errors** — the agent had a bad strategy or failed to change plans when needed

What makes TRAIL especially useful for this project is that it comes with a **dataset**: 148 real agent traces, each with 841 errors labeled by humans. The traces come from software engineering and information retrieval tasks — the closest thing in the literature to what we are working with (coding agent sessions).

TRAIL also gives an important warning: **even the best current AI models score only about 11% accuracy when asked to find errors in these traces.** And the longer the trace, the worse they do. This tells us that just throwing a raw trace at a language model and asking "what went wrong?" is not good enough. We need smarter methods.

### 2.3 AgentOps Taxonomy (Dong et al., 2024)

This paper takes a different angle. Instead of asking "what types of errors exist?", it asks "what data do you need to collect in order to detect errors?" It comes from a DevOps perspective — thinking about how you would monitor an agent system in production.

The answer is: you need to log everything that moves through the agent's pipeline — inputs, outputs, tool call names and parameters, token counts, timestamps, and tool results. The paper also points out an important challenge: when something goes wrong, it is often not clear whose fault it is. Was it the agent? The tool? The model provider? This "shared accountability" problem is something our project will need to handle.

---

## 3. Detection Methods

Now that we know what we are looking for, here are the main ways researchers have tried to find it.

### 3.1 LLM-as-a-Judge

The most common approach is to give a language model the agent's trace and ask it to judge whether something went wrong.

The basic process is:
1. Compress the trace into a shorter, cleaner format.
2. Give this compressed trace to a judge model, along with a rubric (a checklist of what errors look like).
3. The judge outputs: did an error happen? What kind? Where?

This approach is used by both AgentDebug and TRAIL. The problem is that it does not work well out of the box. TRAIL's 11% accuracy number comes from exactly this setup — just asking a model to read a trace and find errors. The main reasons it fails:

- **Bias toward position** — the judge tends to focus on parts of the trace that appear first or last, not the most important parts.
- **Faithfulness problem** — if the agent's own explanation of what it did is wrong (it says "I searched for X" but actually searched for something else), the judge can be fooled. The paper *Gaming the Judge* (arXiv 2601.14691) shows this is a real risk.
- **Long context problem** — as the trace gets longer, the judge's accuracy drops. A full coding session can have many thousands of tokens.

The best way to make LLM judging work better is to not judge the whole trace at once. Instead, first use cheap methods to find the parts that look suspicious, then run the judge only on those small windows. We also need to give the judge a clear rubric — not just "is there an error?" but "check each of these five things" (aligned with the taxonomy modules above).

### 3.2 Feedback-Based Detection

This is one of the most interesting methods for this project, and also one of the least explored in the literature.

The idea is simple: the human's own messages are a signal. When a user says "no, that's wrong" or asks the same question twice, that is a sign that the agent made a mistake. When the user accepts the result and moves on, that is a positive signal.

These human messages are called **weak labels** — they are not perfectly reliable (sometimes a user clarifies their intent, not because the agent failed, but because they were not clear the first time), but they contain real information about where things went wrong.

**AgentRewardBench** (arXiv 2504.08942) tests this idea for web agents. The paper evaluates whether automatic judges can predict task success using exactly this kind of signal, and compares different ways of representing the trajectory for the judge.

Another related idea is **Process Reward Models (PRMs)** — models that score each individual step of the agent, not just the final result. This is useful because it helps you find *where* in the trace the problem started, not just that something went wrong overall.

For practical use in this project, the approach would be:
- Parse the JSONL transcript and find "correction turns" — user messages that correct, redirect, or repeat something.
- Use these turns as weak labels to mark suspicious regions.
- The hard part: separating genuine corrections (the agent failed) from normal clarifications (the user was not precise enough at the start).

### 3.3 Specialized Anomaly Detection Models

Both AgentDebug and TRAIL rely on general-purpose language models as judges. But there is growing evidence that **specialized small models, trained specifically for this task, work better**.

**TrajAD** (arXiv 2602.06443) is built for runtime anomaly detection — catching problems as they happen, not just after the session ends. It trains a special verifier model using step-level labels (each step in the trace is labeled as OK or not OK). This fine-grained training helps the model learn patterns that general models miss.

The key finding: general LLMs with zero-shot prompting cannot reliably detect these anomalies. You need training data with step-level labels. This makes TRAIL and AgentErrorBench valuable — they are exactly the kind of labeled data needed to train such a model.

**Trajectory Guard** (arXiv 2601.00516) takes a different approach. Instead of using a language model at all, it uses a neural architecture called a **Siamese Recurrent Autoencoder** — a model that learns what a "normal" trajectory looks like, and flags anything that deviates. It runs at 32 milliseconds per inference, which is 17–27 times faster than LLM-based judges. It achieves F1 scores of 0.88–0.94 on test sets.

This speed matters a lot for the project's long-term goal: surfacing hints in real time, while a session is in progress. An LLM judge call takes seconds; a specialized model call takes milliseconds.

Also worth noting — from the broader literature — some simple statistics about how often problems occur:
- About 17% of agent failures involve the agent repeating the same step
- About 14% involve a mismatch between the agent's reasoning (what it says it will do) and its actual action (what it actually does)

These are useful baselines for building simple heuristic detectors.

### 3.4 Heuristic / Rule-Based Detection

Not every friction point needs an AI to detect it. Some problems have clear patterns that can be found with simple rules.

**Tool call problems:**
- *Parameter hallucination*: the agent calls a tool with an argument that does not appear anywhere earlier in the trace. You can check this by looking for the argument string in the previous turns.
- *Wrong tool*: the agent calls a tool that does not exist or is not appropriate for the current situation. You can check this against the list of available tools.

**Repetition patterns:**
- Reading the same file twice with no change in between
- Running the same search query more than once
- A sequence like: edit → undo → same edit again

**Wasted information in the context (AgentDiet, arXiv 2509.23586):**

This paper identifies three types of waste that appear frequently in coding agent traces:

- **Useless information** — content that the agent loaded into context but never used. For example, a file was read, but nothing from it appeared in the next action.
- **Redundant information** — the same content appears multiple times in the growing context window.
- **Expired information** — content that was once useful but is no longer relevant. For example, an error message from a test run that has already been fixed.

AgentDiet found that these three types of waste are so common that removing them reduces input tokens by 40–60% without hurting performance. This means they are both widespread and detectable — making them good targets for a simple heuristic detector.

One extra note relevant to Claude Code specifically: according to data from the OpenRouter platform (September 2025), 99% of all tokens in Claude agent sessions are *input* tokens accumulated in the context window. Only 1% are newly generated. This makes waste detection directly relevant to the money cost dimension.

### 3.5 Planning-Level Detection

Planning errors are the most common category in the AgentDebug study — especially "Inefficient Plan." These errors are dangerous because they happen early and affect everything that comes after.

Simple ways to detect planning problems:

- **Goal-action alignment**: does each action the agent takes connect to a goal it stated earlier? An action with no clear reason is a warning sign.
- **Plan completeness**: did the agent write out a plan before starting work? Agents that jump in without planning tend to fail more.
- **Backtracking count**: how many times did the agent undo something it had already done? More backtracking usually means a worse initial plan.

These checks can be done as heuristics first, and then confirmed by an LLM judge if needed.

---

## 4. Preprocessing Traces

Before any detection method can run, you need to convert the raw JSONL trace into a cleaner format. Several papers discuss how to do this well.

**Fields to extract from Claude Code traces:**

- Turn number, role (user or assistant), timestamp
- Token counts per turn (input and output separately)
- Tool calls: which tool, what parameters, what result, how long it took
- The agent's reasoning text (if extended thinking is available)
- Classification of each human message: is this a new task, a correction, a clarification, or an acceptance?

**Compact formats used in the literature:**

- **Summary per turn**: replace each long tool result with a one-sentence summary. AgentDiet's reflection module does this automatically.
- **Action sequence**: strip all the reasoning text and keep only the action type and key parameters. Useful for finding repetition patterns quickly.
- **State diff**: for coding agents, the change to the codebase between turns is often more informative than the full file content.

The reason these compact formats matter: as the TRAIL study shows, LLM judges get worse as traces get longer. A compressed representation lets you run a judge on a more manageable input, and also makes it faster and cheaper.

---

## 5. Datasets

| Dataset | Size | Task type | Labels | Use for this project |
|---|---|---|---|---|
| **TRAIL** | 148 traces, 841 errors | Software engineering + IR | Human error labels per step | Best fit — start here |
| **AgentErrorBench** | ~500 failed rollouts | ALFWorld, GAIA, WebShop | Module-level error labels | Good for validating taxonomy |
| **TrajBench** | Synthetic | Various | Anomaly labels per step | Training specialized detectors |
| **SWE-chat** | Large | Real coding sessions | Benchmark pass/fail | Our own target domain |
| **DeepSWE** | Benchmark runs | SWE-bench tasks | Pass/fail per task | Cross-checking detectors |
| **AgentRewardBench** | Web sessions | Web navigation | Human success/failure labels | Feedback-based detection |

---

## 6. How to Evaluate Detectors

Once we build a detector, how do we know it is working?

**Precision and recall** against human-labeled traces (TRAIL, AgentErrorBench). The 11% accuracy ceiling from TRAIL gives us a useful starting point — any method that beats 11% is better than a raw LLM judge.

**Agreement with task outcomes** — for DeepSWE traces: if a task failed, did our detector find at least one friction point? This is not a perfect measure, but it is practical.

**Cost-weighted evaluation** — something we can add that is not in the literature yet: weight missed detections by how expensive the friction was. Missing a 10,000-token wasted detour should count more than missing a single redundant file read.

**Counterfactual attribution** (arXiv 2603.21563) — the most careful method: ask "if the agent had done something different at step X, would the final result have changed?" This is expensive to compute but gives the most reliable signal for finding which steps really mattered.

---

## 7. Gaps in the Literature

After reviewing the existing work, here are the areas where the literature does not have good answers yet — and where this project can contribute something new:

**1. Coding agent traces.** Most papers test on web navigation tasks (WebShop, WebArena) or household tasks (ALFWorld). TRAIL is the closest to our setting, but it is still not Claude Code traces. We need to check whether the existing taxonomies and methods transfer.

**2. Cost per friction event.** Papers detect that something went wrong, but almost none of them say how much it cost. Our project's cost dimensions (money, human attention, runtime) are novel and practically useful.

**3. Real-time detection.** Almost all existing work is post-hoc — you analyze the trace after the session ends. Trajectory Guard is the only exception. The project's real-time hints vision is ahead of the literature.

**4. Human feedback as a signal.** The feedback-based detection approach (using the user's replies as weak labels) is almost completely unexplored. This is a real opportunity.

**5. Local deployment.** Every major observability tool — LangSmith, Phoenix, Arize, AgentOps — is cloud-hosted. A small, local application that runs on your own machine and reads your own traces does not exist yet.

---

## 8. Proposed Detection Pipeline

Based on everything above, here is a reasonable starting architecture for the MVP phase:

```
Raw JSONL trace
      │
      ▼
[Preprocessor]
  - Extract: turns, tool calls, token counts, timestamps
  - Classify each human turn: task / correction / clarification / accept
  - Compute per-turn token cost and latency
      │
      ▼
[Heuristic pre-filter]  ← fast, cheap, runs on every turn
  - Repetition detector: same tool + same args?
  - Waste classifier: useless / redundant / expired content?
  - Correction counter: how many user correction turns?
  - Output: list of suspicious regions with a score and type
      │
      ▼
[LLM judge — windowed]  ← only on suspicious regions
  - Give judge: compressed window + rubric from AgentErrorTaxonomy
  - Output: error category, explanation, severity
      │
      ▼
[Cost estimator]
  - Token waste, human turns wasted, latency added per friction event
      │
      ▼
[Structured report]
  - Per session: total friction events, cost breakdown
  - Per event: location in trace, category, explanation, estimated cost
```

This matches the README's three-stage design (preprocess → heuristic/retrieval → LLM judge) and is grounded in how the best systems in the literature work.

---

## References

- **Where LLM Agents Fail and How They Can Learn From Failures** (AgentDebug, AgentErrorTaxonomy) — Zhu et al., 2025
  https://arxiv.org/abs/2509.25370
  GitHub: https://github.com/ulab-uiuc/AgentDebug

- **TRAIL: Trace Reasoning and Agentic Issue Localization** — Deshpande et al., 2025
  https://arxiv.org/abs/2505.08638
  Dataset: https://huggingface.co/datasets/PatronusAI/TRAIL
  GitHub: https://github.com/patronus-ai/trail-benchmark

- **Reducing Cost of LLM Agents with Trajectory Reduction** (AgentDiet) — 2025/2026
  https://arxiv.org/abs/2509.23586

- **TrajAD: Trajectory Anomaly Detection for Trustworthy LLM Agents** — Liu et al., 2026
  https://arxiv.org/abs/2602.06443

- **Trajectory Guard: A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI** — Advani et al., 2026
  https://arxiv.org/abs/2601.00516

- **AgentOps: Enabling Observability of LLM Agents** — Dong et al., 2024
  https://arxiv.org/abs/2411.05285

- **Survey on Evaluation of LLM-based Agents** — Yehudai et al., 2025
  https://arxiv.org/abs/2503.16416

- **AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories** — 2025
  https://arxiv.org/abs/2504.08942

- **Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation** — 2026
  https://arxiv.org/abs/2601.14691

- **LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods** — Li et al., 2024
  https://arxiv.org/abs/2412.05579
