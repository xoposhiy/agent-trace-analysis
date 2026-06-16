# Local Trace Analysis Tools — Landscape & Verified Notes

Analysis of local-first tools for inspecting AI coding-agent runs, focused on Claude
Code (sessions stored as JSONL in `~/.claude/projects/<key>/<id>.jsonl`).

**Verification legend** — claims here are marked by how we confirmed them:
- ✅ **Verified** hands-on in this environment (lm-assist v0.1.70)
- ⚠️ **Partly true** — real, but with an important caveat
- ❌ **Not true** for the installed version
- ❓ **Unverified** — per the tool's own docs/marketing; not tested, or required
  components were not installed

> Only **lm-assist** was tested hands-on. **claude-trace** and **AgentTrace** sections
> are from their docs (❓) and should be verified before relying on them.

---

## How tools get their data (3 categories)

- **JSONL readers** — parse Claude Code files post-hoc, no code changes
  (lm-assist, claude-trace, claude-code-trace, AgentTrace, ccusage, …)
- **SDK instrumentation** — add a tracing library to *your own* agent code; do **not**
  read Claude Code JSONL (AgentTrace-by-Rxflex, Langfuse, Arize Phoenix)
- **Custom parser** — read the JSONL directly and compute derived signals (`lm_assist.py`)

---

## lm-assist — advertised features vs. what we verified

lm-assist is a local observability platform (web UI on `:3848`, REST API on `:3100`)
that parses Claude Code JSONL. Below, each advertised feature is annotated with reality.

### 1. Tracking token usage and costs

- **Per-turn token & cost breakdown** (input/output, 5-min & 1-hour cache reads, live USD)
  — ❌ **Not true in v0.1.70.** Every per-message API endpoint (`/conversation`,
  `/messages/last`, `/compact-messages`, `responses[]`) returns **only**
  `inputTokens` / `outputTokens` per turn. The 5m/1h cache split and cost are computed
  **only at the session level** (`modelUsage`, `totalCostUsd`). The per-turn cache
  breakdown exists **only in the raw JSONL** — which is why `lm_assist.py turns` parses
  the file directly. *(The cost engine internally knows the ephemeral fields, but no
  per-turn view or endpoint exposes them.)*
- **Real-time status line** (context %, 5h/7d rate limits, session cost, PIDs)
  — ✅ **Verified.** Statusline component is installed (`statusline-worktree.js`).
- **Ecosystem analytics** (session browser aggregates all projects; sort/filter by burn)
  — ✅ **Verified.** `/projects` returns every project with `sessionCount`,
  `storageSize`, git info, last activity.

### 2. Spotting infinite loops and tool errors

- **Visual subagent tree** (hierarchical nodes: prompt, tool-call count, token cost,
  status) — ⚠️ **Partly true.** Verified the tree populates: spawning 3 `Task` agents
  took `/subagents` from 0 → 3 and `/dag` to 417 nodes / 6 branches. **Caveats:**
  (a) only **inline `Task`/`Agent` spawns** appear — agents spawned by the `Workflow`
  tool live in a *separate* transcript dir and do **not** show in this tree;
  (b) "see a massive repeating branch if stuck in a loop" is **visual inference, not an
  automated detector** — lm-assist has **no automated loop detection**, and same-tool
  loops on the *main thread* (e.g. one bash retried 10×) don't appear as branches at all.
- **Detailed tool-call inspector** (formatted stdout/stderr instead of raw JSON)
  — ❓ Plausible (the Tools tab; data is in the JSONL) but not specifically tested.

### 3. Exposing workflow friction points

- **15 insight views** (Chat, Thinking, subagents, Plans, DAG, files, Git, team, …)
  — ✅ **Verified.** The tabbed dashboard exists with these dimensions.
- **Extended thinking scratchpads** (renders thinking blocks verbatim)
  — ✅ **Verified.** Thinking blocks are in the JSONL and shown in the Thinking tab.
- **Live terminal view** (ttyd-powered, watch/abort active sessions)
  — ✅ **Verified.** Required installing `ttyd` (`brew install ttyd`); the Console then
  worked. Not installed by default on macOS.
- *Note:* lm-assist has **no automated friction detection** (errors/retries/corrections).
  You can read errors in the tool inspector, but nothing flags them. `lm_assist.py
  friction` adds this.

### 4. Context reuse & caching (knowledge / MCP injection)

Advertised: auto-extracts structured knowledge from past sessions into a **local vector
DB**, then injects only relevant context via **MCP** on new prompts — claimed to shrink
startup from 100k+ tokens to 3k–5k.

— ❓ **Unverified, and required components were absent.** In our setup, `lm-assist
status` showed **MCP Server: (not installed)** and **Hook: (not installed)**. This
feature depends on the MCP + hook + plugin components, which were not present, so the
token-savings claim could not be confirmed. Installing the plugin
(`/plugin install lm-assist@langmartai`) would be needed to test it.

### lm-assist — net assessment

A genuinely rich **read / browse / monitor dashboard** for individual sessions
(✅ session-level cost, DAG, thinking, live terminal, 15 views). It is **not** an
*analysis* engine: no automated loop detection, no friction detection, and **no per-turn
cache/cost breakdown** (session-level only). Those gaps are exactly what a parser fills.

### Verified in-built data (from this session)

What lm-assist's UI actually showed for our session `a7526c9f`, captured live.

**Meta tab — session-level token & cost breakdown** (this is the ONLY token breakdown
lm-assist exposes; note it is aggregate, not per-turn, and has no 5m/1h split):

| Field | Value |
|---|---|
| Model | claude-opus-4-8 |
| Total cost | $24.3534 |
| Turns / Messages | 300 / 384 |
| Input tokens | 17,894 |
| Output tokens | 295,721 |
| Cache read | 26,947,270 |
| Cache creation | 543,564 |

**Agents tab — subagent tree** (after spawning 3 `Task` agents; `3 subagents / 3 completed`):

| Agent | Type | Turns | Tool uses | Tools | Tokens | Status |
|---|---|---|---|---|---|---|
| a24acce89350 | general-purpose | 7 | 4 | Read, Bash×3 | 7.4k | ✅ Completed |
| ab38602b703d | general-purpose | 3 | 1 | Read | 7.2k | ✅ Completed |
| a4ff94240294 | general-purpose | 3 | 1 | Read | 5.7k | ✅ Completed |

Each node shows the exact prompt received, tool-call count, independent token cost, and
status — and is clickable to jump into that agent's log. *(Confirms the subagent tree
works for inline `Task`/`Agent` spawns; `Workflow`-spawned agents are not shown here.)*

### Per-turn statistics — `lm_assist.py` (the gap-filler)

What lm-assist can't show per-turn, the parser computes. Live rollup for the same session:

```
MAIN THREAD: 137 turns · 98 tool calls · est. $35.68
  tools: Bash×64, Edit×20, Write×5, Agent×3, WebSearch×2, ToolSearch×1, Skill×1, Workflow×1, Read×1
SPAWNED SUBAGENTS: 106 agents · 517 tool calls · est. $45.85   (103 workflow + 3 Task)
GRAND TOTAL est. $81.53
```

- **Token totals** (parser, matches Meta tab ±growth): input 17,900 · output 302,627 ·
  cache_creation 565,326 · cache_read 27,544,582 · est. $24.96.
- **Per-turn** (`turns`/`detail`): each turn's input/output/**5m cache**/**1h cache**/
  cache-read + estimated cost, with `<-- spike` flags. *(lm-assist shows none of this.)*
- **Loop signal** (`loops`): consecutive same-tool runs (e.g. `19x Bash in a row`) and
  identical repeated commands — neither exposed by lm-assist.
- **Turn counting differs:** lm-assist reports `Turns 300` (counts message events incl.
  multi-line duplicates); the parser reports `137` (deduped by assistant message id).

> **Bottom line, validated two ways:** lm-assist's session-level numbers and the parser's
> JSONL parse agree. The difference is granularity — lm-assist stops at the session total;
> the parser goes per-turn, detects loops, and descends into subagents.

---

## Category A — tools that read existing Claude Code JSONL (no instrumentation)

| Tool | Loop detection | Per-turn | Token (in/out/cache) | Cost | Friction | Full trace | Local / License |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **claude-trace** ⭐ | ✅ same tool 3+ consecutive | ✅ | ✅ | ~ | ✅ `--reflect` | ✅ | Local · MIT |
| **AgentTrace** (luoyuctl) ⭐ | ✅ retry loops + latency gaps | ✅ | ✅ + cache | ✅ est. | ✅ incident timeline | ✅ | Local · TUI |
| **lm-assist** (tested) | ❌ (visual only) | ✅ in/out only | ✅ session-level | ✅ | ❌ | ✅ richest | Local |
| token-dashboard | ~ partial (repetition tips) | ✅ | ✅ | ✅ | ~ | ✅ | Local |
| claude-token-analyzer | ❌ | ✅ | ✅ | ✅ per-model | ~ | ~ | Local |
| claude-hindsight | ❌ | ✅ | ~ | ~ | ✅ error friction | ✅ | 100% local |
| claude-session-dashboard | ❌ | ✅ (dispatch order) | ✅ vs 200K window | ~ | ✅ errors | ✅ | Local · read-only |
| claude-code-trace (delexw) | ❌ | ✅ | ✅ | ~ | ❌ | ✅ viewer | Local |
| claude-code-otel | ❌ | ✅ | ✅ | ✅ | ~ | ✅ Grafana/Prometheus | Local · OTel |

> **Refuted during research:** claude-token-analyzer's "6 anomaly types incl.
> ExcessiveToolUse" does **not** amount to real loop detection (2 of 3 verifiers
> rejected the claim). Don't rely on it for loops.

---

## Deep dive: claude-trace (vexorkai) ⭐  — ❓ per docs

The only mainstream JSONL reader with **both loop and friction detection** built in.

- **Loop detection** — flags when the **same tool is called 3+ times consecutively**,
  the classic stuck-agent signature.
- **Friction detection** — a `--reflect` mode that surfaces where the run struggled.
- **Per-turn inspection, token breakdown, full trace** — standard reader capabilities.
- **Local, MIT.** Lightweight.
- **Why it matters here:** it ships the two rare capabilities lm-assist lacks. Best
  paired with lm-assist (dashboard) rather than replacing it.

## Deep dive: AgentTrace (luoyuctl) ⭐  — ❓ per docs

A **local-first terminal UI (TUI) + diagnostic engine** for post-run auditing. The
strongest single tool on paper for loops + friction + health scoring.

- **Multi-format parsing** — one tool for Claude Code, Codex CLI, Gemini CLI, Aider,
  Cursor exports, Cline, Hermes Agent, and generic JSONL. Unified translation layer.
- **Loop & latency auditing** — detects **repetitive retry loops** (grouped by tool,
  with normalized error messages) and **"hanging gaps"** (unusual delays between tool
  calls) to isolate where an agent stalled.
- **Token / context / cost** — accumulated token burn, **context-pressure alerts**
  (transcript growing too large), prompt-cache auditing, estimated API cost.
- **Baseline regression** — compare a run against a known-good baseline; flags when a
  workflow got slower, costlier, or touched a broader file/tool surface, and the exact
  point divergence began.
- **Tool authority / safety classification** — buckets executions into read-only vs.
  high-risk (shell writes, git writes, network), proving if a session crossed a boundary.
- **Friction timeline** — failures/exceptions/slow commands organized into a searchable
  incident timeline; details files/tools touched since last git checkpoint (blast radius).
- **Per-session health scores** — composite of tokens, loops, latency, tool errors;
  usable as a **CI/CD quality gate** (fail the build if score drops / failure rate spikes).
- **Offline reports** — exports to terminal text, Markdown, JSON, or self-contained
  interactive HTML (zero network).
- **Note:** distinct from *AgentTrace by Rxflex* (an SDK instrumentation tool — Category
  B) and *AgentTrace by tensorstax* (a trace+eval library). Three different "AgentTrace"
  projects; this luoyuctl one is the JSONL auditor.

---

## Category B — SDK instrumentation (trace your own agents; do NOT read Claude Code JSONL)

Relevant only if you build your own agents. They require adding a tracing SDK to your code.

- **Langfuse** — most mature OSS LLM observability; best-in-class token/cost
  (`cache_read_input_tokens`, tiered USD). Self-host or cloud.
- **Arize Phoenix / MLflow tracing** — OSS, run locally, OpenTelemetry-based.
- **AgentTrace (Rxflex)** — local-first step debugger, Python SDK + web tree. MIT.
- **AgentTrace (tensorstax)** — lightweight trace + eval library.

None of these ingest Claude Code's `~/.claude/projects` JSONL.

---

## What a raw JSONL parser extracts (the superset)

The JSONL holds more than any single tool surfaces. Everything below is in the file or
derivable from it:

| Signal | In JSONL | Surfaced by |
|---|:---:|---|
| Conversation (user/assistant/system) | ✅ | all |
| Tool calls + full inputs | ✅ | all |
| Tool results + `is_error` flag | ✅ | parser, claude-trace, AgentTrace → friction |
| Per-turn token usage (input/output) | ✅ | lm-assist, codetrace, parser |
| **Per-turn 5m / 1h cache split** | ✅ | **parser only** (not lm-assist) |
| **Per-turn cost** (computed) | ✅ derivable | parser, AgentTrace |
| Thinking blocks | ✅ | lm-assist, parser |
| Tool-call sequence → **loop detection** | ✅ derivable | parser, claude-trace, AgentTrace |
| Exit codes / errors → **friction** | ✅ derivable | parser, claude-trace, AgentTrace |
| User correction messages → **friction** | ✅ derivable | **parser only** |
| `parentUuid` links → **DAG / branches** | ✅ | lm-assist, parser |
| Subagent / workflow transcripts | ✅ separate files | lm-assist, parser |
| File changes, git operations | ✅ | lm-assist, parser, AgentTrace |
| Timestamps → **latency / hanging gaps** | ✅ derivable | parser, AgentTrace |
| Model, version, permission mode | ✅ | lm-assist, parser |

> **Tools answer "what happened in this session." A parser answers "analyze *why*"** —
> loops, friction, cost spikes, latency, patterns across runs.

---

## Recommendation for analyzing existing Claude Code traces

1. **lm-assist** — keep as the dashboard / data backbone (rich session browsing, DAG,
   live terminal, REST API). Best for "show me this session." ✅ tested.
2. **`lm_assist.py`** (this repo) — fills lm-assist's gaps with verified parsing: per-turn
   5m/1h cache + cost, loop detection, friction (incl. user corrections), and descent
   into workflow-agent transcripts. See [lm_assist.md](lm_assist.md) for commands. ✅ tested.

## Next steps

We have validated lm-assist + the parser. To complete the evaluation, **try the two
analysis-focused tools next** (both ❓ untested so far):

- **claude-trace (vexorkai)** — install and run on these sessions to verify its built-in
  **loop detection** (same tool 3+ consecutive) and **`--reflect` friction** output, then
  compare against `lm_assist.py loops` / `friction`.
- **AgentTrace (luoyuctl)** — evaluate its **health scores, baseline regression, latency
  gaps, tool-authority classification, and CI/CD gates** — the richest friction/loop
  feature set on paper. Confirm it parses Claude Code JSONL as advertised.

Goal: upgrade their ❓ marks to ✅/❌ with hands-on evidence, the same way we did for
lm-assist.
