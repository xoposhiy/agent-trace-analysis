# Local Trace Analysis Tools — Landscape & Verified Notes

Analysis of local-first tools for inspecting AI coding-agent runs, focused on Claude
Code (sessions stored as JSONL in `~/.claude/projects/<key>/<id>.jsonl`).

**Verification legend** — claims here are marked by how we confirmed them:
- ✅ **Verified** hands-on (lm-assist v0.1.70; `@vexor/claude-trace` v0.4.0; `luoyuctl/agenttrace` v0.5.4)
- ⚠️ **Partly true** — real, but with an important caveat
- ❌ **Not true** for the installed version
- ❓ **Unverified** — per the tool's own docs/marketing; not tested, or required
  components were not installed

> **lm-assist**, **claude-trace (vexorkai)**, and **AgentTrace (luoyuctl)** were all tested
> hands-on (see their deep-dive sections). The Category-B SDK tools (Langfuse, Phoenix,
> AgentTrace by Rxflex/tensorstax) remain ❓ — they don't read Claude Code JSONL, so they
> are out of scope for this evaluation.

---

## How tools get their data (3 categories)

- **JSONL readers** — parse Claude Code files post-hoc, no code changes
  (lm-assist, claude-trace, claude-code-trace, AgentTrace, ccusage, …)
- **SDK instrumentation** — add a tracing library to *your own* agent code; do **not**
  read Claude Code JSONL (Langfuse, Arize Phoenix)
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
| **claude-trace** (tested) ⭐ | ✅ same tool 3+ consecutive (no identical-cmd) | ⚠️ `--reflect` only | ⚠️ `--reflect` only | ❌ broken (incl. `--reflect`✅) | ⚠️ tool-error count only | ✅ | Local · MIT |
| **AgentTrace** (luoyuctl, tested) ⭐ | ✅ repeated-result loops + latency gaps | ✅ | ✅ + cache | ⚠️ est. (no opus-4-8 in catalog) | ✅ incident timeline (no user-correction) | ✅ | Local · TUI/CLI · Go MIT |
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

## Deep dive: claude-trace (vexorkai) ⭐  — ✅/⚠️/❌ tested hands-on

Tested `@vexor/claude-trace` v0.4.0 (`npm i -g @vexor/claude-trace`; bin `claude-trace`;
zero-dependency, MIT, 8★, last pushed 2026-03-03). It is primarily a **token/cost
analytics CLI** (`--tools`, `--sessions`, `--projects`, `--timeline`, `--session <id>`)
with **loop detection + a `--reflect` efficiency mode** layered on. Also ships a `/reflect`
Claude Code plugin/skill. Convenience wins: accepts a **short session-id prefix** and
**scans subagent/workflow transcripts** (111 files in our setup).

All findings below are from the head-to-head on session `a7526c9f` (the same session used
for the lm-assist + `lm_assist.py` baseline above).

- **Loop detection — ✅ real and accurate.** `--reflect` flags "same tool 3+ consecutive"
  and **agrees with `lm_assist.py loops` on every major run** — both find the 19x, 9x, 6x,
  4x Bash runs, Edit 4x, Agent 3x (claude-trace found 12 of the parser's 13; one off-by-one
  from turn-merging). **Caveats:** (a) consecutive-same-tool **only** — it does **not**
  detect *identical-command repetition*, the parser's stronger stuck signal; (b) it flags
  every 3+ run indiscriminately (14 "loops," many benign exploration runs); (c) the
  per-loop **"estimated wasted $" is a fabricated constant** (every 3x run = $2.3792 =
  invocation-share × session-cost), not measured waste.
- **`--reflect` efficiency mode — ✅ exists and its cost reconciles.** Gives a sensible
  tool breakdown (Bash 68 / Edit 22 / Agent 3 … ≈ parser's main-thread counts), the loop
  list, and CLAUDE.md suggestions. Its session cost **$82.48 ≈ parser's $81.53 (~1%)** —
  the **only trustworthy cost number** the tool produces.
- **Friction — ⚠️ much weaker than implied.** "Friction" is just `Tool errors: 1/104` plus
  a raw stdout dump of the failing command. It has **no user-correction detection** — it
  misses all 4 user corrections ("this does not work," etc.) that `lm_assist.py friction`
  surfaces, which are the most valuable friction signal.
- **Cost engine outside `--reflect` — ❌ broken and self-contradictory.**
  - `--tools` dollar amounts sum to **~$4,426** while stated total cost is **$184** (~24×
    inflated; the % column is internally consistent, the dollars are not).
  - Same session reads **$175.78** in `--sessions`/overall but **$82.48** in `--reflect` —
    a ~2× self-disagreement.
  - `--session` drill-down attributes **$116 to a 3,238-token Bash result** (≈$36,000/M,
    impossible) and sees only **3 Bash + 1 Read** where `--reflect` correctly sees 68 Bash
    — two parsing code paths, one badly broken.
- **Net:** Use it for **loop detection** and the **`--reflect`** summary (cost + loops +
  CLAUDE.md tips) — those work and reconcile with the parser. **Do not trust** its
  `--tools` / `--sessions` / `--session` / overall **cost attribution**, and don't rely on
  it for friction. The parser still wins on identical-command loops and user-correction
  friction; lm-assist still wins on session-level cost accuracy and the DAG/live views.

## Deep dive: AgentTrace (luoyuctl) ⭐  — ✅/⚠️/❌ tested hands-on

Tested **`luoyuctl/agenttrace` v0.5.4** (Go, MIT, 71★; `github.com/luoyuctl/agenttrace`,
*not* a fork). Installed by downloading the prebuilt `agenttrace-darwin-arm64` binary from
the v0.5.4 release and **verifying its published sha256** (matched). It is a **local-first
TUI + report/diagnostic engine** that auto-discovers many agent log dirs. Of the three
tools tested, this is **by far the richest analysis engine** — and most headline claims
held up. Verified on the same project dir as the lm-assist/parser baseline (session
`a7526c9f` + its workflow subagents).

> *Identity note:* this is the **luoyuctl** JSONL auditor — distinct from *AgentTrace by
> Rxflex* (SDK step-debugger, Category B) and *AgentTrace by tensorstax* (trace+eval lib).
> Caveat on the author: the `luoyuctl` GitHub account is mostly **mass-forked "awesome-*"
> lists** (271 repos), but `agenttrace` itself is an original, non-fork Go project with
> real releases, CI, homebrew tap, and tests.

**Headless usage** (TUI not needed): `--doctor`, `--overview -f json|markdown|html`,
`--search <q>`, `--latest`, `--waste`, `--compare`, `--baseline`, CI-gate flags, `-d <dir>`
to scope, `--list-models`. The TUI (`agenttrace`) is the default interactive view.

- **Multi-format parsing — ✅.** `--doctor` auto-discovered Claude Code (✅), Codex CLI
  (✅), Gemini CLI tmp (✅) plus slots for Qwen/Cline/OpenCode/Aider/etc. Genuinely
  multi-agent, unlike the Claude-only tools.
- **Loop & latency auditing — ✅, and smarter than claude-trace.** Its **failure-loop**
  signal is *"tool X repeated the same **result** N times"* — a genuine stuck signal, not
  claude-trace's naive "3+ consecutive same tool." Verified: it flagged a real workflow
  subagent (`agent-ac08d2b4…`, 69 `StructuredOutput` calls) as *"StructuredOutput repeated
  the same result 8×"* (HIGH, health 58 — the worst session), and `Edit repeated the same
  result 4×` in `63f0bb29`. **Notably it did NOT false-flag `a7526c9f`'s 19× Bash run**
  (those were distinct exploration commands) — where claude-trace flagged all 14 runs
  indiscriminately. **Latency/"hanging gaps" ✅** (`109.5s`, `56811.9s` idle gaps).
- **Token / context / cost — ✅ with a pricing caveat.** "Burn divergence" gives both
  **loop-cost-within-total** (`$1.5375 loop cost inside $28.2448 total`) and **context
  pressure** (`tokens per assistant turn`). Per-session cost ✅ matches the parser:
  `a7526c9f` main = **$36.22** vs parser main-thread **$35.68**. ⚠️ **Pricing caveat:**
  its model catalog (`--list-models`) has `claude-opus-4` ($15/$75) and `claude-opus-4.5`
  ($5/$25) but **no `claude-opus-4-8`** — so the current model is **fuzzy-matched to older
  Opus pricing**. Run `--update-pricing` (pulls LiteLLM rates) before trusting dollars.
- **Baseline regression — ✅.** `--baseline <json>` + `--baseline-max-*-delta-pct`
  thresholds emit a `baseline_comparison` block.
- **Tool authority / safety classification — ✅ but incomplete.** Buckets calls into
  `git_write / network_access / package_install / read_only_files / shell_exec /
  test_or_build / write_files`. ⚠️ But **154 of 662 calls (~23%) land in
  `unknown_authority`** — its classifier doesn't recognize newer Claude Code tools
  (Agent, StructuredOutput, WebFetch, Workflow, Skill…), and `unknown` was the "highest"
  authority reported. Treat the safety read as partial.
- **Friction / incident timeline — ✅ (tool/timing only).** Per-session timeline of
  milestones, idle gaps, failure loops, burn divergence, touched-surface (blast radius:
  files + tool counts), with severities. ❌ **No user-correction detection** — like
  claude-trace, it reads tool/result/timing signals, not the human's reply sentiment;
  the parser's user-correction friction remains unique.
- **Per-session health + CI gates — ✅ verified working.** Health scores per session
  (`a7526c9f`=70, the looping subagent=58). CI gates **functionally tested**: a lenient
  gate (`--fail-under-health 80 --max-tool-fail-rate 15`) exited **0**; a strict gate
  (`--fail-under-health 99 --max-tool-fail-rate 5`) exited **2**. Plus a health-trend
  (direction up/down, `regressing` bool).
- **Anomaly typing — ✅** (`hanging`, `latency`, `tool_failures`). ⚠️ **Latency/"hanging"
  false-positives on human-paced sessions:** a `56811.9s` (~15.8h) "idle gap" and
  "hanging: now" flags are just sessions left open while the human stepped away, not real
  stalls — for interactive Claude Code use these need human judgement.
- **Offline reports — ✅.** Verified `text`, `json`, `markdown`; `html` flag present.
  Zero network, fully local.
- **Net:** the **strongest analysis engine of the three** — real failure-loop detection
  (better-targeted than claude-trace), latency gaps, burn/context-pressure, health scores,
  tool-authority, baseline regression, working CI gates, and multi-agent + multi-format.
  Watch three things: **pricing** (no opus-4-8 in catalog → `--update-pricing`),
  **tool-authority gaps** (~23% unknown), and **latency false-positives** on interactive
  sessions. Still no **user-correction friction** (parser-only). Best role here: the
  **diagnostic/CI layer**, complementing lm-assist (browse/DAG/live) and the parser
  (per-turn cache split + user-correction friction).

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

All three JSONL-reader tools are now validated hands-on (see deep dives):

- ✅ **claude-trace (vexorkai)** — *done.* Loop detection ✅ accurate, `--reflect` cost ✅
  matches the parser ($82 vs $81); cost attribution outside `--reflect` is ❌ broken and
  "friction" ⚠️ misses user corrections.
- ✅ **AgentTrace (luoyuctl)** — *done.* Richest analysis engine: repeated-result failure
  loops ✅ (better-targeted than claude-trace), latency gaps ✅, burn/context-pressure ✅,
  health scores + working CI gates ✅, baseline regression ✅, tool-authority ⚠️ (~23%
  unknown), multi-format ✅. Caveats: pricing catalog lacks `claude-opus-4-8`
  (`--update-pricing`), latency false-positives on human-paced sessions, no
  user-correction friction.

## AgentTrace → LLM-as-judge (DeepEval / Inspect-AI) workflow

Use `AgentTrace` (luoyuctl) to surface candidate regions (repeated-result loops, idle gaps, high-cost turns), then feed those regions to a local LLM-as-judge to answer "why."

### Build-your-own frameworks for LLM-as-judge root-cause analysis

All run **offline only if pointed at a local judge** (Ollama/vLLM); defaults call cloud APIs. All need a small custom loader to feed JSONL turns in.

| Framework | License | Why it fits | Local judge | Caveat |
|---|---|---|---|---|
| **DeepEval G-Eval** | — | Plain-language criteria → outputs **score + reason + explanation**. Fastest path to a root-cause classifier. | ✅ `deepeval set-ollama` | Non-deterministic (use DAGMetric for determinism) |
| **Inspect AI** (UK AISI) | MIT | Most rigorous; custom scorers with Score.explanation, model-graded scorers. Reusable classifier. | ✅ local provider | Targets its own eval logs; needs JSONL→EvalLog converter |
| **promptfoo llm-rubric** | MIT | YAML-config judge → {reason, score, pass}. Pick this if you prefer config over Python. | ✅ vLLM/Ollama provider | Default grader is GPT-4o; "Tracing" is live OTLP only |

### Next steps

**The winning architecture:** Use `AgentTrace` as the **signal harvester** (per-turn cache/cost, identical-command loops, user-correction friction, tool failures), then feed suspicious regions to a **local G-Eval or Inspect-AI rubric** that surfaces the actual error text and explains the root cause. That's the combination that exceeds AgentTrace alone.

