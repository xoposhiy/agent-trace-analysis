# Agent Trace Analysis — Tooling & `lm_assist.py`

Analyzing Claude Code agent runs. Claude Code stores every session as an append-only
JSONL file in `~/.claude/projects/<encoded-path>/<session-id>.jsonl`. That file is the
**complete source of truth**; observability tools are just different windows onto it.

This doc covers: (1) the tool landscape and how they compare, (2) what a raw JSONL
parser can extract beyond any tool, and (3) the `lm_assist.py` command reference.

---

## 1. Tool landscape

Three categories of tooling, by how they get their data:

- **JSONL readers** — parse Claude Code's `~/.claude/projects` files post-hoc, no code
  changes. (lm-assist, claude-code-trace, claude-trace, ccusage)
- **SDK instrumentation** — you add a tracing library to *your own* agent code; they do
  **not** read Claude Code JSONL. (AgentTrace, Langfuse, Arize Phoenix)
- **Custom parser** — read the JSONL directly and compute derived signals. (`lm_assist.py`)

### Comparison

> lm-assist is hands-on verified. AgentTrace and claude-code-trace are from research,
> not installed here — treat as "per docs." claude-trace (vexorkai) ≠ claude-code-trace
> (delexw): the former adds loop/friction detection, the latter is a viewer.

| Capability | AgentTrace (Rxflex) | lm-assist | claude-code-trace (delexw) | Raw JSONL parser |
|---|:---:|:---:|:---:|:---:|
| **Type** | SDK instrument | JSONL reader | JSONL reader | parser |
| **Reads Claude Code JSONL** | ❌ | ✅ | ✅ | ✅ |
| **Interface** | Py SDK + web | Web UI + REST API | Desktop/web/TUI | CLI |
| **Local / open-source** | ✅ MIT | ✅ local | ✅ local | ✅ |
| Loop / stuck detection | ❌ | ❌ | ❌ | ✅ |
| Per-turn / per-step | ✅ | ✅ in/out only | ✅ | ✅ |
| Token breakdown (in/out/cache) | ~ | ✅ session-level | ✅ | ✅ |
| Cost tracking | ~ | ✅ session + per-model | ~ | ✅ (est.) |
| Friction (errors/retries/corrections) | ❌ | ❌ | ❌ | ✅ |
| Full trace (convo/tools/DAG/files/git) | ✅ decorated fns | ✅ richest | ✅ convo+tools | ✅ |
| Real-time monitoring | ❌ | ✅ live + SSE | ✅ live tail | ❌ |

**Takeaways**
- No mainstream JSONL reader detects **loops** or **friction** out of the box (only
  claude-trace by vexorkai). A parser fills this gap.
- **AgentTrace doesn't read Claude Code traces** — it instruments your own agents.
- **lm-assist is the richest reader** (DAG, files, git, 155+ API endpoints) but all its
  cache/cost detail is **session-level, not per-turn**.

---

## 2. What a raw JSONL parser extracts (the superset)

The JSONL holds more than any tool surfaces. Everything below is in the file or derivable:

| Signal | In JSONL | Surfaced by |
|---|:---:|---|
| Conversation (user/assistant/system) | ✅ | all |
| Tool calls + full inputs | ✅ | all |
| Tool results + `is_error` flag | ✅ | parser → friction |
| Per-turn token usage (input/output) | ✅ | lm-assist, codetrace, parser |
| **Per-turn 5m / 1h cache split** | ✅ | **parser only** |
| **Per-turn cost** (computed) | ✅ derivable | **parser only** |
| Thinking blocks | ✅ | lm-assist, parser |
| Tool-call sequence → **loop detection** | ✅ derivable | parser, claude-trace |
| Exit codes / errors → **friction** | ✅ derivable | parser, claude-trace |
| User correction messages → **friction** | ✅ derivable | **parser only** |
| `parentUuid` links → **DAG / branches** | ✅ | lm-assist, parser |
| Subagent / workflow transcripts | ✅ separate files | lm-assist, parser |
| File changes, git operations | ✅ | lm-assist, parser |
| Timestamps → **latency per turn** | ✅ derivable | parser |
| Model, version, permission mode | ✅ | lm-assist, parser |

> **Tools answer "what happened in this session." A JSONL parser answers "analyze *why*"**
> — loops, friction, cost spikes, and patterns across runs.

---

## 3. `lm_assist.py` — command reference

A CLI that reads two sources: the **lm-assist REST API** (`http://localhost:3100`) for
session lists/summaries, and the **raw JSONL** for per-turn and friction detail.

**Requires:** lm-assist service running (`lm-assist status`) and `requests` in the venv.

```bash
.venv/bin/python lm_assist.py <command> [session_id]
# or: source .venv/bin/activate  →  python lm_assist.py <command> [session_id]
```

Get a session id with `sessions`. Override the API host with `LM_ASSIST_URL`.

| Command | Source | Shows |
|---|:---:|---|
| `sessions` | API | List all sessions (id, prompts, project) |
| `projects` | API | Projects with session counts |
| `show <id>` | API | Summary: model, turns, cost, tools, subagents |
| `tools <id>` | API | Tool-use frequency |
| `tokens <id>` | API | Token totals (input/output/cache); `--all` for every session |
| `turns <id>` | JSONL | Per-turn token + estimated cost breakdown, with column legend |
| `detail <id>` | JSONL | Per-turn: tokens, cost, thinking, tool calls; workflow turns annotated with spawned-agent rollup |
| `subagents <id>` | JSONL | Descends into workflow/subagent transcripts: per-agent tokens, cost, tools |
| `loops <id>` | JSONL | Repetition/stuck detection: same tool 3+ consecutive, identical commands repeated |
| `friction <id>` | JSONL | Errors, non-zero exits, repeated commands, user corrections |
| `cost` | API | Cost rollup across all sessions |

### Per-turn columns (`turns` / `detail`)

| Column | Meaning | Rate (Opus 4.x) |
|---|---|---|
| `#` | turn number (row label) | — |
| `in` | new uncached input tokens | $15 / M |
| `out` | tokens the model generated (priciest) | $75 / M |
| `c5m` | tokens written to 5-min cache | $18.75 / M |
| `c1h` | tokens written to 1-hour cache | $30 / M |
| `cRead` | tokens read back from cache (grows all session; dominant cost) | $1.50 / M |
| `$` | estimated cost; `<-- spike` = over $0.10 | — |
| `T` (detail) | thinking block present | — |

---

## Notes & caveats

- **Token/tool counts are exact** (from JSONL). **USD is an estimate** using the `PRICE`
  dict at the top of the script — adjust to your model/plan. `turns` cross-checks its
  total against the API's reported cost and warns if rates look off.
- **lm-assist exposes cache/cost only at the session level**, never per-turn. The
  per-turn 5m/1h cache split exists *only* in the raw JSONL — that's what `turns` parses.
  (Some third-party descriptions claim lm-assist does this per-turn; the installed
  v0.1.70 does not.)
- **"Workflow agents" vs subagents:** agents spawned by the `Workflow` tool live in
  `<id>/subagents/workflows/<wf_id>/`, separate from the main thread. `subagents`
  aggregates them; `detail` annotates the launching turn. Direct `Task`/`Agent` spawns
  appear inline in the main session instead.
- **Loop detection nuance:** consecutive same-tool runs are normal during exploration.
  The stronger stuck-agent signal is **identical commands repeated** — `loops` reports
  both separately.
