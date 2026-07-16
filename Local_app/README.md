# Claude Split Advisor

A small **local** tool that scans your own Claude Code sessions and tells you which
ones were worth splitting into separate sessions — and roughly how much money a split
would have saved. It runs entirely on your machine against the transcripts Claude Code
already writes; nothing is uploaded.

It works for **anyone's** Claude Code traces, not one specific setup: paths resolve
per-user at runtime, and the LLM features (task summaries + task-forest detection) work
against any OpenAI-compatible provider **you** configure with your own key. With no key
set, it still runs fully offline as a structural-only analysis.

## What it does

For every session under `~/.claude/projects/**/*.jsonl` it:

1. Rebuilds the ordered tool-call sequence and detects the session's shape —
   a front-loaded **reading phase** (a *plan-mode* opportunity) or a mid-session
   **reading burst** (a *sub-agent* opportunity).
2. Prices a semantic split at that boundary with a chunk-split cost model that
   charges the real cost of carrying a summary across the split.
3. **Suggests** the split only if the modelled saving clears both a percentage floor
   and a dollar floor — so tiny sessions never nag you.

When an LLM endpoint is reachable (the default — see [Enabling LLM summaries](#enabling-llm-summaries)),
it also (4) segments the session into its **task forest** and prices a split at every
boundary, and (5) writes a one-line **task summary** of what each considered session was
about — because a bare session ID tells you nothing.

### The task forest

Rather than a single yes/no "did you switch tasks", the LLM assigns every one of your
messages to a task and returns the tasks **in order**, letting a task **recur**:

```
T1 → T2 → T1 → T3
```

Task ids are **hierarchical**, and each split point is priced in dollars:

- **Top-level tasks** (`T1`, `T2`, `T3`) are genuinely **independent** goals. Moving to a
  new one you never return to is an **independent task switch** — a fresh-session split
  point. A top-level task that *comes back* (`T1 → T2 → T1`) is **interleaving** — the
  strongest signal the work belonged in separate sessions.
- **Sub-agent children** (`T1.1`, `T1.2`) are related-but-tangential side-tasks *within*
  `T1` — a self-contained detour (or an interruption you returned from). These are
  **sub-agent opportunities**: work that could have run in an isolated sub-agent so the
  main thread never carried its context. They're priced with a distinct "excise-and-
  rejoin" model, not as a full session split.

The report shows each session's forest, its task legend, an interleaving callout, the
sub-agent tangents, and a headline for splitting the whole forest into separate sessions.

Output is a console summary plus a Markdown report (`split_report.md`).

## Web dashboard (local)

There's also a local web dashboard that shows the same analysis with **colored
task-forest timeline strips** (one row per session, segments colored by task, so
interleaving is visible at a glance), live saving thresholds, and a project filter.
Suggestions are split into two tabs — **Split suggestions** (task switches + sub-agents,
from the forest) and **Plan-mode opportunities** (the structural front-loaded-reading
hint) — since those are different kinds of advice. Each suggested session has a **View
graph** button that opens a chart of the modelled context growth with the task bands and
split points marked, so you can see where the switches and sub-agents fall.

```bash
pip install ".[web]"      # FastAPI + uvicorn + the LLM extra
split-advisor serve       # -> http://127.0.0.1:8000  (Ctrl-C to stop)
# or, without installing:  python split_advisor.py serve --port 8000
```

It runs entirely on `localhost`; nothing is uploaded. Press **Refresh** in the page
to analyze your sessions. Results are cached to `~/.claude/split_advisor_cache.json`,
and **refresh is incremental** — it re-analyzes only sessions that are new or that
changed since last time, and serves everything else from the cache (so it makes no
LLM calls for sessions already analyzed). Delete the cache file to force a full
re-analysis.

## Install

Requires Python 3.8+. Installing in a virtual environment is recommended.

```bash
# from this directory — includes the LLM extra (openai) so summaries work out of the box
pip install ".[llm]"

# structural-only, no LLM dependency at all
pip install .
```

This puts a `split-advisor` command on your PATH. (You can also just run it in place
with `python split_advisor.py` — no install needed.)

## Usage

```bash
split-advisor                       # scan all sessions, write ./split_report.md
split-advisor --no-llm              # fully offline, structural-only (session IDs, no summaries)
split-advisor --summary-only        # console summary only, no file
split-advisor --project my-repo     # only sessions whose project dir matches
split-advisor --min-pct 8 --min-dollars 0.25   # loosen the suggestion thresholds
split-advisor --out ~/report.md     # choose where the report is written
```

By default the tool uses an LLM (for summaries + task-forest detection) if one is
reachable; set your key first (see below). If no endpoint is configured it prints a
short hint and falls back to the offline structural analysis automatically.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--projects-dir` | `$CLAUDE_CONFIG_DIR/projects` or `~/.claude/projects` | Where Claude Code stores transcripts. |
| `--project` | (all) | Only sessions whose project-dir name contains this substring. |
| `--min-pct` | `10.0` | Minimum saving **%** to suggest a split. |
| `--min-dollars` | `0.50` | Minimum saving **$** to suggest a split (both floors must clear). |
| `--alpha` | `0.10` | Carried-summary size as a fraction of the first chunk (the "summary parameter"). |
| `--out` | `./split_report.md` | Markdown report path. |
| `--summary-only` | off | Console only; don't write the report. |
| `--no-llm` | off | Disable all LLM use — structural heuristics only, fully offline. |
| `--llm-base-url` | `$OPENAI_BASE_URL` or SDK default | OpenAI-compatible endpoint. |
| `--llm-model` | `$SPLIT_ADVISOR_MODEL` or `gpt-4o-mini` | Model name — set it to match your endpoint. |

## Enabling the LLM (task forest + summaries)

The LLM features are on by default but need an API key you provide, against any
OpenAI-compatible endpoint. Configure it with a project **`.env` file** — it's loaded
automatically and git-ignored, so your key is never committed:

```bash
cp .env.example .env
# then edit .env:
#   OPENAI_API_KEY=sk-...
#   OPENAI_BASE_URL=https://your-endpoint/v1        # only for non-OpenAI providers
#   SPLIT_ADVISOR_MODEL=anthropic/claude-haiku-4-5  # match your endpoint (default: gpt-4o-mini)
```

Installing the `llm` or `web` extra pulls in `python-dotenv`, so `.env` works out of the
box. (If you'd rather, exporting the same variables in your shell also works and takes
precedence over the file.)

**No key?** Run `split-advisor --no-llm` (or just leave it unset): you still get the full
structural analysis — only the task forest and summaries are omitted. If a key is set but
a call fails, the tool prints the reason and falls back to structural analysis.

## Notes

- All dollar figures are token-based **estimates** at blended Claude (Opus-4.x) rates,
  computed from each session's real recorded token usage.
- The saving % depends on **where** the split falls and how **cache-heavy** the session
  is: a single mid-session split removes the most (up to ~half of the "context re-read"
  cost — often 15–25% of total on long, read-heavy sessions), an early plan-mode split
  less, and a sub-agent offload less still (it excises one segment, not half the ramp).
  Splitting the whole forest into separate sessions saves more than any single split, but
  none approach the ~70% theoretical ideal because carrying a summary is never free.
- **Message vs turn:** a *message* is one prompt you typed; a *turn* is one assistant step
  (tool call or reply). Split points are labelled by message, the cost math runs in turns.
