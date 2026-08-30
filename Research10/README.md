# Claude Split Advisor

A small **local** tool that scans your own Claude Code sessions and tells you which
ones were worth splitting into separate sessions — and roughly how much money a split
would have saved. It runs entirely on your machine against the transcripts Claude Code
already writes; nothing is uploaded.

It works for **anyone's** Claude Code traces, not one specific setup: paths resolve
per-user at runtime, and the LLM features (task summaries + task-switch detection) work
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
it also (4) asks a model whether you genuinely **switched to an unrelated task**
mid-session and prices a split there too, and (5) writes a one-line **task summary** of
what each considered session was about — because a bare session ID tells you nothing.

Output is a console summary plus a Markdown report (`split_report.md`).

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

By default the tool uses an LLM (for summaries + task-switch detection) if one is
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

## Enabling LLM summaries

The LLM features (task summaries + task-switch detection) are **on by default** — but
they need an API key you provide. Nothing is baked in: you point the tool at your own
OpenAI-compatible endpoint, so it works against any provider and spends only your quota.

**If you have an OpenAI key**, that's the whole setup (the default model is `gpt-4o-mini`):

```bash
export OPENAI_API_KEY=sk-...
split-advisor                       # summaries + task-switch on by default
```

**For any other OpenAI-compatible provider** (a Claude/LiteLLM proxy, OpenRouter, a
local server, …), also set the endpoint and a model name it exposes:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-endpoint/v1
export SPLIT_ADVISOR_MODEL=anthropic/claude-haiku-4-5
split-advisor
```

Tip: put those `export` lines in your shell profile (`~/.zshrc` / `~/.bashrc`) so every
run picks them up. To confirm a run used the LLM, its first line reads
*"…with LLM task summaries + task-switch judge…"*; if the key is missing you'll instead
see a short hint and *"…structural heuristics only; no LLM…"*.

**No key / don't want to use an LLM?** Run `split-advisor --no-llm` (or just leave the
key unset). You still get the full structural analysis and pricing — only the summaries
and task-switch options are omitted, and sessions are shown by ID.

If the LLM client can't be created or a call fails, the tool prints the reason and
continues with the offline structural analysis rather than erroring out.

## Notes

- All dollar figures are token-based **estimates** at blended Claude (Opus-4.x) rates,
  computed from each session's real recorded token usage.
- The saving % depends on **where** the split falls: a single split can remove at most
  ~half of the "context re-read" cost (≈16% of total, at a mid-session split), and less
  for the early plan-mode split. It never approaches the ~70% theoretical ideal because
  re-establishing context via a summary is never free.
