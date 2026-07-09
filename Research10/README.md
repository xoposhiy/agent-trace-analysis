# Claude Split Advisor

A small **local** tool that scans your own Claude Code sessions and tells you which
ones were worth splitting into separate sessions — and roughly how much money a split
would have saved. It runs entirely on your machine against the transcripts Claude Code
already writes; nothing is uploaded.

It works for **anyone's** Claude Code traces, not one specific setup: paths resolve
per-user at runtime, the default analysis needs no API key and no network, and the
optional LLM feature works against any provider you configure.

## What it does

For every session under `~/.claude/projects/**/*.jsonl` it:

1. Rebuilds the ordered tool-call sequence and detects the session's shape —
   a front-loaded **reading phase** (a *plan-mode* opportunity) or a mid-session
   **reading burst** (a *sub-agent* opportunity).
2. Prices a semantic split at that boundary with a chunk-split cost model that
   charges the real cost of carrying a summary across the split.
3. **Suggests** the split only if the modelled saving clears both a percentage floor
   and a dollar floor — so tiny sessions never nag you.

Output is a console summary plus a Markdown report (`split_report.md`).

## Install

Requires Python 3.8+. Installing in a virtual environment is recommended.

```bash
# from this directory
pip install .

# with the optional LLM task-switch judge
pip install ".[llm]"
```

This puts a `split-advisor` command on your PATH. (You can also just run it in place
with `python split_advisor.py` — no install needed.)

## Usage

```bash
split-advisor                       # scan all sessions, write ./split_report.md
split-advisor --summary-only        # console summary only, no file
split-advisor --project my-repo     # only sessions whose project dir matches
split-advisor --min-pct 8 --min-dollars 0.25   # loosen the suggestion thresholds
split-advisor --out ~/report.md     # choose where the report is written
```

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
| `--llm` | off | Also use the LLM task-switch judge (see below). |
| `--llm-base-url` | `$OPENAI_BASE_URL` or SDK default | OpenAI-compatible endpoint. |
| `--llm-model` | `$SPLIT_ADVISOR_MODEL` or `gpt-4o-mini` | Model name — set it to match your endpoint. |

## Optional: the LLM task-switch judge

The default analysis is heuristic and fully offline. Adding `--llm` also asks a model
whether you genuinely switched to an unrelated task mid-session, and considers a split
there too. It uses the OpenAI-compatible chat API, so it works against **any** provider:

```bash
export OPENAI_API_KEY=sk-...
# OpenAI directly:
split-advisor --llm --llm-model gpt-4o-mini
# a Claude-compatible proxy / OpenRouter:
export OPENAI_BASE_URL=https://your-proxy.example.com
split-advisor --llm --llm-model anthropic/claude-haiku-4-5
```

If the LLM client can't be created or a call fails, the tool logs it and continues
with heuristic-only detection.

## Notes

- All dollar figures are token-based **estimates** at blended Claude (Opus-4.x) rates,
  computed from each session's real recorded token usage.
- The saving % depends on **where** the split falls: a single split can remove at most
  ~half of the "context re-read" cost (≈16% of total, at a mid-session split), and less
  for the early plan-mode split. It never approaches the ~70% theoretical ideal because
  re-establishing context via a summary is never free.
