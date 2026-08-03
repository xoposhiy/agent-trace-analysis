# TraceLens

Local dashboard for Claude Code session traces. Reads
`~/.claude/projects/**/*.jsonl` directly — nothing is uploaded anywhere.

## Run

The import package is the directory itself, `Final_app`, so `uvicorn` is run
from the **parent** directory. No install is needed to run or test.

```bash
pip install fastapi uvicorn python-dotenv openai   # or: pip install -e "Final_app[llm]"
cp Final_app/.env.example Final_app/.env           # add OPENAI_API_KEY / OPENAI_BASE_URL
python -m uvicorn Final_app.api.app:app --reload --port 8765
```

Then open <http://localhost:8765>.

The LLM is **optional**. With no key, sessions fall back to the title Claude
Code writes itself (`ai-title`). The LiteLLM proxy is VPN-only; without VPN
calls time out and the UI says so instead of hanging.

## What works today

- **Last sessions** — every session newest-first with title, LLM summary, id,
  last message time, project, tokens, tool calls, duration, and subagent count.
  Live project filter.
- **Session page** — header, stats, and the **vertical bar**: one coloured block
  per stretch of activity, Y-axis selector (tokens / time / messages), hover
  readout, legend with per-kind counts.
- **Summaries** — automatic, generated in the background 4 at a time so the list
  never waits on the LLM. Cached to `~/.cache/tracelens/summaries.json`;
  one-time cost per session.
- **Block classification** — rules for unambiguous tools, LLM judge for `Bash`
  (only the command text tells `cat foo.py` from `pytest`). Cached to
  `~/.cache/tracelens/tool_kinds.json` by `hash(tool + input)`, so a command is
  judged once ever across all sessions.

Not yet: problem detection and the severity filter (control present but
disabled), the problem-specific pane, and money as a Y-axis metric (needs a
price table).

### Colours

Six kinds, one hue each — green read, blue write, red execute, grey
coordination, orange subagents, purple chatting-with-user. The steps were picked
for colour-blind separation on a dark surface; subagents and chatting-with-user
also differ in *form*, so hue is never the only cue. Rationale and the validator
output are in the header of `web/bar.js`.

## Tests

```bash
pytest tests -q               # from Final_app/
pytest Final_app/tests -q     # or from the parent
```

152 tests, ~0.5s. They build synthetic transcripts in `tmp_path` and point
`CLAUDE_CONFIG_DIR` at them — nothing reads your real `~/.claude`, and nothing
calls the LLM, so the suite runs offline and off-VPN.

Every change ships with tests; see `CLAUDE.md` §6.

## Layout

```
ir/models.py            Event / Block / Session — the IR
adapters/claude_code.py raw JSONL -> IR, including subagent files
analysis/blocks.py      event stream -> contiguous activity blocks
analysis/classify.py    block kind: rules + LLM judge for Bash, disk-cached
judge/summary.py        LLM session summaries, disk-cached
api/app.py              FastAPI: JSON API + pages
web/                    vanilla JS + CSS, no build step
config.py               .env loading, LLM diagnostics
```

See `DESIGN.md` for the IR design and what was taken from the Entire CLI.

## Two things worth knowing

**Tokens exclude cache reads.** `cache_read` is the whole prompt prefix re-read
on every message, so summing it counts the same context hundreds of times — one
real session totalled 13.2M tokens of which 12.5M (95%) was cache reads. The UI
shows `input + output + cache_creation`; hover for the breakdown.

**Project labels are a heuristic.** Claude Code's directory slug replaces both
`/` and `-` with `-`, so the original path is unrecoverable. The label is a best
guess; the raw slug is in the tooltip.
