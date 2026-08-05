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

**Session list** — every session newest-first with Claude Code's own title, id,
last message time, project, tokens, tool calls, duration and subagent count.
Live project filter.

**Session page** — header, stats, and the **vertical bar**: one coloured block
per stretch of activity, Y-axis selector (tokens / time / messages), hover
readout, legend with per-kind counts.

**Every block is clickable** and opens its own page in a new tab, listing each
step: the file that was read, the command that ran with its full text, the
result, and whether it failed. Reachable by keyboard — Tab to a block, Enter to
open — since a block can be 3px tall.

**Subagents drill down two levels.** Agents spawned back to back are one solid
orange band; agents spawned at different times stay separate bands. Clicking a
band lists the agents under it, and clicking an agent opens **its own vertical
bar** — same colours, same Y-axis selector, its own clickable blocks.

**Block labels name the work**, not just the volume:

```
write · split_advisor.py ×7 · 10 steps          repetition is the finding
execute · Confirm new tests fail +2 · 3 steps   Bash's own description
read · 6 files in 4 dirs · 8 steps              scattered work is counted
read · btbkb1kha.output · 8 steps · 1 failed    failures always shown
```

**Context-window attribution** — the session's billed `input + output +
cache_creation` divided across the blocks that caused it, so a `Read` is charged
for the file it pulled into context. Sums to the header exactly.

**Block classification** — rules for unambiguous tools, LLM judge for `Bash`
(only the command text tells `cat foo.py` from `pytest`). Cached to
`~/.cache/tracelens/tool_kinds.json` by `hash(tool + input)`, so a command is
judged once ever across all sessions.

Not yet: problem detection and the severity filter (control present but
disabled), the problem-specific pane, and money as a Y-axis metric (needs a
price table).

### Colours

Six kinds, one hue each — green read, blue write, red execute, grey
coordination, orange subagents, purple chatting-with-user. Steps were picked for
colour-blind separation on a dark surface and re-validated when subagents became
a solid fill (orange↔red ΔE 14.7 deutan). Chatting-with-user also differs in
*form* — a thin rule, never a fill — because purple↔blue cannot clear the CVD
floor on hue alone. Rationale and validator output are in the header of
`web/bar.js`.

## Tests

```bash
pytest tests -q                 # from Final_app/  — 308 tests, ~0.9s
node --test tests/bar.test.js   # 22 layout tests (also run by pytest)
```

Tests build synthetic transcripts in `tmp_path` and point `CLAUDE_CONFIG_DIR` at
them — nothing reads your real `~/.claude`, and nothing calls the LLM, so the
suite runs offline and off-VPN. The bar's layout maths is unit-tested through
Node; when Node is absent those tests skip rather than fail.

Every change ships with tests; see `CLAUDE.md` §6.

## Layout

```
ir/models.py            Event / Block / Session — the IR
adapters/claude_code.py raw JSONL -> IR, including subagent files
analysis/blocks.py      event stream -> contiguous activity blocks
analysis/classify.py    block kind: rules + LLM judge for Bash, disk-cached
analysis/attribution.py billed tokens -> the blocks that caused them
analysis/labels.py      what a block calls itself on the bar
analysis/steps.py       per-step detail for the block page
analysis/tokens.py      exact per-content-block counts (optional, needs VPN)
api/app.py              FastAPI: JSON API + pages
web/                    vanilla JS + CSS, no build step
config.py               .env loading, LLM diagnostics
```

See `DESIGN.md` for the pipeline, the URL scheme and what was taken from the
Entire CLI.

## Three things worth knowing

**Tokens exclude cache reads.** `cache_read` is the whole prompt prefix re-read
on every message, so summing it counts the same context hundreds of times — one
real session totalled 13.2M tokens of which 12.5M (95%) was cache reads. The UI
shows `input + output + cache_creation`; hover for the breakdown.

**Cost is per API call, not per step.** `usage` is reported once per
`message.id`, and streaming splits that message across several JSONL lines that
each repeat it. A per-block token figure is therefore *derived* — see
`DESIGN.md`.

**Project labels are a heuristic.** Claude Code's directory slug replaces both
`/` and `-` with `-`, so the original path is unrecoverable. The label is a best
guess; the raw slug is in the tooltip.
