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

**Context-window attribution** — every billed token divided across the blocks
that caused it, so a `Read` is charged for the file it pulled into context. Two
figures per block: the work it added, and what later calls paid to re-read it
while it stayed in the window. Sums to the header exactly.

**Block classification** — rules for unambiguous tools, LLM judge for `Bash`
(only the command text tells `cat foo.py` from `pytest`). Cached to
`~/.cache/tracelens/tool_kinds.json` by `hash(tool + input)`, so a command is
judged once ever across all sessions.

**Problems tab** — pages over every session's own detectors and lists what
they find, sortable by dollar or percent saving, each row linking to its
session. Two detectors run today: a missed **plan-mode** opportunity (a
front-loaded reading phase never wrapped in plan mode) and **independent task
switching** (a session that pursued several unrelated goals at once,
LLM-segmented, with no offline fallback). Each session page shows its own
problems in a **problem-specific panel**, and each problem opens its own page
with the session's plain bar beside the same bar with the suggested split cut
into it.

Not yet: the child/tangent "sub-agent opportunity" pattern, and a richer
hover readout.

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
pytest tests -q                 # from Final_app/  — 322 tests, ~0.9s
node --test tests/*.test.js     # 32 frontend tests (also run by pytest)
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
analysis/problems.py    runs every problem detector over a session
analysis/plan_mode.py   detects a missed plan-mode opportunity
analysis/task_forest.py detects independent task switching (LLM-only)
api/app.py              FastAPI: JSON API + pages
web/                    vanilla JS + CSS, no build step
config.py               .env loading, LLM diagnostics
```

See `DESIGN.md` for the pipeline, the URL scheme and what was taken from the
Entire CLI.

## Two things worth knowing

**Cost is per API call, not per step.** `usage` is reported once per
`message.id`, and streaming splits that message across several JSONL lines that
each repeat it. A per-block token figure is therefore *derived* — see
`DESIGN.md`.

**Project labels are a heuristic.** Claude Code names each transcript folder
after the working directory, collapsing `/`, `_` *and* `-` all into `-`. So
`/Users/me/Desktop/Agent_traces/agent-trace-analysis` becomes
`-Users-me-Desktop-Agent-traces-agent-trace-analysis`, and nothing in it says
which dashes were separators, which were underscores, and which were real
hyphens. The label is a best guess with the home prefix dropped; the raw slug is
in the tooltip.
