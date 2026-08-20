# TraceLens — design

Local dashboard: reads Claude Code traces, draws each session as a vertical bar
of coloured blocks, and drills down to what each block actually did.

**Status:** session list, bar, block pages and subagent drill-down work.
Problem detection not started.

---

## The pipeline

```
  ~/.claude/projects/*.jsonl
          |
    1. PARSE      ->  Events   (one per user message / tool call / reply)
          |
    2. CLASSIFY   ->  each event gets a kind: read, write, execute,
          |            coordination, subagent, chatting-with-user
    3. MERGE      ->  Blocks   (neighbouring same-kind events join)
          |
    4. DRAW       ->  the vertical bar
          |
    5. DRILL      ->  block page -> subagent -> that agent's own bar
```

## 0. The transcript format

*What the app reads.*

A session file is flat and append-only; the conversation tree lives in
`parentUuid`. One line carries one content block — streaming writes `thinking`,
`text` and each `tool_use` to separate lines sharing one `message.id` and
repeating one identical `usage`.

Cost lives only on assistant lines, at `message.usage`, in four buckets:
`input`, `output`, `cache_creation`, `cache_read`. No dollar figure is recorded
anywhere.

Four identity levels, each addressing something different:

| level | key | what it is |
|---|---|---|
| session | `sessionId` | one transcript file |
| turn | human `user` line | one exchange |
| API call | `message.id` | **the unit cost is reported for** |
| line | `uuid` | one content block |

Most lines are not Events. Attachments, tool-result carriers, thinking and
system lines all carry a `uuid` and none becomes a block.

## 1. Parse

*What the IR is built from.*

`adapters/claude_code.py` turns raw lines into `Event`s: streamed fragments
sharing a `message.id` merge into one reply, tool results inline into the call
that produced them, and each subagent file
(`<session>/subagents/agent-*.jsonl`) is loaded with every event tagged by
`agent_id`.

A user-role line inside a subagent is the task its parent handed it, never a
human turn, so `is_human_prompt` is false throughout a subagent.

## 2. Classify

*What decides a block's colour.*

Most tools classify by name alone — no LLM:

| kind | tools |
|---|---|
| read | Read, Grep, Glob, WebFetch |
| write | Edit, Write, MultiEdit |
| coordination | TodoWrite, AskUserQuestion, plan mode |
| subagent | Agent, Task |
| chatting with user | a *human* user message |

Anything unrecognised is coordination, the catch-all.

`Bash` is the exception and 39% of all tool calls: `cat foo.py` is a read,
`pytest` an execute, `cat > f <<'PY'` a write. Bash commands go to an LLM judge
that reads the command text. Batched one request per session and cached forever
by `hash(tool + input)`, so a command is judged once ever.

## 3. Merge

*What turns 3,000 events into a readable bar.*

- Neighbouring blocks of the same kind join into one.
- Coordination between two same-kind blocks is absorbed;
  `read, coordination, write` stays three, since that coordination is a real
  boundary.
- A small coordination run ahead of work is its preamble and joins it
  (`PREAMBLE_MAX_RATIO`).
- Adjacent subagent containers become one band. `Block.agents` holds the
  individual agents; `Block.inner_blocks` stays flat for the bar.

**Invariant:** no two adjacent blocks share a kind.

### Block labels

`kind · subject · N steps · M failed`, capped at 80 characters with the subject
capped separately at 44. The subject is the first of these available
(`analysis/labels.py`):

1. **Files touched, repetition counted** — `write · bar.js ×3`. Widely
   scattered work is counted instead: `read · 6 files in 4 dirs`.
2. **Bash's own `description` field** — `execute · Run tests and re-audit +3`.
3. **Tool mix**, then prose for pure-coordination blocks.

## 4. Draw

*What the bar shows.*

An SVG column. Block **height** is its share of the selected Y-axis metric
(tokens / time / messages) above a 3px floor; switching the metric changes only
heights, never order or colour.

- normal blocks — inset rectangles
- subagents — solid, full width; width carries "delegated"
- chatting with user — thin full-width marker

The bar sizes itself to leave real proportional room and the column scrolls; it
is not capped to a screen. Unpainted space always shows a filled track, never
the page. A test renders every local session across all three metrics and
asserts nothing is left unpainted beyond a 3px hairline.

Colours are one hue per kind, validated for colour-blind separation on a dark
surface. Chatting-with-user is a rule rather than a fill, so hue is never its
only cue.

## 5. Drill down

*What clicking gives you.*

```
/session/<id>                              the session bar
/session/<id>/block/<n>                    one block's steps
/session/<id>/agent/<agentId>              that subagent's own bar
/session/<id>/agent/<agentId>/block/<n>    one block of that bar
```

A block page lists each step: the file read, the command run with its full
text, the result, whether it failed, and the API call it was billed to. A
subagent band lists the agents under it, each linking to its own bar — same
colours, same Y-axis selector, its own clickable blocks. A band's own step list
shows the spawning calls only.

Blocks are derived, not stored, so URLs are positional and every page states
"block 12 of 105". Out-of-range indexes 404 with the real count. Step detail is
fetched per block rather than shipped with the session, and long arguments and
outputs are clipped with their full length reported.

## How tokens are reported

The bar and the session header show **every billed token, cache reads included**.
Each block also carries its **own cache-read figure**, which the hover readout
states next to the total with its share of it — `5.3M tokens · 5.3M re-read (99%)`.

Cache reads are attributed by **residency**. Every reply re-sends the whole
conversation, so each reply's cache-read cost is divided by size among whatever is
still in the conversation at that moment, and a block's figure is its share summed
over every reply it was present for — roughly its own size times the number of
replies that re-sent it. A compaction ends residency. The system prompt and tool
definitions are in every reply but belong to no block, so they are shared across
the session instead.

Details and the reasoning behind each choice are in `analysis/attribution.py`.

## Provenance

Cited from `adapters/claude_code.py` and `ir/models.py`.

Four pieces of line-handling follow Entire CLI's compact transcript package
(`cli/cmd/entire/cli/transcript/compact/compact.go`, MIT, © Entire Inc.),
reimplemented in Python, each carrying an attribution comment naming its source
line:

| taken | source |
|---|---|
| merge streamed assistant fragments sharing `message.id` | `compact.go:432` |
| inline tool results into their `tool_use` block | `compact.go:453` |
| drop `thinking` / `redacted_thinking` blocks | `compact.go:685` |
| enrich results from the `toolUseResult` envelope | `compact.go:552` |

The IR schema starts from the same package's compact format (`compact.go:27`).

Not taken: Entire's IR has no representation for subagents as separate
transcripts, for `agent_id` as a parent→child link, or for cache tokens, and it
compacts for a model to re-read rather than for a person to look at. Blocks,
kinds, merge rules, attribution, labels and everything drawn are ours.

## Layout

```
adapters/claude_code.py   1. parse
analysis/classify.py      2. classify (rules + judge)
analysis/blocks.py        3. merge + subagent banding
analysis/labels.py           what a block calls itself
web/bar.js                4. draw
analysis/steps.py         5. per-step detail
web/block.js  web/agent.js   the drill-down pages
ir/models.py              Event / Block / Session
api/app.py                FastAPI
```

Tests: `pytest tests -q` — 308, offline, no VPN. Plus 22 Node tests for the
bar's layout maths (`tests/bar.test.js`, also run by pytest).

## Left to do

1. Problem detection, the severity filter, the Problems tab
2. The problem-specific pane — a second bar beside the main one, anchored to
   event `uuid` ranges rather than block indexes or timestamps
3. A richer hover readout
4. Other vendors — the adapter boundary exists, only Claude Code is written
