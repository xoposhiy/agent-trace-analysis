# TraceLens — how the bar is built

Local dashboard: reads Claude Code traces, draws each session as a vertical bar
of coloured blocks.

**Status:** session list and bar work. Problem detection not started.

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
```

## 1. Parse

Read the JSONL. Three things need care:

- **Streamed replies arrive in pieces** sharing a `message.id` — merge them, or
  one reply becomes a dozen fake blocks.
- **Tool results are on a separate line** from the call — attach each result to
  its call, so one tool call = one block.
- **Subagents are in separate files** (`<session>/subagents/agent-*.jsonl`).
  Load them, tag each event with its `agent_id`.

## 2. Classify

**Most tools classify by name alone — no LLM:**

| kind | tools |
|---|---|
| read | Read, Grep, Glob, WebFetch |
| write | Edit, Write, MultiEdit |
| coordination | TodoWrite, AskUserQuestion, plan mode |
| subagent | Agent, Task |
| chatting with user | a user message |

Anything unrecognised → coordination. Coordination is the catch-all.

**`Bash` is the exception, and it's 39% of all tool calls.** The name tells you
nothing — `cat foo.py` is a read, `pytest` is an execute, `cat > f <<'PY'` is a
write. So Bash commands go to the **LLM judge**, which reads the command text
and returns a kind.

The judge is batched (one request per session) and cached forever by
`hash(tool + input)`, so each distinct command is judged once ever. Without it,
a word-list guess gets Bash right only 55% of the time.

## 3. Merge

Two rules, both from the sketch:

- **Neighbouring blocks of the same kind join into one.**
  `read, read, read` → one read block.
- **Coordination between two same-kind blocks is absorbed.**
  `read, coordination, read` → one read block.
  But `read, coordination, write` stays three — that coordination is a real
  boundary.

Subagents don't merge into the timeline: each becomes one container block
holding its own inner blocks.

**Invariant:** no two adjacent blocks ever share a kind.

## 4. Draw

An SVG column. Each block's **height** is its share of the selected Y-axis
metric (tokens / time / messages), with a 3px floor so nothing disappears.
Switching the metric only changes heights — never order or colour.

- normal blocks — inset rectangles
- subagents — full-width container with a ring, inner blocks drawn inside
- chatting with user — thin full-width marker

Colours: green read, blue write, red execute, orange subagents, purple chat,
grey coordination.

---

## Notes worth keeping

- **Never sum `cache_read` into a token count.** It's the prompt re-read every
  message: one session showed 13.2M total, 12.5M of it cache reads. We show
  `input + output + cache_creation`.
- **The subagent tool is `Agent`, not `Task`** — keying on `Task` finds zero.
- **Claude Code writes its own session title** (`ai-title` line) — free, no LLM.
- **Everything degrades**: no key, no VPN, or a corrupt transcript, the page
  still renders.

## Layout

```
adapters/claude_code.py   1. parse
analysis/classify.py      2. classify (rules + judge)
analysis/blocks.py        3. merge
web/bar.js                4. draw
ir/models.py              Event / Block / Session
api/app.py                FastAPI
```

Tests: `pytest tests -q` — 152, offline, no VPN.

## Left to do

1. Problem detection, the severity filter, the Problems tab
2. The problem-specific pane
3. Money as a Y-axis metric (needs a price table)
4. Frontend tests for the bar's layout maths
5. Other vendors — the adapter boundary exists, only Claude Code is written
