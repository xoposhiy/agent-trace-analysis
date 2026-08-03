# TraceLens — working agreement

Local dashboard for Claude Code session traces. Python + FastAPI backend,
vanilla JS/SVG frontend, no build step.

These rules are not style preferences — they are the contract for changes to
this project. Follow them exactly.

---

## 1. Every file opens with a docstring

The first thing in every `.py` file is a module docstring; every `.js` file
opens with a comment block. It answers **what this file is for** and **how it
fits** — not a restatement of the filename.

```python
"""Claude Code adapter: raw ``~/.claude/projects/**/*.jsonl`` -> TraceLens IR.

Layout on disk (verified against real transcripts, 2026-08-03):

    ~/.claude/projects/
      <project-slug>/
        <session-id>.jsonl                       <- main transcript
        <session-id>/subagents/agent-<id>.jsonl  <- one file per subagent
"""
```

Include anything a reader cannot get from the code: on-disk layouts, wire
formats, ordering guarantees, where the logic came from. Bad opening: `"""Parser
for Claude Code."""` — that is the filename with more words.

## 2. Segmentation comments divide every file

Files are split into labelled sections with a banner. No file is one
undifferentiated wall.

```python
# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
```

```javascript
// --- session list -----------------------------------------------------
```

Sections group by responsibility (`Discovery`, `Line-level helpers`,
`Session assembly`), not by language construct — never a `# Constants` section
holding unrelated constants. If a section grows past ~150 lines, it is usually
a new module.

## 3. Names say what a thing is

- No single letters outside a tight comprehension. `for e in events` is fine;
  `e = load()` is not.
- No abbreviations that are not already domain vocabulary. `session`, not
  `sess`. `response`, not `resp`. `transcript`, not `tr`. `agent_id` and
  `uuid` are fine — that is what the format calls them.
- Booleans read as assertions: `is_subagent`, `has_summary`, `llm_available`.
- Functions are verbs: `discover_sessions()`, `load_session()`. Values are
  nouns: `pending_summaries`, `projects_root`.
- A name that needs a comment to explain it is the wrong name. Fix the name.
- Private module helpers take a leading underscore: `_parse_ts`, `_read_jsonl`.

## 4. Comments explain *why*, never *what*

The code already says what it does. A comment earns its place by recording a
reason, a constraint, or a trap — something that would otherwise be rediscovered
the hard way.

```python
# GOOD — records a trap and the evidence for it.
# ``cache_read`` is the whole prompt prefix re-read on *every* message, so
# summing it counts the same context hundreds of times: in a real 227-message
# session here, total was 13.2M of which 12.5M (95%) was cache reads.

# BAD — restates the line below it.
# Loop over the sessions
for session in sessions:
```

When a comment states a fact about the data (a field name, a count, a format
quirk), it must have been **verified against a real transcript**, and say so.
Never write a confident claim you have not checked.

## 5. Failures degrade, they do not crash

This tool reads files someone else is actively writing, and calls an LLM behind
a VPN. Both fail routinely. One bad transcript must never empty the dashboard;
an unreachable LLM must never block a page.

Catch narrowly, degrade visibly, and say why in a comment. When the UI is
missing something, it explains what and why — never a silent blank.

---

## 6. Tests — required for every change

**Every update ships with tests. A change without tests is not finished.**
This is not negotiable and not deferred to "later".

### What to test

| Change | Required tests |
|---|---|
| Parsing / IR | A fixture exercising it, plus the degenerate case (empty, malformed, missing field) |
| A bug fix | A test that **fails before the fix** and passes after. Write it first. |
| API endpoint | Status code, response shape, and the filtered/empty variants |
| Caching | That a hit avoids the work, and that invalidation actually triggers |
| Frontend logic | Pure helpers (`formatDuration`, `relativeTime`) get unit tests |

### How to run

```bash
pytest tests -q            # all
pytest tests -q -k adapter # one area
```

### Rules

- **Fixtures, not the developer's home directory.** Tests build transcripts in
  `tmp_path` and point `CLAUDE_CONFIG_DIR` at it. A test that reads the real
  `~/.claude` is broken — it passes or fails based on whose machine it runs on.
- **Never call the LLM.** Judge tests stub the client or hit the cache only. A
  test suite that needs VPN is not a test suite.
- **Name the behaviour, not the function**:
  `test_subagent_events_are_linked_to_parent_session`, not `test_load_session`.
- **One assertion subject per test.** Several `assert`s about the same claim is
  fine; testing three unrelated things in one function is not.
- **Assert on real numbers.** If a fixture has 3 tool calls, assert `== 3`, not
  `> 0`.
- **Every claim in a comment or docstring about the data format should have a
  test pinning it.** That is how format drift gets caught — `Task` -> `Agent`
  would have been caught instantly by a test.

### Before saying a change is done

```bash
pytest tests -q
```

Report the actual result. If tests fail, say so and show the output. Never
describe work as complete or verified when it was not run.

---

## 7. Project-specific traps

Learned the hard way; do not rediscover them.

- **The subagent tool is `Agent`, not `Task`.** Claude Code renamed it; keying
  on `Task` alone finds zero subagents in current transcripts. Accept both.
- **Subagent transcripts are separate files**
  (`<session-id>/subagents/agent-<id>.jsonl`), never inline. `isSidechain: true`
  does not appear in main transcripts at all.
- **`toolUseResult.agentId`** is the direct parent -> subagent link. No scanning.
- **Never sum `cache_read` into a user-facing token count.** See §4.
- **Streamed assistant fragments share `message.id`** and must be merged, or one
  turn becomes a dozen phantom blocks.
- **Project slugs are lossy** — Claude Code maps both `/` and `-` to `-`, so the
  original path is unrecoverable. Labels are a heuristic; keep the raw slug in a
  tooltip and never present the label as authoritative.
- **A blank `.env` value must never override a real one.** Merge with
  `dotenv_values`, not `load_dotenv`.

## 8. Provenance

Four pieces of transcript handling follow the Entire CLI's compact package
(`cli/cmd/entire/cli/transcript/compact/`, MIT, (c) Entire Inc.). Where code
follows it closely, carry an attribution comment naming the source file. See
`DESIGN.md` §2 for what was taken and what deliberately was not.
