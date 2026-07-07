# Where Long Coding-Agent Sessions Spend Money, and How Splitting Could Save It

*Project report — analysis of real coding-agent sessions (SWE-chat dataset)*

## 1. Background: what this is about

When a coding agent (like Claude Code) works on a task, it does everything
inside one **context window**: it reads files, runs commands, and writes
code, all in the same place. Every time the agent takes a new step, it
re-reads everything already in that context — and re-reading cached text
still costs money. So the longer a session runs, the more the early
material gets re-read again and again, and the more it costs.

This report studies **769 long sessions** (more than 30 turns each) from
the SWE-chat dataset, all from the Claude Code agent. The goal was to
answer three questions:

1. What kind of work fills the context, and what does it cost?
2. Do users mix unrelated tasks in one long session (so it could be split)?
3. Do sessions start with a long "reading" phase that a cheaper model could
   have handled instead?

For each question I also estimated the money that a fix could save.

## 2. What kind of work fills the context

I sorted every tool call the agent made into four kinds of activity:
**exploration** (reading files, searching), **editing** (writing code),
**execution** (running commands and tests), and **coordination** (planning
and task management). One detail matters here: a command like `grep` or
`cat` run through the shell is really *reading*, so I check the actual
command text and count those as exploration, not execution.

Across all 769 sessions there were 188,549 tool calls. Counting how many
**tokens** each activity pushes into the context gives the key result:

| Activity | Share of calls | Share of context tokens |
|---|---|---|
| Exploration (reading files) | 35% | **57%** |
| Execution (running commands) | 33% | 17% |
| Editing (writing code) | 20% | 2% |
| Coordination (planning) | 11% | 24% |

The headline: **exploration (file-reading) is 57% of all context tokens** —
by far the biggest cost. Editing is only 2%, so writing code is cheap; the
expensive part is all the reading. In dollar terms, that reading costs
about **$10,704** across these sessions.

One idea from research (the "SWE-Edit" paper) is to give the reading to a
cheaper "Viewer" model, keeping only the useful parts for the expensive
model. That paper found a Viewer removes about 60% of what it reads.
Applying that rate to my measured figure gives an estimated saving of about
**$6,422**. (The reading cost is measured on this data; only the 60% rate
is borrowed from the paper.)

## 3. Do users mix unrelated tasks in one session?

If a user finishes one task and then starts a completely different one in
the same session, the old context becomes useless — a good place to start
a fresh session ("split" it). The question is how often this actually
happens.

Earlier attempts to detect this by counting shared words or shared files
failed: a single big task naturally touches many files and uses new words
each turn, so those methods flagged almost everything. Instead I used an
**LLM as a judge** — I gave a language model the user's messages from each
session, in order, and asked it whether the user truly switches to an
unrelated task (while telling it that moving from design to planning to
testing *within the same feature* does **not** count as a switch).

Result across 765 judged sessions: only **56 (7%)** contained a genuine
task switch. And they happen **late** — 44 of the 56 fall in the second
half of the session, most in the final quarter.

The meaning is important: **long sessions are long because the task itself
is big, not because users cram unrelated tasks together.** When a switch
does happen, it is usually a small extra request tacked on near the end.

## 4. Do sessions start with a long reading phase?

Even within a single task, a session might begin with a long stretch of
pure reading before any code is written. That opening reading phase could
have been done in "**plan mode**" — a cheaper model reads and plans first,
then hands a focused summary to the expensive model for the actual editing.

To find this, I walked through each session's tool calls **in order** and
grouped them into phases. I then classified each session:

| Pattern | Share | Meaning |
|---|---|---|
| **A** — reading first, then editing | **73%** | plan-mode opportunity |
| C — reading burst in the middle | 12% | sub-agent opportunity |
| B — reading and editing mixed | 1% | cannot be cleanly split |
| none — no clear reading phase | 13% | — |

So **73% of long sessions begin with a clear reading phase** (about 11
calls on average) before shifting to editing. This is the "missed plan-mode
opportunity": that opening reading could have been offloaded to a cheaper
model. A separate 12% have a reading burst in the *middle* of the session,
which is a candidate for handing that burst to a sub-agent instead.

## 5. How much could splitting actually save?

To turn these findings into money, I used a math model that prices what
happens when a session is "split" at a chosen point: after the split, the
new part no longer re-reads all the old context (so the repeated cache-read
cost drops), but it pays a small one-time cost to carry a summary forward.
I fed the model each session's **real** token counts and split it at the
two points found above.

**Splitting at a task switch (from Section 3):** of the 56 sessions with a
switch, 43 had a switch early enough to be worth splitting. On those, the
cost dropped from **$1,551.61 to $1,296.45 — a saving of $255.16 (16%)**,
removing about 639 million re-read tokens. This is a real saving, but it
applies to few sessions, because switches are rare.

**Splitting at the end of the opening reading phase (plan mode, from
Section 4):** this applies to far more sessions — **366** of them. The cost
dropped from **$11,756.28 to $10,441.37 — a saving of $1,314.91 (11%)**,
removing about 3.1 billion re-read tokens. This is the bigger opportunity,
because most sessions have this pattern.

One honest note: the opening reading phase ends early, so the plan-mode
split happens near the start of the session. The 11% is what the math model
computes for a split at that point — a genuine, if modest, per-session
saving that adds up because it applies to so many sessions.

## 6. Conclusion

The three findings fit together into one clear story:

- **Reading dominates cost.** Exploration is 57% of all context tokens
  (~$10,704); editing is trivial (2%).
- **Sessions are long because tasks are big, not messy.** Only 7% contain a
  genuine task switch, and those happen late.
- **The real opportunity is plan mode.** 73% of sessions open with a reading
  phase that a cheaper model could have handled.

For potential savings, there are two separate levers on the same underlying
cost (reading), and they should **not** be added together:

- Handing reading to a cheaper Viewer model: ~**$6,422** estimated.
- Splitting at the opening reading phase (plan mode): **$1,315 (11%)**
  measured with the math model, across 366 sessions.
- Splitting at genuine task switches is minor: **$255 (16%)** across just
  43 sessions.

All dollar figures are token-based estimates. Where possible the underlying
counts (tokens, calls, switch positions) are measured directly on the 769
real sessions; only the Viewer's 60% filter rate is borrowed from outside
research.
