# SWE-chat Session Analysis: Context Coupling and Task-Switch Detection

*Project report – coding agent traces (SWE-chat)*

This report has two parts. Part 1 covers the context-coupling row
(exploration vs. editing vs. execution), including a correction made after
review. Part 2 covers a separate row — detecting the moment a
session should have been split into a new one — including an approach that
was tried, reviewed, and ultimately not adopted.

---

# Part 1 — The Context Coupling Problem: Exploration vs Editing vs Execution

## 1. What the problem is

A coding agent usually does everything in one context window: it reads
files, plans changes, and writes the edits all in the same place. The
SWE-Edit paper calls this the **context coupling problem**. The trouble is
that all the file-reading piles up in the expensive model's context and
clogs it. SWE-Edit's fix is to split the work by role: a cheap "Viewer"
model reads the files and keeps only the useful parts, and a separate
"Editor" writes the changes. On their benchmark this cut cost by 17.9% and
even improved accuracy by 2.1 points.

My goal for this row was to measure, on real sessions, how big this
coupling problem is, and to estimate how much money the SWE-Edit idea
could save.

## 2. What I did

- I selected long sessions (more than 30 turns) from Claude Code – the ones
  big enough to really have this problem.
- I counted how often the agent did each kind of action, using the tool it
  called (reading, editing, running commands).
- I measured the token size of each action's result – how much text each
  activity pushes into the context – because reading a file adds far more
  than making a small edit.
- **After review**, I found a gap in the first version: activity was
  sorted into a category using only the **tool's name**. A `Bash` call was
  always counted as "execution," even when the actual command was just
  `grep` or `cat` – which is reading, not running tests or builds. I fixed
  the categorizer so that, for every `Bash` call, it now looks at the real
  command text. If it starts with a read-only command (`cat`, `grep`,
  `find`, `ls`, `head`, `tail`, and similar), it is counted as
  **exploration**, not execution. Commands like `npm test` or `git commit`
  still count as execution.

## 3. What I saw

There are 769 long Claude Code sessions. I found three main activities, not
two – so I added a third category, "Execution" (running tests and
commands). The table below shows the numbers **before and after** the
Bash-command fix:

| Activity | Share of actions (before → after) | Share of context tokens (before → after) |
|---|---|---|
| Exploration (reading files) | 30% → **35%** | 55% → **57%** |
| Execution (running commands) | 38% → **33%** | 19% → **17%** |
| Editing (writing code) | 20% → 20% (no change) | 2% → 2% (no change) |
| Coordination (planning/tasks) | 11% → 11% (no change) | 24% → 24% (no change) |

The most important line is the first one. Exploration is only 30–35% of the
actions, but 55–57% of all the tokens in the context. Reading files is by
far the biggest thing filling the expensive model's window – which is
exactly what SWE-Edit moves to a cheaper model. Editing, on the other hand,
is only 2% of tokens, so it is cheap: the Editor part helps quality, not
cost.

**A surprise – Execution:** running commands (Bash) is the most frequent
action (33% after the fix) and still adds 17% of the tokens. The SWE-Edit
paper does not focus on this, but in real sessions command output (test
logs, errors) is a second thing that fills the context. This is my own
extra finding.

**About the fix:** roughly 9,500 calls moved from execution to exploration
– these were Bash calls that were really just reading (`grep`, `cat`,
`find`), not running tests or builds. The total number of calls stayed
exactly the same (188,549 before and after); nothing was lost, only
relabeled correctly. Editing and coordination not moving at all is a good
sign the fix only changed what it was supposed to change.

## 4. What I concluded

My measurements show clearly that the coupling problem is real, and that
reading is what drives it. Exploration takes up 55–57% of all the tokens in
the context – by far the largest source – while editing takes only 2%. So
the expensive model spends most of its context on file-reading, which is
exactly the work a cheaper Viewer model could take over. Because reading
dominates the context so strongly, that is where the real savings are;
splitting off editing barely changes cost, so its value is in reliability,
not money.

**My key finding: file-reading is 57% of the context in long sessions
(~$10,704 of cost) – and offloading it to a cheaper Viewer could save about
$6,422.**

**From size to savings:** I measured the size of the problem directly:
file-reading is 57% of the context, which works out to about $10,704 of
context cost on these long sessions. A Viewer model would not remove all of
it – SWE-Edit reports its Viewer filters out about 60% of the code it
reads. Applying that rate to my own figure gives an estimated saving of
about $6,422. The reading cost is measured here; only the 60% filter rate
comes from the paper.

The saving estimate moved up slightly from an earlier calculation
(~$6,150), because more of the cost is now correctly attributed to
exploration after the Bash fix. This is a small, expected shift – the
underlying problem (reading dominating the context) was already true
before the fix, not a sign anything else in the analysis was wrong.

## 5. What to put in the table cell

**File-reading (exploration) is 57% of all context tokens in long sessions
– the single biggest cost of the coupling problem. Offloading it to a
cheaper Viewer model could save an estimated $6,422.**

**Sessions affected:** 769 long Claude Code sessions. File-reading accounts
for about $10,704 of context cost – 57% of all tool-result cost, the
single largest source. Applying SWE-Edit's ~60% filter rate, roughly
$6,422 is removable by moving reads to a cheaper Viewer model.

**Categories:** three needed – Exploration, Execution, Editing (the
dataset's own 'action' count merges editing and execution). Tool calls are
sorted using both the tool name and, for Bash specifically, the actual
command text (so a plain `grep`/`cat` counts as exploration, not
execution). Editing is only 2% of tokens, so decoupling it improves
reliability, not cost.

**Limits:** the $10,704 reading cost is measured on this data; only the 60%
filter rate is borrowed from the SWE-Edit paper, which tested different
models, not Claude. Token sizes are approximated (characters ÷ 4) and reads
are assumed to stay in context, so $6,422 is an upper-ish estimate. The
Bash-command check catches the most common read-only commands but is not
exhaustive, so a small amount of exploration hidden inside other Bash
commands may still be miscounted as execution.

---

# Part 2 — Detecting the Moment a Session Should Have Been Split

> **Status note:** the approach below (time gaps) was reviewed: a long gap only shows the person
> stepped away – it does not prove the context became useless. In fact, if
> someone returns after 4+ hours and *keeps going in the same session*,
> that could mean the context is valuable enough to protect, which would
> make it a **bad** point to split, not a good one. This section is kept as
> a documented finding, not an adopted conclusion. The next attempt will
> instead follow original suggestion: split based on the
> **sequence of tool categories** from Part 1 (e.g. a long exploration
> phase followed by a clean shift to editing is a good split point; a
> session that constantly mixes exploration and execution is not
> splittable this way).

## 1. What the problem is

The idea for this row: find the exact **moment** in a long session where
the user should have started a **new** session instead of continuing in
the same one. If we can find that moment, we can measure how much old,
now-useless context got carried forward after it – and how much money that
wasted.

My goal was to build a detector for this moment, test it on real sessions,
check if the result can be trusted, and then put a real dollar number on
it.

## 2. What I did

I tried several signals, in order, and kept only the one whose *pattern*
looked trustworthy (see the status note above for what happened next):

- **Word overlap** in what the user typed. A sudden drop in shared words
  could mean a new topic started.
- **File overlap.** If the agent suddenly works on completely different
  files, that could mean a new task.
- **Linear issue tracker.** Some sessions reference tickets in Linear (a
  project-management tool); a switch to a different ticket is a strong,
  human-defined signal of a new task.
- **Time gaps.** The real elapsed time between one user message and the
  next.

For each signal I ran it on all 769 long Claude Code sessions and checked
the result by hand before trusting it. For the one whose pattern held up
(time gaps), I went further and calculated the real dollar saving using the
exact token counts recorded for each turn – the same method used for the
context-splitting cost analysis.

## 3. What I saw

**Word overlap and file overlap did not work.** Both flagged almost every
session (78–99%) as having a "switch," almost always right at the start.
Reading real examples showed why: a coding task naturally spans many files
and uses new words every turn (new file names, new errors) even when it is
still the *same* task. A big single task and several small tasks look the
same to these methods.

**The Linear issue tracker was a better idea, but too rare to use.** Only
**7 of 769 sessions (1%)** had enough Linear activity to judge – too small
a sample to trust.

**Time gaps** showed the most internally-consistent pattern of the four
signals tried. A time gap does not depend on *what* the work is about – it
just measures whether the person stepped away. Five gap sizes were tested,
checking coverage, where the gap falls in the session, and the dollar
saving from splitting there (using real, per-turn billing data, with a
realistic cost added for the "catch-up" summary each new session part would
need):

| Gap size | Sessions with this gap | Where it happens in the session | Money saved (realistic) |
|---|---|---|---|
| 15 min | 76% | mostly early (not very meaningful yet) | $18,059 (50%) |
| 30 min | 64% | still somewhat early | $15,486 (43%) |
| 1 hour | 54% | more spread out | $13,056 (36%) |
| 4 hours | 35% | spread almost evenly through the session | $8,556 (24%) |
| 24 hours | 5% | mostly in the *second half* | $853 (2%) |

Two things stood out. First, the **position of the gap got more spread out
as the threshold grew**: at 15 minutes, most gaps clustered near the start
of the session – a sign this may just be normal early back-and-forth, not
a real break. At 4 hours, the gaps were spread almost evenly across the
whole session; at 24 hours they mostly landed in the *second half*. This
mattered because word overlap and file overlap always clustered at the
start (a sign of a false signal) – time gaps did not.

Second, **the money saved went down as the threshold grew** – the opposite
direction from the apparent trustworthiness of the signal. A short
15-minute threshold triggers on almost every normal pause, so it "finds"
more chances to cut costs, even though most of those pauses are probably
not real breaks. A 4-hour threshold triggers less often, but each detection
looked more distinct from ordinary back-and-forth.

## 4. What I concluded (at the time)

A real task switch could not be reliably detected from *what* the user
typed or *which* files were touched – both signals are naturally noisy in
coding work, because one task can span many files and words.

The time-gap signal's *pattern* (spread across the session rather than
clustered at the start) looked more trustworthy than the content-based
signals. About 35% of long sessions have a 4-hour-or-longer gap between
user messages, and splitting there was estimated to save **$8,556 – about
24%** of the total cost of these sessions, using real per-turn billing
data.

**However**, as the status note above explains, a spread-out pattern only
shows the method isn't making the *same kind* of mistake as the earlier
methods (clustering near the start) – it does not prove the underlying
idea is correct. One objection stands: stepping away for a long
time does not mean the context is now useless. It could mean the opposite.
So this "conclusion" should be read as **what the data pattern suggested**,
not as a validated recommendation.

## 5. What was in the table cell (superseded)

*(Kept for the record; see the status note above. Not the current
recommendation for this row.)*

About 35% of long sessions have a 4-hour-or-longer gap between user
messages. Splitting the session there was estimated to save $8,556 (about
24% of the total cost of these sessions), calculated from real per-turn
billing data.

Method: tested four signals (word overlap, file overlap, Linear issues,
time gaps) on 769 long Claude Code sessions. The first three failed – they
could not tell apart "one big task using many files/words" from "a real
new task." Time gaps produced a more evenly-spread pattern, but
review identified that the underlying logic (long gap = good split point)
does not hold up, since a long gap can equally mean the context is worth
protecting. The Linear-issue signal, while rare (1% of sessions), is the
most authoritative one when present. The next attempt will use the tool
categories from Part 1 (exploration/editing/execution sequence) to find
split points instead of time.
