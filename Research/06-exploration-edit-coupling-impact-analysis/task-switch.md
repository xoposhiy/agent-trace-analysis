# Detecting the Moment a Session Should Have Been Split

## 1. What the problem is

The idea for this row: find the exact **moment** in a long session where the
user should have started a **new** session instead of continuing in the same
one. If we can find that moment, we can measure how much old, now-useless
context got carried forward after it – and how much money that wasted.

My goal was to build a detector for this moment, test it on real sessions,
check if the result can be trusted, and then put a real dollar number on it.

## 2. What I did

I tried several signals, in order, and kept only the one that worked:

- **Word overlap** in what the user typed. A sudden drop in shared words
  could mean a new topic started.
- **File overlap.** If the agent suddenly works on completely different
  files, that could mean a new task.
- **Linear issue tracker.** Some sessions reference tickets in Linear (a
  project-management tool); a switch to a different ticket is a strong,
  human-defined signal of a new task.
- **Time gaps.** The real elapsed time between one user message and the next.

For each signal I ran it on all 769 long Claude Code sessions and checked
the result by hand before trusting it. For the one that worked (time gaps),
I then went further and calculated the real dollar saving using the exact
token counts recorded for each turn – the same method I used earlier for
the context-splitting cost analysis.

## 3. What I saw

**Word overlap and file overlap did not work.** Both flagged almost every
session (78–99%) as having a "switch," almost always right at the start.
Reading real examples showed why: a coding task naturally spans many files
and uses new words every turn (new file names, new errors) even when it is
still the *same* task. A big single task and several small tasks look the
same to these methods.

**The Linear issue tracker was a better idea, but too rare to use.** Only
**7 of 769 sessions (1%)** had enough Linear activity to judge – too small a
sample to trust.

**Time gaps worked.** A time gap does not depend on *what* the work is
about – it just measures whether the person stepped away. I tested five gap
sizes and checked two things for each: how many sessions have a gap that
big, and how much money splitting there would save (using real, per-turn
billing data, with a realistic cost added for the "catch-up" summary each
new session part would need):

| Gap size | Sessions with this gap | Where it happens in the session | Money saved (realistic) |
|---|---|---|---|
| 15 min | 76% | mostly early (not very meaningful yet) | $18,059 (50%) |
| 30 min | 64% | still somewhat early | $15,486 (43%) |
| 1 hour | 54% | more spread out | $13,056 (36%) |
| **4 hours** | **35%** | **spread almost evenly through the session** | **$8,556 (24%)** |
| 24 hours | 5% | mostly in the *second half* | $853 (2%) |

Two things stand out. First, the **position of the gap gets more spread out
as the threshold grows**: at 15 minutes, most gaps still cluster near the
start of the session – a sign this may just be normal early back-and-forth,
not a real break. At 4 hours, the gaps are spread almost evenly across the
whole session; at 24 hours they mostly land in the *second half*. This
matters because word overlap and file overlap always clustered at the start
(a sign of a false signal) – time gaps do not, and the effect gets stronger
as the gap grows, which is what a real signal should look like.

Second, **the money saved goes down as the threshold grows** – the opposite
direction from trust. This is not a contradiction: a short 15-minute
threshold triggers on almost every normal pause, so it "finds" more chances
to cut costs, even though most of those pauses are probably not real
breaks. A 4-hour threshold triggers less often, but each time it does, it is
far more likely to be a genuine one.

## 4. What I concluded

A real task switch cannot be reliably detected from *what* the user typed or
*which* files were touched – both signals are naturally noisy in coding
work, because one task can span many files and words. What works is *when*
the work happens.

**My key finding:** about **35% of long sessions have a 4-hour-or-longer
gap** between user messages, and splitting there would save an estimated
**$8,556 – about 24% of the total cost of these sessions.** This number
comes from real, per-turn billing data, not an estimate, and includes a
realistic cost for the "catch-up" summary each new session part would need.

Shorter gap sizes show bigger savings on paper (up to 50% at 15 minutes),
but their detections cluster suspiciously at the start of sessions – the
same warning sign that made the earlier word/file methods untrustworthy. So
the 4-hour number is smaller, but it is the one to trust – a bigger number
built on a shakier signal is not actually a better answer.

One honest limit: a long gap proves the person stepped away, not that the
*next* task is different. Someone doing short daily check-ins on one
feature would also show up as a "gap." So this measures **a good moment to
split**, not proof that the task itself changed.

## 5. What to put in the table cell

**About 35% of long sessions have a 4-hour-or-longer gap between user
messages. Splitting the session there would save an estimated $8,556
(about 24% of the total cost of these sessions), calculated from real
per-turn billing data.**

Method: tested four signals (word overlap, file overlap, Linear issues,
time gaps) on 769 long Claude Code sessions. The first three failed – they
could not tell apart "one big task using many files/words" from "a real new
task." Time gaps worked because they measure real elapsed time, not the
*content* of the work; their detections are spread through the session
instead of clustered at the start, which is the sign that the earlier
methods were false positives. The dollar figure uses the same real,
per-turn cost data as the earlier compaction-splitting analysis, plus a
realistic handoff cost for each split (the cost of generating and carrying
forward a short "catch-up" summary).

Limits: a time gap shows the person was away, not that the task changed –
read it as "a good point to split," not "proof this is a new task." Shorter
gap thresholds show larger savings but are less trustworthy (their
detections cluster near the session start); the 4-hour threshold is the
honest middle ground between coverage and reliability. The Linear-issue
signal, while rare (1% of sessions), is the most authoritative one when
present and could be combined with the time-gap signal in future work for
extra confidence.
