# Saving Money on Long Coding-Agent Sessions

*Project report — analysis of post-session coding agent traces (SWE-chat dataset)*

---

## 1. What I did

I worked with SWE-chat, a public dataset from Stanford with about 6,000 real coding-agent sessions from developers using tools like Claude Code. My goal was to estimate how much money could be saved by splitting up long sessions, and to fill the empty table cell "upper bound of saved money".

- I focused on long sessions (more than 30 turns), because these are where the context grows large and costs go up.
- I built a cost model using the real token counts in the data and Claude's official prices (Opus: $5 per million input tokens, $25 output; cache reads are 90% cheaper).
- I broke the cost into four token types (fresh input, output, cache writes, cache reads) to see which part is actually expensive.
- I opened the raw turn-by-turn transcripts and ran a split simulation: every time the context passed a limit, I "split" the session into a new part that does not carry the old history, and recomputed the cost.

## 2. What I saw

Long sessions are a small part of all sessions, but they cost a very large part of the money:

| Finding | Result |
|---|---|
| Total spend (all sessions) | $51,204 |
| Share of sessions that are long (>30 turns) | 14% |
| Share of total money spent on long sessions | 49% ($25,076) |
| Cache reads (re-reading old context) | 63% of long-session cost |
| Output tokens (cannot be reduced by splitting) | 6% of long-session cost |

The most important thing I saw: most of the cost (63%) is cache reads — the old conversation history being re-read on every turn. But cache reads are already discounted by 90%, so the "expensive-looking" big context is actually the cheap part.

I then simulated splitting the long sessions. This was measured directly on all 761 long Claude Code sessions (not a sample, not scaled), so the savings below come straight from the data:

| Splitting assumption | Saving |
|---|---|
| Free handoff (perfect, impossible — the ceiling) | 53% (~$18,600) |
| Real handoff (with a summary cost per split) | 39% (~$13,600) |

**Free vs. real handoff:** when you split a session, the new part does not know what happened before. In real life you must give it a short summary, and that summary costs tokens. "Free handoff" pretends the summary is free (best case). "Real handoff" charges a realistic cost. The gap between 53% and 39% is the price of these summaries.

## 3. What I concluded

The simple guess — "long sessions waste a lot of money on big context, so splitting saves a fortune" — is only partly right. Most of the cost is cache reads, which are already discounted, so splitting can never remove all of it. But when measured on all 761 long sessions, splitting still saves a large share, because the biggest, most expensive sessions are exactly the ones that gain the most. A small sample misses these, which is why my earlier 300-session test showed a much lower number.

**My measured result: about 39% saving realistically (~$13,600), with an upper bound (best case) of about 53% (~$18,600).**

**How these numbers were measured:** the simulation ran on all 761 long Claude Code sessions and read the real per-turn token counts — the dollar savings (~$13,600 and ~$18,600) are summed straight from the data, not scaled from a sample. The baseline cost of these sessions, measured from the transcripts, is $35,030. Because the percentage saving (39% / 53%) does not depend on which cost total you use, the percentage is the safest figure to quote.

## 4. What to put in the table cell

For the empty cell "upper bound of saved money", I suggest writing a range with a short reason, not a single number:

> **Upper bound ≈ 53%** of long-session cost (~$18,600 measured, best case, free/lossless splits).
> **Realistic estimate ≈ 39%** (~$13,600, with handoff summary cost).

**Method:** simulated context-limit splits measured directly over all 761 long Claude Code transcripts (no extrapolation). The saving is large because it concentrates in the biggest sessions; only cache reads are reducible (input, output and cache writes are unchanged). Quote the percentage as the primary figure — the dollar amount depends on the cost baseline ($35,030 measured from transcripts).
