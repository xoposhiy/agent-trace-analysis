# Splitting Sessions into Chunks — Cost Model & Results

**Scope:** Claude Code sessions only (`agent ∈ {"Claude Code", "claude-code"}`) — 4,853 sessions
from the `sessions` table of `SALT-NLP/SWE-chat`.
We use `sessions` (not `conversations`) because it is the trustworthy source for absolute
token/cost totals — `conversations` undercounts API calls ~9×.

Script: [`context_rot_split_cost.py`](context_rot_split_cost.py)

---

## 1. The question

Every API call in an agent session re-sends the whole conversation history. With prompt
caching that re-read shows up as **cache-read tokens**, and it grows with session length —
this is the **quadratic rebilling** (the "context tax"). We ask:

> If we split a session into independent chunks at a context threshold `T`, billing each
> chunk **as if it were a fresh session** (so the rebilling of earlier chunks' history is
> removed), how much would it cost — including a per-chunk overhead for re-establishing
> context — and how does that compare to billing the session as-is?

---

## 2. Inputs (per session, from `sessions`)

| symbol | column | meaning |
|---|---|---|
| `IT` | `input_tokens` | uncached input |
| `OT` | `output_tokens` | generated output |
| `CW` | `cache_creation_tokens` | cache writes (new unique content) |
| `CR` | `cache_read_tokens` | cache reads (re-read history) — **the quadratic term** |
| `N`  | `api_call_count` | number of API calls |

**Rates** (per MTok, blended — the `sessions` table has no `model` column; the Claude Code
mix is mostly Opus 4.x):

```
r_in  = $4.75      cache-write = 1.25 × r_in
r_out = $23.75     cache-read  = 0.10 × r_in
```

---

## 3. The math model

### 3.1 Cost "as they are" (full, with quadratic rebilling)

```
C_full = r_in·IT + r_out·OT + 1.25·r_in·CW + 0.10·r_in·CR
```

`CR` already contains the full quadratic tax, so this reproduces the recorded billing.

### 3.2 Where the quadratic lives — and what `P` is

Per call *i*, the context that gets re-read, `ctx_i`, **grows** as the conversation
accumulates:

- **Call 1:** small context (system prompt + first message).
- **Call 2:** re-reads call 1 + the new turn → bigger.
- **Call N:** re-reads the *entire* accumulated history → biggest.

Model that growth as a **linear ramp from ~0 up to a peak `P`** (a triangle):

```
ctx
 P |                          •      ← peak context, last call
   |                    •
   |              •
   |        •
   |  •
 0 +------------------------------ call number
   1                          N
```

`P` = the **peak context size** (tokens in the window on the biggest/last call).

Cache-read total `CR` is just the **sum of all those per-call re-reads = the area under the
ramp**. The area of a triangle with height `P` and base `N` is `½·N·P`, so:

```
CR = Σ_i ctx_i  ≈  N · (P/2)      ⇒      P = 2·CR / N
```

`P` is **reverse-engineered per-session** from its own recorded `CR` and `N` — derived so the
triangle's area matches the real cache-read total (self-calibrating, no fabricated numbers).

*Sanity check (all Claude Code sessions):* avg context/call = `CR/N` ≈ 132k tokens, so peak
`P ≈ 264k` — consistent with long agent sessions pushing past the 200k window via compaction.

### 3.3 Cost when split at threshold `T` (each chunk = a fresh session)

**The mapping: one big triangle → many small saw-teeth.**

Splitting resets the context ramp to ~0 every time it reaches `T`, keeping the **same growth
slope**. Step by step:

```
slope          s = P / N                tokens of context added per call
calls/chunk    m = T / s = T·N / P      calls to climb 0 → T at slope s
num chunks     k = N / m = P / T        → ⌈ P / T ⌉  (rounded up)
one saw-tooth  area = m · T/2 = N·T²/(2P)
all k chunks   CR_split = k · N·T²/(2P) = (P/T)·N·T²/(2P) = N · T / 2
```

The `P` **cancels**: the big triangle of area `N·P/2` becomes `k` saw-teeth whose total area
is just `N·T/2` — height capped at `T` instead of `P`.

```
   FULL (one triangle, area N·P/2)        SPLIT at T (saw-teeth, area N·T/2)
 P |              •                      T |  /| /| /|
   |          •                            | / |/ |/ |
   |      •                                |/  |  |  |
 0 +----------------                     0 +----------------
   1            N                          reset reset reset
```

Geometrically: slice the tall triangle at height `T` and slide every slice above `T` back
down to the baseline — same slope, same total calls `N`, but no slice ever sits higher
than `T`. The result:

```
k        = ⌈ P / T ⌉                  number of chunks
CR_split = min( CR , N · T / 2 )       rebilling: quadratic → LINEAR
```

The rebilling drops from `N·P/2` (grows with session length) to `N·T/2` (capped by the
threshold). Reduction factor = `T/P`. If a session never reaches the threshold (`P < T`)
then `k = 1` and nothing changes.

#### Important: the previous chunk is subtracted PER CALL, not once

A previous chunk is **not** re-read a single time — it sits inside the context of **every
call** in every later chunk, so it must be subtracted once *per call*. A naive "remove the
prior chunk once" would undercount the savings. The `CR_split = N·T/2` (area) formula already
handles this, because subtracting areas removes the prior-chunk weight at every vertical slice
(every call). Worked example:

`N = 4` calls, slope `s = 50k`/call, peak `P = 200k`, threshold `T = 100k` → `k = 2` chunks.

```
       FULL ctx (re-reads everything)        SPLIT ctx (chunk 2 resets)
 call  chunk                                   subtracted
   1     1     50k                  50k            0
   2     1    100k                 100k            0
   3     2    150k = 100k(ch1)+50k   50k         100k   ← chunk 1 re-read in call 3
   4     2    200k = 100k(ch1)+100k 100k         100k   ← chunk 1 re-read AGAIN in call 4
              --------------------  ------       -----
   CR_full = 500k            CR_split = 300k    subtracted = 200k
```

Chunk 1 (100k) is subtracted **twice** — once for each call in chunk 2 — because it appears in
both calls' context. Not 100k removed, but 200k. The formula agrees:

```
subtracted = CR_full − CR_split = 500k − 300k = 200k
           = N·T·(k−1)/2 = 4·100k·(2−1)/2 = 200k   ✓
```

General form: each call in chunk `c` drops `(c−1)·T`; summed over all calls this is
`N·T·(k−1)/2 = N(P−T)/2 = CR − N·T/2`. A one-time-per-chunk subtraction would have removed
only 100k here and **undercounted the savings by half**.

### 3.4 Per-chunk overhead (re-establishing context)

Each **new** chunk must re-load carried-over state (summary, key files, system prompt).
This is a flat cost — a fraction `α` of the threshold, priced at the **global average $/token**:

```
O_cost = α · T · c̄          α ∈ {5%, 10%, 15%, 20%}   (swept)
c̄      = (total C_full across all sessions) / (total tokens across all sessions)
```

Applied to `k − 1` chunks (the first chunk is the original session start and pays nothing).
Because overhead factors out of `α` linearly, the total for any `α` is
`split_no_overhead + α · T · c̄ · Σ(k−1)` — so we sweep several `α` values cheaply.

### 3.5 Split total and savings

```
C_split = r_in·IT + r_out·OT + 1.25·r_in·CW + 0.10·r_in·CR_split + (k−1)·O_cost

savings = 0.10·r_in·(CR − CR_split)     ← rebilling saved
          − (k−1)·O_cost                ← overhead paid
```

Everything is computed straight from the `sessions` aggregates — no per-call data needed.
Dollar figures are token-based **estimates** at blended Anthropic rates.

### 3.6 Per-chunk cost model (each chunk billed as a fresh mini-session)

The section 3.5 total is the sum of `k` independent chunk costs. A session splits into
`k = ⌈P/T⌉` chunks; each is a mini-session with `m = N/k` calls whose context ramps `0 → T`
and then resets. Because every chunk resets to 0 and climbs to the same height `T`, **all
chunks have the same rebilling** (the chunk index `j` doesn't change the cache-read).

Cost of one chunk `j` (`j = 1 … k`):

```
C_chunk(j) =  r_in · (IT/k)              its share of uncached input
            + r_out · (OT/k)             its share of output
            + 1.25 · r_in · (CW/k)       its share of cache-writes (unique content)
            + 0.10 · r_in · CR_chunk     its OWN rebilling (re-reads only itself)
            + O_cost · [j ≥ 2]           overhead, only for chunks 2 … k

  CR_chunk = m · T / 2 = N·T / (2k)      area of this chunk's little triangle
  O_cost   = α · T · c̄                   re-establish cost (first chunk pays none)
  [j ≥ 2]  = 1 if j ≥ 2 else 0
```

- `CR_chunk = m·T/2` — `m` calls, context ramps 0→T, so average re-read `T/2`. It re-reads
  **only its own content**, never earlier chunks — that reset is the per-call subtraction.
- `IT/k, OT/k, CW/k` — the unique work is **unchanged** by splitting, just divided among the
  `k` chunks (even-split assumption: equal calls per chunk).
- Overhead hits only chunks `2 … k`; chunk 1 is the original start.

**It sums back to the session formula** (§3.5):

| term | Σ over `j = 1 … k` |
|---|---|
| `CR_chunk` | `k · N·T/(2k) = N·T/2 = CR_split` |
| overhead | `(k−1) · O_cost` |
| `IT/k`, `OT/k`, `CW/k` | `IT`, `OT`, `CW` |

so `Σ_j C_chunk(j) = C_split`. The script computes the already-summed session form (faster,
needs only the aggregates); this per-chunk form is the equivalent decomposition underneath.

---

## 4. Results

Global average cost/token `c̄ = $0.6559 / MTok` (overhead unit = `α·T·c̄`, `α = 0.10`).

Every dollar figure below is given three ways: **total** (all 4,853 Claude Code sessions),
**/session** (total ÷ 4,853), and **%** (share of the full as-is cost).

#### Column glossary

Symbols: `IT`=input, `OT`=output, `CW`=cache-write, `CR`=cache-read tokens; `N`=API calls;
`P=2·CR/N`=peak context; `T`=threshold; `k=⌈P/T⌉`=chunks; `c̄`=global avg $/token; `α`=overhead %.

**Cost as-is table**

| Column | Meaning |
|---|---|
| **metric** | which cost: the full as-is cost, or the cache-read "context tax" portion of it |
| **total** | dollars summed across all 4,853 Claude Code sessions |
| **/ session** | total ÷ 4,853 (average per session) |
| **% of full** | share of the full as-is cost (full = 100%; cache-read tax = 70.6%) |

**Split-into-chunks table & comparison matrices**

| Column | Meaning |
|---|---|
| **T** | context threshold — start a new chunk each time the running context reaches `T` tokens |
| **α** | overhead size as a fraction of `T`; overhead per new chunk = `α·T·c̄` |
| **Split** | **the cost of a session once it is split into chunks** — i.e. the sum of all its chunk costs `C_chunk(1)+…+C_chunk(k)`, each chunk billed as a fresh mini-session (re-reads only itself) plus overhead on chunks 2…k. This is the new, lower bill under the chunking strategy, the counterpart to the *full* as-is cost. |
| **Split (total)** | the split cost summed across all 4,853 sessions |
| **Split /sess** | split cost ÷ 4,853 = **the per-session cost as split into chunks** |
| **Split %** | split cost as a % of the full as-is cost (what fraction you'd still pay) |
| **Saved (total)** | full cost − split cost = money saved, summed across all sessions |
| **Saved /sess** | saved ÷ 4,853 (per session) |
| **Saved %** | saved ÷ full = the **percentage of money saved** at that `T`/`α` |
| **OH (total)** | overhead dollars paid to re-establish context in the new chunks, `(k−1)·α·T·c̄` |
| **OH /sess** | overhead ÷ 4,853 (per session) |
| **% split** | share of sessions large enough to actually split (`k > 1`, i.e. peak `P > T`) |
| **Avg chunks** | average chunks per session (`k = ⌈P/T⌉`, averaged over all sessions) |

The three matrices show the same sweep as **(a)** saved $/session, **(b)** saved $ total, and
**(c)** saved % of full. The **percentage of saved money per threshold and α** is matrix **(c)**
— e.g. 100k @ α=10% saves **44.4%**, 50k @ α=10% saves **57.1%**.

**Per-bucket table**

| Column | Meaning |
|---|---|
| **bucket** | the token type the cost comes from: input, output, cache-write, cache-read, the overhead row, or TOTAL |
| **full total / /sess / %** | that bucket's cost *before* splitting: total, per session, and as % of the full column's total |
| **split total / /sess / %** | that bucket's cost *after* splitting: total, per session, and as % of the split column's total |
| **saved total / /sess / %** | full − split for that bucket: total, per session, and as % of the *total savings* (so cache-read = 100% of savings) |
| **cache-read tokens** | the rebilling tokens before vs after splitting, with the closed-form `N·T/2` check |

### Cost as-is (full, with quadratic rebilling)

| metric | total | / session | % of full |
|---|---|---|---|
| full cost | $29,187 | $6.0142 | 100% |
| of which cache-read "context tax" | $20,592 | $4.2431 | 70.6% |

### Split into chunks — at the two named thresholds, per overhead α

| α | T | Split (total) | Split /sess | Split % | Saved (total) | Saved /sess | Saved % | OH (total) | OH /sess | % split | Avg chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5%  | 100k | $16,206 | $3.3394 | 55.5% | $12,981 | $2.6748 | 44.5% | $13 | $0.0027 | 59% | 1.83 |
| 5%  | 150k | $19,432 | $4.0041 | 66.6% | $9,755  | $2.0101 | 33.4% | $10 | $0.0020 | 34% | 1.40 |
| 10% | 100k | $16,219 | $3.3422 | 55.6% | $12,968 | $2.6721 | 44.4% | $27 | $0.0055 | 59% | 1.83 |
| 10% | 150k | $19,442 | $4.0061 | 66.6% | $9,745  | $2.0081 | 33.4% | $19 | $0.0039 | 34% | 1.40 |
| 15% | 100k | $16,233 | $3.3449 | 55.6% | $12,954 | $2.6693 | 44.4% | $40 | $0.0082 | 59% | 1.83 |
| 15% | 150k | $19,451 | $4.0081 | 66.6% | $9,736  | $2.0062 | 33.4% | $29 | $0.0059 | 34% | 1.40 |
| 20% | 100k | $16,246 | $3.3476 | 55.7% | $12,941 | $2.6666 | 44.3% | $53 | $0.0109 | 59% | 1.83 |
| 20% | 150k | $19,461 | $4.0100 | 66.7% | $9,726  | $2.0042 | 33.3% | $38 | $0.0079 | 34% | 1.40 |

The overhead is so small (≤ $53 total, ≤ $0.011/session even at α=20%) that the savings barely
move with α.

### Comparison matrices — threshold (rows) × overhead α (cols), 10k → 300k sweep

Same sweep shown three ways. (Abbreviated past 200k; the script prints every 10k step.)

**(a) Saved $ / session**

| T | α=5% | α=10% | α=15% | α=20% | Avg chunks |
|---|---|---|---|---|---|
| 10k  | 4.0788 | 4.0746 | 4.0703 | 4.0661 | 13.94 |
| 20k  | 3.9190 | 3.9149 | 3.9108 | 3.9067 | 7.23 |
| 30k  | 3.7592 | 3.7552 | 3.7513 | 3.7474 | 4.99 |
| 40k  | 3.5995 | 3.5958 | 3.5920 | 3.5882 | 3.87 |
| 50k  | 3.4404 | 3.4368 | 3.4332 | 3.4296 | 3.20 |
| 60k  | 3.2823 | 3.2789 | 3.2755 | 3.2720 | 2.75 |
| 70k  | 3.1259 | 3.1227 | 3.1194 | 3.1161 | 2.42 |
| 80k  | 2.9719 | 2.9689 | 2.9658 | 2.9627 | 2.17 |
| 90k  | 2.8215 | 2.8186 | 2.8157 | 2.8128 | 1.99 |
| 100k | 2.6748 | 2.6721 | 2.6693 | 2.6666 | 1.83 |
| 110k | 2.5325 | 2.5300 | 2.5275 | 2.5250 | 1.69 |
| 120k | 2.3946 | 2.3923 | 2.3900 | 2.3877 | 1.58 |
| 130k | 2.2614 | 2.2593 | 2.2571 | 2.2550 | 1.50 |
| 140k | 2.1332 | 2.1312 | 2.1291 | 2.1271 | 1.45 |
| 150k | 2.0101 | 2.0081 | 2.0062 | 2.0042 | 1.40 |
| 160k | 1.8923 | 1.8905 | 1.8886 | 1.8868 | 1.35 |
| 170k | 1.7813 | 1.7795 | 1.7778 | 1.7760 | 1.31 |
| 180k | 1.6766 | 1.6750 | 1.6734 | 1.6717 | 1.28 |
| 190k | 1.5817 | 1.5802 | 1.5788 | 1.5773 | 1.23 |
| 200k | 1.4967 | 1.4955 | 1.4942 | 1.4929 | 1.19 |
| 250k | 1.2197 | 1.2191 | 1.2186 | 1.2180 | 1.07 |
| 300k | 1.0359 | 1.0355 | 1.0351 | 1.0346 | 1.04 |

**(b) Saved $ total (all 4,853 sessions)**

| T | α=5% | α=10% | α=15% | α=20% |
|---|---|---|---|---|
| 10k  | 19,795 | 19,774 | 19,753 | 19,733 |
| 20k  | 19,019 | 18,999 | 18,979 | 18,959 |
| 30k  | 18,243 | 18,224 | 18,205 | 18,186 |
| 40k  | 17,468 | 17,450 | 17,432 | 17,414 |
| 50k  | 16,696 | 16,679 | 16,661 | 16,644 |
| 60k  | 15,929 | 15,912 | 15,896 | 15,879 |
| 70k  | 15,170 | 15,154 | 15,138 | 15,123 |
| 80k  | 14,423 | 14,408 | 14,393 | 14,378 |
| 90k  | 13,693 | 13,679 | 13,665 | 13,650 |
| 100k | 12,981 | 12,968 | 12,954 | 12,941 |
| 110k | 12,290 | 12,278 | 12,266 | 12,254 |
| 120k | 11,621 | 11,610 | 11,599 | 11,587 |
| 130k | 10,975 | 10,964 | 10,954 | 10,943 |
| 140k | 10,353 | 10,343 | 10,333 | 10,323 |
| 150k | 9,755  | 9,745  | 9,736  | 9,726  |
| 160k | 9,183  | 9,174  | 9,165  | 9,156  |
| 170k | 8,644  | 8,636  | 8,627  | 8,619  |
| 180k | 8,137  | 8,129  | 8,121  | 8,113  |
| 190k | 7,676  | 7,669  | 7,662  | 7,655  |
| 200k | 7,264  | 7,257  | 7,251  | 7,245  |
| 250k | 5,919  | 5,916  | 5,914  | 5,911  |
| 300k | 5,027  | 5,025  | 5,023  | 5,021  |

**(c) Saved % of full as-is cost**

| T | α=5% | α=10% | α=15% | α=20% |
|---|---|---|---|---|
| 10k  | 67.8% | 67.7% | 67.7% | 67.6% |
| 20k  | 65.2% | 65.1% | 65.0% | 65.0% |
| 30k  | 62.5% | 62.4% | 62.4% | 62.3% |
| 40k  | 59.8% | 59.8% | 59.7% | 59.7% |
| 50k  | 57.2% | 57.1% | 57.1% | 57.0% |
| 60k  | 54.6% | 54.5% | 54.5% | 54.4% |
| 70k  | 52.0% | 51.9% | 51.9% | 51.8% |
| 80k  | 49.4% | 49.4% | 49.3% | 49.3% |
| 90k  | 46.9% | 46.9% | 46.8% | 46.8% |
| 100k | 44.5% | 44.4% | 44.4% | 44.3% |
| 110k | 42.1% | 42.1% | 42.0% | 42.0% |
| 120k | 39.8% | 39.8% | 39.7% | 39.7% |
| 130k | 37.6% | 37.6% | 37.5% | 37.5% |
| 140k | 35.5% | 35.4% | 35.4% | 35.4% |
| 150k | 33.4% | 33.4% | 33.4% | 33.3% |
| 160k | 31.5% | 31.4% | 31.4% | 31.4% |
| 170k | 29.6% | 29.6% | 29.6% | 29.5% |
| 180k | 27.9% | 27.9% | 27.8% | 27.8% |
| 190k | 26.3% | 26.3% | 26.3% | 26.2% |
| 200k | 24.9% | 24.9% | 24.8% | 24.8% |
| 250k | 20.3% | 20.3% | 20.3% | 20.3% |
| 300k | 17.2% | 17.2% | 17.2% | 17.2% |

**Optimal threshold (max net savings), per α — always the smallest, since overhead never bites:**

| α | Optimal T | Saved (total) | Saved / session | Saved % |
|---|---|---|---|---|
| 5%  | 10k | $19,795 | $4.0788 | 67.8% |
| 10% | 10k | $19,774 | $4.0746 | 67.7% |
| 15% | 10k | $19,753 | $4.0703 | 67.7% |
| 20% | 10k | $19,733 | $4.0661 | 67.6% |

### Per-bucket breakdown (explicit per-call simulation, α=10%)

Built the literal way: walk every call, add its cost into the right bucket, and **reset the
context to 0 at each threshold crossing** so later calls stop re-reading the earlier chunk
(that reset *is* the per-call subtraction of earlier context-window re-reads). Input, output
and cache-write are the same unique work either way; only cache-read changes.

Each $ is shown three ways: **total | /session | %** (% = share of that column's own total;
for the saved column, % = share of total savings).

**T = 100k**

| bucket | full total | full /sess | full % | split total | split /sess | split % | saved total | saved /sess | saved % |
|---|---|---|---|---|---|---|---|---|---|
| input | 55 | 0.0114 | 0% | 55 | 0.0114 | 0% | 0 | 0.0000 | 0% |
| output | 2,411 | 0.4968 | 8% | 2,411 | 0.4968 | 15% | 0 | 0.0000 | 0% |
| cache-write | 6,129 | 1.2630 | 21% | 6,129 | 1.2630 | 39% | 0 | 0.0000 | 0% |
| **cache-read** | **20,592** | **4.2431** | **71%** | **7,003** | **1.4430** | **45%** | **13,589** | **2.8001** | **100%** |
| overhead | 0 | 0.0000 | 0% | 28 | 0.0058 | 0% | −28 | −0.0058 | −0% |
| **TOTAL** | **29,187** | **6.0142** | **100%** | **15,626** | **3.2199** | **100%** | **13,561** | **2.7943** | **46%** |

cache-read tokens: full 43,351M → split 14,742M.

**T = 150k**

| bucket | full total | full /sess | full % | split total | split /sess | split % | saved total | saved /sess | saved % |
|---|---|---|---|---|---|---|---|---|---|
| input | 55 | 0.0114 | 0% | 55 | 0.0114 | 0% | 0 | 0.0000 | 0% |
| output | 2,411 | 0.4968 | 8% | 2,411 | 0.4968 | 13% | 0 | 0.0000 | 0% |
| cache-write | 6,129 | 1.2630 | 21% | 6,129 | 1.2630 | 33% | 0 | 0.0000 | 0% |
| **cache-read** | **20,592** | **4.2431** | **71%** | **9,791** | **2.0175** | **53%** | **10,801** | **2.2256** | **100%** |
| overhead | 0 | 0.0000 | 0% | 20 | 0.0041 | 0% | −20 | −0.0041 | −0% |
| **TOTAL** | **29,187** | **6.0142** | **100%** | **18,406** | **3.7928** | **100%** | **10,781** | **2.2215** | **37%** |

cache-read tokens: full 43,351M → split 20,613M.

**What it shows:** three buckets don't move at all — splitting changes *only* the re-reading.
All savings live in the cache-read bucket (100k: $20,592 → $7,003), net of the tiny overhead.

**Method note — simulation vs closed form.** The explicit per-call simulation gives cache-read
14,742M @ 100k, while the closed form `N·T/2` gives 16,350M — so the simulation's net savings
($13,561) runs ~5% above the closed-form's ($12,968). Same model; the closed form is the smooth
continuous-triangle average, the simulation is the discrete saw-tooth with a hard reset (plus
the per-session `min(CR, …)` behaviour), so slightly less cache-read survives. The per-bucket
conclusion is identical either way.

---

## 5. Key finding & caveat

With the overhead defined as `α · T · c̄`, the **overhead is negligible at every α we swept
(5%–20%)** — at 100k it ranges from just $13 (α=5%) to $53 (α=20%), versus ~$13k saved. The
α=5% and α=20% saved/session differ only in the 3rd–4th decimal (e.g. 100k: $2.6748 vs
$2.6666). Reason: `c̄ = $0.656/MTok` is dragged way down by the dirt-cheap cache-read tokens
(0.10×) that dominate the token count, so `α·T` worth of an *average* token costs almost
nothing.

Consequence: savings are **monotonic — smaller chunks always win**, for all α, with no
break-even inside the sweep (optimal T = 10k, the smallest tested, for every α). If you want a
realistic trade-off that produces a true optimal chunk size, the overhead needs more teeth —
e.g. bill it at the **input rate (1.0×)** or **cache-write rate (1.25×)** instead of the
blended `c̄`, or make it scale super-linearly with `k`.

---

## 6. Conclusion

### What splitting can save — ideal ceiling vs actual reachable bound

Splitting only ever removes the **cache-read** cost (the re-reading of old context). Input,
output and cache-write are the same real work however you chunk, so they can never be saved.

There are **two different "upper bounds"** and they are NOT the same number:

**1. Ideal ceiling (never actually reached).** If you could remove *all* cache-read, you'd save
the entire context tax. This only happens in the limit `T → 0` (infinitely small chunks), which
is not physical — a chunk can't be smaller than the base context one call already carries — so
this is a theoretical ceiling we never hit:

| | total | / session | % of full |
|---|---|---|---|
| Full cost (as-is) | $29,187 | $6.0142 | 100% |
| *Ideal* ceiling (remove ALL cache-read) | $20,592 | $4.2431 | 70.6% |
| Never-savable floor (input+output+cache-write) | $8,595 | $1.7712 | 29.4% |

**2. Actual upper bound (what we really reach).** With real, finite chunks the savings are
always *less* than that ideal, because each chunk still re-reads its own ≤`T` of context and
pays overhead. The actual upper bound is therefore **the best split we actually compute** — one
of the rows in the sweep, not the 70.6% ideal. From the full 10k→300k sweep the highest reached
value is:

| best split reached | T | α | saved total | saved / session | saved % |
|---|---|---|---|---|---|
| **smallest chunk swept** | 10k | 5% | $19,795 | $4.0788 | **67.8%** |

So the **actual achievable upper bound ≈ 67.8% ($4.08/session)** at the smallest threshold we
tested — close to, but below, the 70.6% ideal. And the smaller the chunk, the closer to the
ideal you creep (but never reach it).

### Caveat: the overhead depends on T, so the best split sits at *lower* T

Overhead per new chunk is `α·T·c̄` — it **grows with `T`**. And the cache-read you remove also
grows as `T` shrinks (`CR_split = N·T/2`). Both effects point the same way: **smaller `T` saves
more.** So the best split is always toward the **low** end of the threshold range, not the high
end. (Higher `α` also costs more, so lower `α` is better too — but the effect is tiny here.)

### Best split around 100k

Within the 100k neighbourhood, the savings climb steadily as you lower `T` (α=5%, the cheapest
overhead):

| T | saved total | saved / session | saved % |
|---|---|---|---|
| 120k | $11,621 | $2.3946 | 39.8% |
| 110k | $12,290 | $2.5325 | 42.1% |
| **100k** | **$12,981** | **$2.6748** | **44.5%** |
| 90k | $13,693 | $2.8215 | 46.9% |
| 80k | $14,423 | $2.9719 | 49.4% |

- **At exactly T=100k:** best is **α=5% → save $12,981 ($2.67/session, 44.5%)**; α barely matters
  (α=20% still saves 44.3%).
- **Best in the 80k–120k band:** **T=80k, α=5% → save $14,423 ($2.97/session, 49.4%)** — i.e.
  pushing the threshold down from 100k to 80k lifts savings from ~44.5% to ~49.4%.

So: a ~100k split recovers **~44–49% of total spend (~$2.7–3.0 per session)**, and the absolute
ceiling this whole approach can ever reach is **70.6% (~$4.24/session)** — the full size of the
quadratic re-reading tax it is designed to remove.
