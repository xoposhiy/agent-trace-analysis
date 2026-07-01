# Splitting Sessions into Chunks — **Summary-Regeneration Overhead**

**Scope:** Claude Code sessions only — **4,853 sessions** from the `sessions` table of
`SALT-NLP/SWE-chat`. We use `sessions` (not `conversations`) because it is the trustworthy
source for absolute token/cost totals.

Script: [`context_rot_split_summary_overhead.py`](context_rot_split_summary_overhead.py)

This redoes the chunk-splitting analysis with a **different, realistic overhead** and a
**per-chunk** cost model (no blended-average shortcut). Thresholds swept **100k → 150k** in
10k steps, multiple α.

---

## 1. What changed vs the old model

The old overhead was a flat `α·T·c̄` priced at the global blended `$/token` (`c̄ ≈ $0.66/MTok`,
dragged down by cheap cache-read). It was negligible (~$27 total) so α didn't matter and the
optimum was always the smallest chunk.

**New overhead = the real cost of summarising the previous chunk and seeding it into the next
one as a fresh session start.** For each *new* chunk (chunks 2…k), a summary of size
`s = α·T` tokens costs, **at real per-token rates**:

| # | component | what it is | cost |
|---|---|---|---|
| 1 | **Output** | the model *generates* the α·T summary | `r_out · s` |
| 2 | **Cache-write** | the summary is loaded into the new chunk at the start | `1.25·r_in · s` |
| 3 | **Cache-read** | the summary then sits in context, re-read on the chunk's remaining `m_j−1` calls | `0.10·r_in · s·(m_j−1)` |

And costs are computed **per chunk** (the linear ramp — each chunk re-reads only its own ≤T
content, earlier chunks subtracted by the reset) **plus** this overhead on chunks 2…k — *not*
an average token price.

Each chunk is billed as an **independent mini-session** with the **last chunk partial** (it
climbs only its true remainder height, not a full T):
```
k = ⌈P/T⌉ ,  P = 2·CR/N ,  R = P − (k−1)·T   (last-chunk remainder, 0 < R ≤ T)
m_full = T·N/P  (calls in a full chunk)  ,  m_last = R·N/P  (calls in the last chunk)

CR_split = (N/2P)·[ (k−1)·T² + R² ]        ← first k−1 chunks climb T, last climbs only R
                                             (≤ N·T/2 ; each chunk re-reads ONLY its own content)

C_split = r_in·IT + r_out·OT + 1.25·r_in·CW + 0.10·r_in·CR_split
          + Σ_new chunks [ r_out·s + 1.25·r_in·s + 0.10·r_in·s·(m_j−1) ]      s = α·T

savings = 0.10·r_in·(CR − CR_split)                              ← old re-reading removed
          − Σ_new chunks [ r_out·s + 1.25·r_in·s + 0.10·r_in·s·(m_j−1) ]   ← summary overhead
```
The summary overhead's cache-read (`s·(m_j−1)`) is **only the carried α·T summary** re-read
inside the new chunk — never the previous chunk's full content (that re-reading is what's
removed). Input/output/cache-write are the same real work, so they're re-allocated across
chunks (last one smaller) but their total never drops; only CR and the overhead move.

Rates (blended Opus-4.x): `r_in=$4.75`, `r_out=$23.75` per MTok; cache-write `1.25×`,
cache-read `0.10×`. *(The summary's first load is billed as cache-write; the **output** term,
5× input, dominates the new overhead.)*

---

## 2. Cost as-is (unchanged baseline)

| metric | total | / session | % of full |
|---|---|---|---|
| full cost | $29,187 | $6.0142 | 100% |
| of which cache-read "context tax" | $20,592 | $4.2431 | 70.6% |

---

## 3. Chunks are **NOT** a fixed price — cost depends on a chunk's calls/turns

Each chunk is billed **as a fresh mini-session** — it counts *its own* tokens × rates:
```
chunk cost = (m_j/N)·IT·r_in + (m_j/N)·OT·r_out + (m_j/N)·CW·1.25·r_in   ← its share of unique work
           + (m_j·h_j/2)·0.10·r_in                                       ← its OWN cache-read
           + [chunk ≥ 2] summary overhead (grows with m_j)
   full chunks: h_j = T,  m_j = T·N/P     last chunk: h_j = R = P−(k−1)·T,  m_j = R·N/P
```
So a chunk with **more API calls/turns (`m_j`)** and more work costs more, and the **partial
last chunk is smaller in *every* bucket** (input, output, CW, *and* CR), not just cache-read.
The spread across all individual chunks (α=10%, exact) confirms it:

| T | # chunks | mean $ | p10 | median | p90 | max | mean calls/chunk `m` |
|---|---|---|---|---|---|---|---|
| 100k | 8,899 | $1.98 | $0.33 | **$1.16** | $4.31 | **$54.16** | 36.7 |
| 150k | 6,797 | $2.94 | $0.35 | $1.51 | $6.98 | **$97.34** | 48.1 |

At T=100k a single chunk runs from **~$0.33 (p10) to ~$4.31 (p90)** — a ~13× spread — tail to
**$54**. Same 100k cap, wildly different cost, driven by the calls/turns inside it. These
per-chunk costs **sum exactly** to the exact-variant session total (verified to the cent).

**Why input/output/CW are only *re-allocated*, never reduced.** Splitting changes the
**cache-read** bucket only. The agent reads the same files and generates the same outputs no
matter how you chunk, so `Σ chunks` of input/output/CW always equals the session's IT/OT/CW —
chunking can't save real work, only the re-reading of old context. The per-chunk allocation
(proportional to each chunk's calls) is therefore just *attribution*: the partial last chunk
shows a smaller input/output/CW, but the **total** in those buckets is invariant. Only CR and
the summary overhead move the totals. *(Within one session the chunks are equal-sized full ones
plus one partial; chunk 1 carries no overhead; the spread above is **across** sessions.)*

---

## 4. Results — split into chunks, per overhead α

### Detailed thresholds

| α | T | Split /sess | Split % | Saved (total) | Saved /sess | **Saved %** | OH (total) | OH /sess | % split | Avg chunks |
|---|---|---|---|---|---|---|---|---|---|---|
| 5%  | 100k | $3.4265 | 57.0% | $12,558 | $2.5878 | **43.0%** | $957 | $0.1972 | 59% | 1.83 |
| 5%  | 150k | $3.9570 | 65.8% | $9,984  | $2.0572 | **34.2%** | $759 | $0.1564 | 34% | 1.40 |
| 10% | 100k | $3.6237 | 60.3% | $11,601 | $2.3905 | **39.7%** | $1,914 | $0.3944 | 59% | 1.83 |
| 10% | 150k | $4.1134 | 68.4% | $9,225  | $1.9008 | **31.6%** | $1,518 | $0.3128 | 34% | 1.40 |
| 15% | 100k | $3.8209 | 63.5% | $10,644 | $2.1933 | **36.5%** | $2,871 | $0.5917 | 59% | 1.83 |
| 15% | 150k | $4.2698 | 71.0% | $8,466  | $1.7444 | **29.0%** | $2,277 | $0.4691 | 34% | 1.40 |
| 20% | 100k | $4.0181 | 66.8% | $9,687  | $1.9961 | **33.2%** | $3,829 | $0.7889 | 59% | 1.83 |
| 20% | 150k | $4.4262 | 73.6% | $7,707  | $1.5880 | **26.4%** | $3,036 | $0.6255 | 34% | 1.40 |

**α bites.** At 100k the saved share runs 43.0% (α=5%) → 33.2% (α=20%).

> **All percentages here (Split %, Saved %) are over all 4,853 sessions — small/un-split
> sessions included in the denominator — so they are true corpus-wide averages, not
> heavy-session-only figures.**

*`% split`* = the share of the 4,853 sessions large enough to actually split (`k>1`, i.e. peak
`P = 2·CR/N > T`); the rest stay whole and contribute $0 savings and $0 overhead. It falls as T
rises (59% reach >100k, only 34% reach >150k). *`Avg chunks`* is the mean `k` over **all**
sessions (including the un-split `k=1` ones), so it sits just above 1.

### Where the overhead dollars go (α=10%)

| T | OH total | generation (output) | cache-write | re-read (cache-read) |
|---|---|---|---|---|
| 100k | $1,914 | **$961** | $240 | $713 |
| 150k | $1,518 | $693 | $173 | $652 |

**Summary generation (output, 5× input) dominates**; cache-write is minor; the re-read pedestal
(the carried α·T summary, not the previous chunk) is in between.

### Saved % of full — sweep 100k → 150k

| T | α=5% | α=10% | α=15% | α=20% | Avg chunks |
|---|---|---|---|---|---|
| 100k | **43.03** | 39.75 | 36.47 | 33.19 | 1.83 |
| 110k | 41.11 | 38.01 | 34.90 | 31.80 | 1.69 |
| 120k | 39.43 | 36.49 | 33.54 | 30.59 | 1.58 |
| 130k | 37.80 | 34.98 | 32.17 | 29.35 | 1.50 |
| 140k | 36.10 | 33.40 | 30.70 | 28.00 | 1.45 |
| 150k | 34.21 | 31.61 | 29.01 | 26.40 | 1.40 |

### Optimal threshold per α (over 100k → 150k)

| α | Optimal T | Saved /sess | Saved % |
|---|---|---|---|
| 5% | 100k | $2.5878 | **43.0%** |
| 10% | 100k | $2.3905 | **39.7%** |
| 15% | 100k | $2.1933 | **36.5%** |
| 20% | 100k | $1.9961 | **33.2%** |

Savings rise monotonically as T shrinks → the best swept threshold is the smallest (100k) for
every α.

---

## 5. Conclusion — savings and the upper bound

**Savings (realistic summary overhead, exact partial last chunk).** Splitting still recovers a
large share of spend, but α now controls how much:

- **Across 100k → 150k:** from **~26%** (150k, α=20%) up to **~43%** (100k, α=5%) of total
  spend (~$1.59–$2.59 / session), i.e. **$7.7k–$12.6k**.
- **At 100k specifically:** **33.2% – 43.0%** depending on α.
- Savings fall monotonically as T rises across this band — no interior optimum — so the most
  you can reach in-range is pinned at the **smallest threshold, 100k**.

**These %s are the true per-user average — small sessions included.** Saved % is
`total saved ÷ total full` over **all 4,853 sessions**, so the ~41% that never reach T (and save
$0) are in the denominator; they can only pull the figure down. It still lands at ~33–43%
because those small sessions are cheap — nearly all the cost (and savings) lives in the big
sessions. Restricted to just the ~59% that actually split (T=100k, α=15%), the saving is only
slightly higher — **38.2% vs 36.5%** — but **$3.72 vs $2.19 per session** (spread over fewer,
heavier sessions).

**Upper bound — three numbers, not one.**

| bound | value | reachable? |
|---|---|---|
| **Ideal ceiling** (remove *all* cache-read) | **70.6%** ($4.24/sess) | ✗ only as T→0 (unphysical) |
| **In-range upper bound** (T=100k) | **33.2% – 43.0%** (α=20% → 5%) | ✓ best at 100k |
| **At T=150k** | **26.4% – 34.2%** | ✓ |

Within the **100k → 150k** band the best you can do is the **smallest threshold (100k)**: lower
T removes more cache-read while the summary overhead stays modest. The reachable share sits well
**below** the 70.6% ideal because the summary overhead leaves an **irreducible floor** that does
not vanish: regenerating a summary for each of `k ≈ P/T` chunks costs ≈ `k · r_out·(α·T) =
r_out·α·P` — a near-constant ~`α·P` of **output** tokens. That floor scales directly with α:
**a bigger carried summary buys correctness/continuity but linearly erodes savings** (each +5%
of α ≈ −3.3 points at 100k: 43.0% → 36.5% → 33.2%).

## Final conclusion — a feasible upper bound on savings

> **Chunking the SWE-chat Claude Code corpus at a 100k threshold saves ~33–43% of total spend
> (~$2.0–$2.6 per session), and this is a FEASIBLE UPPER BOUND — computed over ALL 4,853
> sessions, small ones included.**
>
> **The ~41% of sessions that never reach the threshold are counted in full but save nothing,
> so they only drag the figure down — yet it still lands at ~33–43%. That means the number is
> not an optimistic best-case over cherry-picked heavy sessions; it is the realistic average a
> user would actually see across their whole mix of sessions.** It also stays **below** the
> unreachable 70.6% ideal (remove all cache-read), because the price of re-establishing context
> via a summary — dominated by summary **generation** (output, 5× input) — leaves an
> irreducible floor of ~`α·P` output tokens that never vanishes.
>
> **In short: ~33–43% at 100k is a conservative, all-sessions-inclusive, realistically
> achievable ceiling — not a theoretical maximum.**

---

## 6. Caveat

Modeling choices (in the script): each chunk is billed as an independent mini-session that
re-reads only its own content (earlier chunks subtracted by the reset); the **last chunk is
partial** — it climbs only its remainder height `R = P−(k−1)·T`, and its share of
input/output/cache-write is proportional to its calls (`m_last/N`), so it is smaller in every
bucket. The carried summary sits on top of each new chunk and its first load is priced as
cache-write. Dollar figures are token-based estimates at blended Opus-4.x rates. If a session
never reaches T (`P<T`), `k=1` and it pays nothing. Per-chunk costs sum exactly to the session
totals (verified).
