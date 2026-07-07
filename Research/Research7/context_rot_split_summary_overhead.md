# Splitting Sessions into Chunks — Summary-Regeneration Overhead

**Scope:** 4,853 Claude Code sessions from the `sessions` table of `SALT-NLP/SWE-chat`
(the trustworthy source for absolute token/cost totals).
**Code:** [`main.py`](main.py) (drivers + reporting) + [`chunk_split_model.py`](chunk_split_model.py) (reusable math model).

---

## 0. Introduction

Most of what a long agent session pays for is **cache-read** — the whole growing context
re-read on every API call (~70% of the bill here). A natural way to cut that is to **split a
session into chunks**: cap the context at a threshold, and when it would overflow, reset into a
fresh chunk seeded with a short summary of the previous one. Each chunk then re-reads only its
own small context instead of the full history.

**What this experiment measures:** *a feasible upper bound on how much token spend chunking
could save* on the SWE-chat Claude Code corpus — not a naive best case, but a realistic one
that **charges the true cost of the summaries** (generating + carrying them) and is computed
over **all** sessions (small ones that never split included). We sweep the chunk threshold
(100k → 150k) and the summary size (α = 5–20% of the threshold) to see how far savings can go
and where the summary overhead stops them.

---

## 1. The model

Over a session the context grows ~linearly to a peak **P** across **N** API calls.
Every call re-reads the whole context, so total cache-read = area under the ramp =
**N·P/2** — this is ~70% of the bill (the "context tax"). If we cap context at a
threshold **T**, we *reset* into **k = ⌈P/T⌉** chunks; each chunk then re-reads only
its own ≤T content (k small triangles instead of one big one). The price of a reset:
each new chunk (2…k) is seeded with a **summary of size s = α·T**, billed at real rates:

| component | what it is | cost |
|---|---|---|
| **output** | the model generates the summary | `r_out · s` |
| **cache-write** | the summary is loaded at the chunk start | `1.25 · r_in · s` |
| **cache-read** | the summary is re-read on the chunk's other `m−1` calls | `0.10 · r_in · s · (m−1)` |

Each chunk is billed as an independent mini-session, so a chunk's cost scales with the
calls/turns inside it — chunks are **not** a fixed price. The tables below use the **exact**
geometry: the last chunk climbs only its true remainder `R = P−(k−1)·T` (not a full T), so it
is smaller in every bucket. Per-chunk costs sum exactly to the session total. (The script also
computes a slightly conservative even-split variant, ~2–3 points lower.)

```
P = 2·CR/N            k = ⌈P/T⌉            s = α·T
CR_split = (N/2P)·Σ hⱼ²                     (hⱼ = chunk heights; last = R in the exact variant)
C_split  = r_in·IT + r_out·OT + 1.25·r_in·CW + 0.10·r_in·CR_split
           + Σ_{new chunks} [ r_out·s + 1.25·r_in·s + 0.10·r_in·s·(mⱼ−1) ]
```

Rates (blended Opus-4.x): `r_in = $4.75`, `r_out = $23.75` per MTok; cache-write `1.25×`,
cache-read `0.10×`. Only cache-read and the summary overhead move totals — input/output/
cache-write are the same real work regardless of how you chunk, so they are re-allocated
across chunks but never reduced. **Summary generation (output, 5× input) dominates the overhead.**

---

## 2. Baseline (no splitting)

| metric | total | / session | % of full |
|---|---|---|---|
| full cost | **$29,187** | $6.0142 | 100% |
| of which cache-read "context tax" | $20,592 | $4.2431 | **70.6%** |

---

## 3. Results — savings per overhead α (exact: each chunk a fresh session, last chunk partial)

| α | T | Saved (total) | Saved /sess | Saved % | Overhead (total) | Avg chunks |
|---|---|---|---|---|---|---|
| 5%  | 100k | $12,558 | $2.5878 | **43.0%** | $957   | 1.83 |
| 5%  | 150k | $9,984  | $2.0572 | **34.2%** | $759   | 1.40 |
| 10% | 100k | $11,601 | $2.3905 | **39.7%** | $1,914 | 1.83 |
| 10% | 150k | $9,225  | $1.9008 | **31.6%** | $1,518 | 1.40 |
| 15% | 100k | $10,644 | $2.1933 | **36.5%** | $2,871 | 1.83 |
| 15% | 150k | $8,466  | $1.7444 | **29.0%** | $2,277 | 1.40 |
| 20% | 100k | $9,687  | $1.9961 | **33.2%** | $3,829 | 1.83 |
| 20% | 150k | $7,707  | $1.5880 | **26.4%** | $3,036 | 1.40 |

**Columns:**
- **α** — summary size as a fraction of the threshold (`s = α·T`); larger α = more faithful carry, more overhead.
- **T** — context threshold; a new chunk starts when context would exceed it.
- **Saved (total)** — dollars saved across all 4,853 sessions vs the $29,187 baseline.
- **Saved /sess** — average saving per session (all sessions, split or not).
- **Saved %** — saved ÷ full cost; a true corpus-wide average (un-split sessions are in the denominator).
- **Overhead (total)** — the summary-regeneration cost that splitting *adds back*.
- **Avg chunks** — mean `k` over all sessions; just above 1 because most sessions never reach T and stay whole.

---

## 4. Sweep 100k → 150k — Saved % of full (exact)

| T | α=5% | α=10% | α=15% | α=20% | Avg chunks |
|---|---|---|---|---|---|
| 100k | **43.03** | 39.75 | 36.47 | 33.19 | 1.83 |
| 110k | 41.11 | 38.01 | 34.90 | 31.80 | 1.69 |
| 120k | 39.43 | 36.49 | 33.54 | 30.59 | 1.58 |
| 130k | 37.80 | 34.98 | 32.17 | 29.35 | 1.50 |
| 140k | 36.10 | 33.40 | 30.70 | 28.00 | 1.45 |
| 150k | 34.21 | 31.61 | 29.01 | 26.40 | 1.40 |

Savings rise monotonically as T shrinks — no interior optimum — so the smallest in-band
threshold (**100k**) is always best.

---

## 5. Conclusion

**Splitting recovers a large share of spend, and α controls how much.** At the best in-band
threshold (100k), savings run **43% → 33%** as α goes 5% → 20% — about **$2.0–$2.6 / session**,
**$9.7k–$12.6k** total. Across the whole band savings span **~26% – 43%**. They fall monotonically
as T rises, so there is no interior optimum: the smallest threshold wins in-range.

**This is a realistic, all-sessions-inclusive figure, not a cherry-picked best case.** Saved %
is `total saved ÷ total full` over all 4,853 sessions, so the ~40–66% that never reach T (and
save $0) are counted in the denominator and can only pull the number down — it still lands at
30–43% because nearly all cost lives in the big sessions.

**It sits below the 70.6% ideal for a structural reason.** Removing *all* cache-read would save
70.6%, but that needs T→0 (unphysical). The gap is an **irreducible floor**: regenerating a
summary for each of `k ≈ P/T` chunks costs ≈ `r_out · α · P` output tokens (~`α·P`) that never
vanish. That floor scales with α — each +5% of α costs ≈ 3.3 points at 100k (43.0% → 39.8% → 36.5% → 33.2%).

> **Bottom line:** chunking this corpus at a 100k threshold saves **~33–43% of total spend**
> (~$2/session) — a feasible, conservative upper bound that stays below the unreachable 70.6%
> ideal because re-establishing context via summaries (output-dominated) can never be free.

---

## 6. Caveat

Modeling choices: each chunk is billed as an independent mini-session re-reading only its own
content; in the exact variant the **last chunk is partial** (height `R = P−(k−1)·T`) and its
share of input/output/cache-write is proportional to its calls, so it is smaller in every
bucket. The carried summary's first load is priced as cache-write. Sessions that never reach T
(`P < T`) stay whole (`k=1`, $0 saved, $0 overhead). Figures are token-based estimates at
blended Opus-4.x rates; per-chunk costs sum exactly to the session totals (verified).
