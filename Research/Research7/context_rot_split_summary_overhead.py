"""Cost of splitting SWE-chat sessions into chunks -- with a SUMMARY-based overhead.

Scope: Claude Code sessions ONLY, from the `sessions` table.

Difference vs context_rot_split_cost.py
---------------------------------------
The old overhead was a flat `alpha*T*c_bar` priced at the blended average $/token
(dominated by cheap cache-read), so it was negligible and "smaller chunks always
won". Here the overhead is the realistic cost of *summarising the previous chunk
and seeding it into the next one as a fresh session start*, billed at REAL rates:

For each NEW chunk (chunks 2..k), a summary of size  s = alpha*T  tokens:
  1. OUTPUT to generate the summary          : r_out          * s
  2. CACHE-WRITE to load it at chunk start    : 1.25 * r_in     * s
  3. CACHE-READ as it sits in context and is
     re-read on the chunk's remaining (m-1)
     calls (m = N/k calls per chunk)          : 0.10 * r_in     * s * (m-1)

Per-chunk model (NOT a global average): each chunk re-reads only its OWN <=T
content (the linear ramp, total cache-read N*T/2 across the session -- earlier
chunks subtracted by the reset), PLUS the summary overhead above on chunks 2..k.

    C_split = r_in*IT + r_out*OT + 1.25*r_in*CW + 0.10*r_in*CR_split
              + (k-1)*[ r_out*s + 1.25*r_in*s + 0.10*r_in*s*(m-1) ]
    CR_split = min(CR, N*T/2)        s = alpha*T      m = N/k

    savings = 0.10*r_in*(CR - CR_split)                      # rebilling removed
              - (k-1)*[ r_out*s + 1.25*r_in*s + 0.10*r_in*s*(m-1) ]   # summary OH

Note: the summary's first load is billed as cache-write (carried context is
cached). Billing it as fresh input (1.0x) instead of 1.25x changes the result by
a hair; the OUTPUT term (5x input) dominates the new overhead.
Dollar figures are token-based ESTIMATES at blended Anthropic rates.
"""
import glob
import math
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SESSIONS_PATH = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--SALT-NLP--SWE-chat/snapshots/*/sessions.parquet"))[0]

CLAUDE_CODE_AGENTS = ["Claude Code", "claude-code"]

R_IN = 4.75
R_OUT = 23.75
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

ALPHAS = [0.05, 0.10, 0.15, 0.20]
THRESHOLDS = [100_000, 150_000]                    # detailed tables
SWEEP = list(range(100_000, 150_001, 10_000))      # 100k -> 150k sweep, 10k steps


def full_cost(it, ot, cw, cr):
    return (it * R_IN + ot * R_OUT
            + cw * R_IN * CACHE_WRITE_MULT
            + cr * R_IN * CACHE_READ_MULT) / 1e6


def load_claude_code():
    t = pq.read_table(SESSIONS_PATH, columns=[
        "agent", "input_tokens", "output_tokens",
        "cache_creation_tokens", "cache_read_tokens", "api_call_count"])
    mask = pc.is_in(t.column("agent"), value_set=pa.array(CLAUDE_CODE_AGENTS))
    t = t.filter(mask)

    def col(c):
        return [x or 0 for x in t.column(c).to_pylist()]

    return (col("input_tokens"), col("output_tokens"),
            col("cache_creation_tokens"), col("cache_read_tokens"),
            col("api_call_count"))


def summary_overhead_cost(k, m, s):
    """Total summary overhead $ for one session (k chunks, m calls/chunk, s tokens).

    Applied to the (k-1) NEW chunks. Returns total and the 3-way breakdown.
    """
    if k <= 1:
        return 0.0, 0.0, 0.0, 0.0
    gen = R_OUT * s / 1e6                                  # output: make summary
    write = CACHE_WRITE_MULT * R_IN * s / 1e6             # cache-write: load it
    reread = CACHE_READ_MULT * R_IN * s * max(m - 1, 0) / 1e6  # cache-read pedestal
    per_chunk = gen + write + reread
    return (k - 1) * per_chunk, (k - 1) * gen, (k - 1) * write, (k - 1) * reread


def analyze(it, ot, cw, cr, n):
    sessions = list(zip(it, ot, cw, cr, n))
    total_full = sum(full_cost(a, b, c, d) for a, b, c, d, _ in sessions)
    total_cr_cost = sum(d * R_IN * CACHE_READ_MULT / 1e6 for *_, d, _ in sessions)

    results = {}
    for T in sorted(set(THRESHOLDS) | set(SWEEP)):
        for alpha in ALPHAS:
            s = alpha * T
            split = 0.0
            oh_total = oh_gen = oh_write = oh_reread = 0.0
            n_split = 0
            sum_chunks = 0
            for (a, b, c, d, nn) in sessions:
                if nn <= 0 or d <= 0:
                    split += full_cost(a, b, c, d)
                    sum_chunks += 1
                    continue
                P = 2.0 * d / nn
                k = 1 if P <= T else math.ceil(P / T)
                m = nn / k
                cr_split = min(d, nn * T / 2.0)
                base = (a * R_IN + b * R_OUT + c * R_IN * CACHE_WRITE_MULT) / 1e6
                base += cr_split * R_IN * CACHE_READ_MULT / 1e6
                oh, g, w, rr = summary_overhead_cost(k, m, s)
                split += base + oh
                oh_total += oh
                oh_gen += g
                oh_write += w
                oh_reread += rr
                sum_chunks += k
                if k > 1:
                    n_split += 1
            results[(T, alpha)] = {
                "split": split,
                "saved": total_full - split,
                "oh_total": oh_total,
                "oh_gen": oh_gen,
                "oh_write": oh_write,
                "oh_reread": oh_reread,
                "n_split": n_split,
                "avg_chunks": sum_chunks / len(sessions),
            }
    return total_full, total_cr_cost, results


def analyze_exact(it, ot, cw, cr, n):
    """EXACT variant: the last (partial) chunk ramps only to its true remainder
    height R = P - (k-1)*T instead of a full T.

    Geometry: context ramps 0->P at slope P/N; we reset at every T. The first
    (k-1) chunks each climb a full T (m_full = T*N/P calls); the last climbs only
    R (m_last = R*N/P calls). Cache-read = sum of triangle areas:
        CR_exact = (N/2P) * [ (k-1)*T^2 + R^2 ]   <=  N*T/2  (even-split form)
    Summary overhead uses each new chunk's own call count (full vs last).
    """
    sessions = list(zip(it, ot, cw, cr, n))
    total_full = sum(full_cost(a, b, c, d) for a, b, c, d, _ in sessions)
    results = {}
    for T in sorted(set(THRESHOLDS) | set(SWEEP)):
        for alpha in ALPHAS:
            s = alpha * T
            split = 0.0
            oh_total = oh_gen = oh_write = oh_reread = 0.0
            sum_chunks = 0
            n_split = 0
            for (a, b, c, d, nn) in sessions:
                if nn <= 0 or d <= 0:
                    split += full_cost(a, b, c, d)
                    sum_chunks += 1
                    continue
                P = 2.0 * d / nn
                if P <= T:
                    split += full_cost(a, b, c, d)
                    sum_chunks += 1
                    continue
                k = math.ceil(P / T)
                R = P - (k - 1) * T                       # remainder height (0,T]
                cr_split = (nn / (2.0 * P)) * ((k - 1) * T * T + R * R)
                m_full = T * nn / P
                m_last = R * nn / P
                base = (a * R_IN + b * R_OUT + c * R_IN * CACHE_WRITE_MULT) / 1e6
                base += cr_split * R_IN * CACHE_READ_MULT / 1e6
                gen = (k - 1) * R_OUT * s / 1e6
                write = (k - 1) * CACHE_WRITE_MULT * R_IN * s / 1e6
                reread = (CACHE_READ_MULT * R_IN * s
                          * ((k - 2) * max(m_full - 1, 0) + max(m_last - 1, 0))) / 1e6
                oh = gen + write + reread
                split += base + oh
                oh_total += oh
                oh_gen += gen
                oh_write += write
                oh_reread += reread
                sum_chunks += k
                n_split += 1
            results[(T, alpha)] = {
                "split": split,
                "saved": total_full - split,
                "oh_total": oh_total,
                "oh_gen": oh_gen,
                "oh_write": oh_write,
                "oh_reread": oh_reread,
                "n_split": n_split,
                "avg_chunks": sum_chunks / len(sessions),
            }
    return total_full, results


def per_chunk_costs(it, ot, cw, cr, n, T, alpha):
    """Cost of EVERY individual chunk, billed as a fresh mini-session (EXACT).

    Each chunk j counts ITS OWN tokens x rates:
      - calls m_j: full chunks (j<k-1) climb T over m_full=T*N/P calls; the LAST
        chunk climbs only its remainder R=P-(k-1)*T over m_last=R*N/P calls.
      - input/output/cache-write: that chunk's SHARE = (m_j/N) of IT/OT/CW
        (proportional to its calls, so the partial last chunk is smaller in EVERY
        bucket -- not just cache-read).
      - own cache-read: m_j * h_j / 2  (h_j = T for full chunks, R for the last).
      - summary overhead on chunks j>=2 (uses that chunk's own m_j).
    Returns list of (chunk_cost, m_calls, is_first_chunk).
    """
    s = alpha * T
    out = []
    for a, b, c, d, nn in zip(it, ot, cw, cr, n):
        if nn <= 0 or d <= 0:
            out.append((full_cost(a, b, c, d), nn, True))
            continue
        P = 2.0 * d / nn
        if P <= T:
            out.append((full_cost(a, b, c, d), nn, True))
            continue
        k = math.ceil(P / T)
        R = P - (k - 1) * T
        for j in range(k):
            h_j = T if j < k - 1 else R          # this chunk's context height
            m_j = h_j * nn / P                    # its calls (sum over j = N)
            frac = m_j / nn                       # its share of the unique work
            base = (a * frac * R_IN + b * frac * R_OUT
                    + c * frac * R_IN * CACHE_WRITE_MULT) / 1e6
            own_cr = (m_j * h_j / 2.0) * R_IN * CACHE_READ_MULT / 1e6
            oh = 0.0 if j == 0 else (
                R_OUT * s + CACHE_WRITE_MULT * R_IN * s
                + CACHE_READ_MULT * R_IN * s * max(m_j - 1, 0)) / 1e6
            out.append((base + own_cr + oh, m_j, j == 0))
    return out


def pctile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(p / 100 * len(xs)))
    return xs[i]


def print_chunk_distribution(it, ot, cw, cr, n):
    print("=== PER-CHUNK cost distribution -- chunks are NOT a fixed price ===")
    print("(every individual chunk across all sessions; alpha=10%)")
    print(f"{'T':>9} {'#chunks':>9}{'mean$':>9}{'p10':>8}{'median':>8}{'p90':>8}"
          f"{'max':>9}{'mean m(calls)':>15}")
    for T in THRESHOLDS:
        chunks = per_chunk_costs(it, ot, cw, cr, n, T, 0.10)
        costs = [c for c, _, _ in chunks]
        ms = [mm for _, mm, _ in chunks]
        mean = sum(costs) / len(costs)
        mean_m = sum(ms) / len(ms)
        print(f"{T:>9,} {len(costs):>9,}{mean:>9.4f}{pctile(costs,10):>8.4f}"
              f"{pctile(costs,50):>8.4f}{pctile(costs,90):>8.4f}{max(costs):>9.2f}"
              f"{mean_m:>15.1f}")
    print()


def main():
    it, ot, cw, cr, n = load_claude_code()
    ns = len(it)
    total_full, total_cr_cost, res = analyze(it, ot, cw, cr, n)

    print(f"=== SWE-chat split-cost -- SUMMARY overhead -- Claude Code only, {ns:,} sessions ===")
    print(f"rates: ${R_IN}/${R_OUT} per MTok (write {CACHE_WRITE_MULT}x, read {CACHE_READ_MULT}x)")
    print(f"overhead per new chunk = OUTPUT(s) + CACHE-WRITE(s) + CACHE-READ(s*(m-1)), s=alpha*T\n")
    print(f"cost AS-IS (full): ${total_full:,.0f} total | ${total_full/ns:.4f}/session | 100%")
    print(f"  of which cache-read tax: ${total_cr_cost:,.0f} | ${total_cr_cost/ns:.4f}/sess | "
          f"{100*total_cr_cost/total_full:.1f}%\n")

    # detailed tables, one per alpha
    for alpha in ALPHAS:
        print(f"--- overhead alpha = {alpha:.0%} of T (summary = {alpha:.0%}*T tokens) ---")
        print(f"{'T':>9} {'split$tot':>10}{'split/ss':>9}{'split%':>7}  "
              f"{'saved$tot':>10}{'saved/ss':>9}{'saved%':>7}  "
              f"{'OH$tot':>9}{'OH/ss':>8}  {'%split':>7}{'chunks':>8}")
        for T in THRESHOLDS:
            r = res[(T, alpha)]
            sp, sv, oh = r["split"], r["saved"], r["oh_total"]
            print(f"{T:>9,} {sp:>10,.0f}{sp/ns:>9.4f}{100*sp/total_full:>6.1f}%  "
                  f"{sv:>10,.0f}{sv/ns:>9.4f}{100*sv/total_full:>6.1f}%  "
                  f"{oh:>9,.0f}{oh/ns:>8.4f}  "
                  f"{100*r['n_split']/ns:>6.0f}%{r['avg_chunks']:>8.2f}")
        print()

    # overhead component breakdown at the named thresholds (alpha=10%)
    print("=== overhead breakdown (alpha=10%): where the summary $ goes ===")
    print(f"{'T':>9} {'OH total':>10}{'gen(out)':>10}{'write':>10}{'reread(cr)':>12}")
    for T in THRESHOLDS:
        r = res[(T, 0.10)]
        print(f"{T:>9,} {r['oh_total']:>10,.0f}{r['oh_gen']:>10,.0f}"
              f"{r['oh_write']:>10,.0f}{r['oh_reread']:>12,.0f}")
    print()

    # comparison matrices
    def matrix(title, cell):
        print(f"=== COMPARISON ({title}): threshold (rows) x alpha (cols) ===")
        print(f"{'T':>9} " + " ".join(f"a={a:.0%}".rjust(11) for a in ALPHAS)
              + f"{'chunks':>9}")
        for T in SWEEP:
            print(f"{T:>9,} " + " ".join(f"{cell(T, a):>11.4f}" for a in ALPHAS)
                  + f"{res[(T, ALPHAS[0])]['avg_chunks']:>9.2f}")
        print()

    matrix("saved $/SESSION", lambda T, a: res[(T, a)]["saved"] / ns)
    matrix("saved $ TOTAL", lambda T, a: res[(T, a)]["saved"])
    matrix("saved % of full", lambda T, a: 100 * res[(T, a)]["saved"] / total_full)

    # optimal threshold per alpha -- now overhead has teeth, so it may NOT be the smallest
    print("optimal threshold (MAX net savings) per alpha:")
    for alpha in ALPHAS:
        best = max(SWEEP, key=lambda T: res[(T, alpha)]["saved"])
        r = res[(best, alpha)]
        print(f"  alpha={alpha:.0%}: T={best:,} -> ${r['saved']:,.0f} total | "
              f"${r['saved']/ns:.4f}/session | {100*r['saved']/total_full:.1f}% of full")
    print()

    print_chunk_distribution(it, ot, cw, cr, n)

    # ---- EXACT variant: last chunk gets only its remainder height R, not full T ----
    _, exact = analyze_exact(it, ot, cw, cr, n)
    print("=== EVEN-SPLIT vs EXACT (last chunk = remainder height R) -- saved % of full ===")
    print(f"{'T':>9}  " + "  ".join(f"a={a:.0%}: even/exact" for a in ALPHAS))
    for T in SWEEP:
        cells = []
        for a in ALPHAS:
            ev = 100 * res[(T, a)]["saved"] / total_full
            ex = 100 * exact[(T, a)]["saved"] / total_full
            cells.append(f"{ev:5.1f}/{ex:5.1f}")
        print(f"{T:>9,}  " + "   ".join(cells))
    print()
    print("delta (exact - even), saved $/session:")
    print(f"{'T':>9} " + " ".join(f"a={a:.0%}".rjust(9) for a in ALPHAS))
    for T in SWEEP:
        print(f"{T:>9,} " + " ".join(
            f"{(exact[(T,a)]['saved']-res[(T,a)]['saved'])/ns:>9.4f}" for a in ALPHAS))


if __name__ == "__main__":
    main()
