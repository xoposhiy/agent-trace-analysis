"""Cost of splitting SWE-chat sessions into chunks at a context threshold.

Scope: Claude Code sessions ONLY (agent in {"Claude Code","claude-code"}), from
the `sessions` table (the trustworthy source for absolute token/cost totals;
`conversations` undercounts API calls ~9x).

--------------------------------------------------------------------------------
THE MATH MODEL
--------------------------------------------------------------------------------
Per session the `sessions` table gives totals: IT (input/uncached), OT (output),
CW (cache-write), CR (cache-read), N (api_call_count).

Rates per MTok: r_in, r_out; cache-write = 1.25*r_in, cache-read = 0.10*r_in.

1) Cost "as they are" (full, WITH the quadratic rebilling baked into CR):
       C_full = r_in*IT + r_out*OT + 1.25*r_in*CW + 0.10*r_in*CR
   CR is the whole quadratic tax: every API call re-reads accumulated history.

2) Where the quadratic lives. Per-call processed context ctx_i ramps up as the
   conversation grows. Model it as a linear ramp to a peak P, so avg ctx = P/2:
       CR = sum_i ctx_i ~= N * P/2   =>   P = 2*CR/N      (self-calibrated)

3) Split at threshold T: each chunk is billed as a FRESH session, so the context
   ramp RESETS to ~0 every time it hits T -> a saw-tooth between 0 and T instead
   of climbing to P. Working the partial sums through (slope P/N kept constant):
       k        = ceil(P / T)                 # number of chunks
       CR_split = min(CR, N * T / 2)          # rebilling: quadratic -> LINEAR
   Reduction factor T/P. If a session never reaches T (P<T) -> k=1, no change.

4) Per-chunk overhead = re-establishing carried-over context in each new chunk.
   Flat cost, billed as a fraction of the threshold at the global avg $/token:
       O_cost = ALPHA * T * c_bar      (ALPHA=0.1)
   c_bar  = (total C_full across all sessions) / (total tokens across all)
   Applied to (k-1) chunks (first chunk = original start, pays nothing).

5) Split total and savings:
       C_split = r_in*IT + r_out*OT + 1.25*r_in*CW + 0.10*r_in*CR_split
                 + (k-1)*O_cost
       savings = 0.10*r_in*(CR - CR_split)   [rebilling saved]
                 - (k-1)*O_cost              [overhead paid]

Everything is computed straight from `sessions` aggregates -- no per-call data.
Dollar figures are token-based ESTIMATES at blended Anthropic rates.
"""
import glob
import math
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq

SESSIONS_PATH = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--SALT-NLP--SWE-chat/snapshots/*/sessions.parquet"))[0]

CLAUDE_CODE_AGENTS = ["Claude Code", "claude-code"]

# Blended per-MTok rates (sessions table has no model column; Claude-Code mix is
# mostly Opus 4.x). Token-weighted blend from the conversations table.
R_IN = 4.75
R_OUT = 23.75
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

ALPHAS = [0.05, 0.10, 0.15, 0.20]  # overhead = ALPHA * T tokens worth, per new chunk
THRESHOLDS = [100_000, 150_000]    # the two requested (detailed tables)
SWEEP = list(range(10_000, 300_001, 10_000))  # finer/wider break-even sweep


def full_cost(it, ot, cw, cr):
    return (it * R_IN + ot * R_OUT
            + cw * R_IN * CACHE_WRITE_MULT
            + cr * R_IN * CACHE_READ_MULT) / 1e6


def load_claude_code():
    """Return per-session lists (it, ot, cw, cr, n) for Claude Code sessions."""
    t = pq.read_table(SESSIONS_PATH, columns=[
        "agent", "input_tokens", "output_tokens",
        "cache_creation_tokens", "cache_read_tokens", "api_call_count"])
    import pyarrow as pa
    mask = pc.is_in(t.column("agent"), value_set=pa.array(CLAUDE_CODE_AGENTS))
    t = t.filter(mask)

    def col(c):
        return [x or 0 for x in t.column(c).to_pylist()]

    return (col("input_tokens"), col("output_tokens"),
            col("cache_creation_tokens"), col("cache_read_tokens"),
            col("api_call_count"))


def analyze(it, ot, cw, cr, n):
    """Per-threshold geometry, INDEPENDENT of the overhead % (alpha).

    Overhead = alpha * T * c_bar * (k-1), so it factors cleanly out of alpha.
    For each T we return the overhead-free split cost and sum_km1 = Σ(k-1), from
    which any alpha's total is split_no_oh + alpha*T*c_bar*sum_km1.
    """
    sessions = list(zip(it, ot, cw, cr, n))
    full = [full_cost(a, b, c, d) for a, b, c, d, _ in sessions]
    total_full = sum(full)
    total_cr_cost = sum(d * R_IN * CACHE_READ_MULT / 1e6 for *_, d, _ in sessions)

    results = {}
    for T in sorted(set(THRESHOLDS) | set(SWEEP)):
        split_no_oh = 0.0   # split cost WITHOUT overhead
        sum_km1 = 0         # Σ (k-1) over sessions -> overhead multiplier
        n_split = 0
        sum_chunks = 0
        for (a, b, c, d, nn) in sessions:
            if nn <= 0 or d <= 0:
                split_no_oh += full_cost(a, b, c, d)
                sum_chunks += 1
                continue
            P = 2.0 * d / nn                 # peak context
            k = 1 if P <= T else math.ceil(P / T)
            cr_split = min(d, nn * T / 2.0)
            base = (a * R_IN + b * R_OUT + c * R_IN * CACHE_WRITE_MULT) / 1e6
            split_no_oh += base + cr_split * R_IN * CACHE_READ_MULT / 1e6
            sum_km1 += (k - 1)
            sum_chunks += k
            if k > 1:
                n_split += 1
        results[T] = {
            "split_no_oh": split_no_oh,
            "sum_km1": sum_km1,
            "n_split": n_split,
            "avg_chunks": sum_chunks / len(sessions),
        }
    return total_full, total_cr_cost, results


def simulate_per_bucket(it, ot, cw, cr, n, T, alpha, c_bar):
    """Explicit PER-CALL simulation, broken out PER BUCKET.

    For each session we distribute its recorded tokens across its N calls and
    walk call-by-call, accumulating context. When the running (within-chunk)
    context crosses T we RESET it to 0 -> later calls stop re-reading the earlier
    chunk (the explicit "subtract earlier context-window re-reads"). Input,
    output and cache-write buckets are unchanged by splitting (same unique work);
    ALL the savings land in the cache-read bucket, against which we net the
    per-new-chunk overhead.

    Returns full and split cost split into the four buckets (+ overhead).
    Validates the closed form: Σ cache_read_split ≈ N·T/2.
    """
    full = {"in": 0.0, "out": 0.0, "cw": 0.0, "cr": 0.0}
    split = {"in": 0.0, "out": 0.0, "cw": 0.0, "cr": 0.0, "overhead": 0.0}
    cr_full_tok = 0.0
    cr_split_tok = 0.0

    for a, b, c, d, nn in zip(it, ot, cw, cr, n):
        # buckets unchanged by splitting
        c_in = a * R_IN / 1e6
        c_out = b * R_OUT / 1e6
        c_cw = c * R_IN * CACHE_WRITE_MULT / 1e6
        for dct in (full, split):
            dct["in"] += c_in
            dct["out"] += c_out
            dct["cw"] += c_cw

        if nn <= 1 or d <= 0:                       # nothing to split
            full["cr"] += d * R_IN * CACHE_READ_MULT / 1e6
            split["cr"] += d * R_IN * CACHE_READ_MULT / 1e6
            cr_full_tok += d
            cr_split_tok += d
            continue

        # per-call context growth so that Σ_i (i-1)·step = recorded CR
        step = 2.0 * d / (nn * (nn - 1))            # tokens added to context/call
        ctx_full = 0.0                              # running context (never resets)
        ctx_chunk = 0.0                             # within-chunk context (resets)
        sess_cr_full = 0.0
        sess_cr_split = 0.0
        resets = 0
        for _ in range(nn):
            sess_cr_full += ctx_full                # full: re-read EVERYTHING so far
            sess_cr_split += ctx_chunk              # split: re-read only this chunk
            ctx_full += step
            ctx_chunk += step
            if ctx_chunk >= T:                      # crossed threshold -> new chunk
                ctx_chunk = 0.0                     # SUBTRACT earlier-chunk re-reads
                resets += 1
        full["cr"] += sess_cr_full * R_IN * CACHE_READ_MULT / 1e6
        split["cr"] += sess_cr_split * R_IN * CACHE_READ_MULT / 1e6
        split["overhead"] += resets * alpha * T * c_bar  # k-1 = resets new chunks
        cr_full_tok += sess_cr_full
        cr_split_tok += sess_cr_split

    return full, split, cr_full_tok, cr_split_tok


def print_per_bucket(it, ot, cw, cr, n, c_bar, alpha=0.10):
    ns = len(it)
    print(f"\n=== PER-BUCKET cost (explicit per-call simulation, alpha={alpha:.0%}) ===")
    print(f"(each $ shown as: total | /session | % of that column's TOTAL)")
    for T in THRESHOLDS:
        full, split, crf, crs = simulate_per_bucket(it, ot, cw, cr, n, T, alpha, c_bar)
        tf = sum(full.values())
        ts = sum(split.values())
        sv_tot = tf - ts
        print(f"\n--- T = {T:,} ---")
        print(f"{'bucket':<12}"
              f"{'full $':>10}{'/sess':>8}{'%':>6}   "
              f"{'split $':>10}{'/sess':>8}{'%':>6}   "
              f"{'saved $':>10}{'/sess':>8}{'%':>6}")
        rows = [("in", "input"), ("out", "output"),
                ("cw", "cache-write"), ("cr", "cache-read")]
        for key, label in rows:
            f_, s_ = full[key], split[key]
            sv = f_ - s_
            print(f"{label:<12}"
                  f"{f_:>10,.0f}{f_/ns:>8.4f}{100*f_/tf:>5.0f}%   "
                  f"{s_:>10,.0f}{s_/ns:>8.4f}{100*s_/ts:>5.0f}%   "
                  f"{sv:>10,.0f}{sv/ns:>8.4f}{(100*sv/sv_tot if sv_tot else 0):>5.0f}%")
        oh = split["overhead"]
        print(f"{'overhead':<12}"
              f"{0:>10,.0f}{0:>8.4f}{0:>5.0f}%   "
              f"{oh:>10,.0f}{oh/ns:>8.4f}{100*oh/ts:>5.0f}%   "
              f"{-oh:>10,.0f}{-oh/ns:>8.4f}{(100*-oh/sv_tot if sv_tot else 0):>5.0f}%")
        print(f"{'TOTAL':<12}"
              f"{tf:>10,.0f}{tf/ns:>8.4f}{100:>5.0f}%   "
              f"{ts:>10,.0f}{ts/ns:>8.4f}{100:>5.0f}%   "
              f"{sv_tot:>10,.0f}{sv_tot/ns:>8.4f}{100*sv_tot/tf:>5.0f}%")
        print(f"  cache-read tokens: full {crf/1e6:,.0f}M -> split {crs/1e6:,.0f}M "
              f"(closed form N·T/2 check: {sum(n)*T/2/1e6:,.0f}M)")


def main():
    it, ot, cw, cr, n = load_claude_code()
    ns = len(it)

    # global effective $/token (c_bar) -- needs full costs across all sessions
    total_full0 = sum(full_cost(a, b, c, d) for a, b, c, d in zip(it, ot, cw, cr))
    total_tokens = sum(it) + sum(ot) + sum(cw) + sum(cr)
    c_bar = total_full0 / total_tokens      # $ per token

    total_full, total_cr_cost, res = analyze(it, ot, cw, cr, n)

    def overhead(T, alpha):
        return alpha * T * c_bar * res[T]["sum_km1"]

    def saved(T, alpha):
        return total_full - (res[T]["split_no_oh"] + overhead(T, alpha))

    print(f"=== SWE-chat split-cost model -- Claude Code only, {ns:,} sessions ===")
    print(f"rates: ${R_IN}/${R_OUT} per MTok (write {CACHE_WRITE_MULT}x, read {CACHE_READ_MULT}x)")
    print(f"global avg cost/token c_bar = ${c_bar*1e6:.4f}/MTok  "
          f"(overhead per new chunk = alpha*T*c_bar)\n")
    print(f"cost AS-IS (full, with quadratic rebilling): "
          f"${total_full:,.0f} total | ${total_full/ns:.4f}/session | 100%")
    print(f"  of which cache-read 'context tax':         "
          f"${total_cr_cost:,.0f} total | ${total_cr_cost/ns:.4f}/session | "
          f"{100*total_cr_cost/total_full:.1f}%\n")

    # ---- detailed per-threshold tables, one per alpha ----
    # each $ shown three ways: total | per-session | % of full
    for alpha in ALPHAS:
        print(f"--- overhead alpha = {alpha:.0%} of T ---")
        print(f"{'T':>8} {'split$tot':>10}{'split/ss':>9}{'split%':>7}  "
              f"{'saved$tot':>10}{'saved/ss':>9}{'saved%':>7}  "
              f"{'OH$tot':>8}{'OH/ss':>8}  {'%split':>7}{'chunks':>8}")
        for T in THRESHOLDS:
            r = res[T]
            sv = saved(T, alpha)
            sp = r["split_no_oh"] + overhead(T, alpha)
            oh = overhead(T, alpha)
            print(f"{T:>8,} {sp:>10,.0f}{sp/ns:>9.4f}{100*sp/total_full:>6.1f}%  "
                  f"{sv:>10,.0f}{sv/ns:>9.4f}{100*sv/total_full:>6.1f}%  "
                  f"{oh:>8,.0f}{oh/ns:>8.4f}  "
                  f"{100*r['n_split']/ns:>6.0f}%{r['avg_chunks']:>8.2f}")
        print()

    # ---- comparison matrices: saved by threshold (rows) x alpha (cols) ----
    def matrix(title, cell):
        print(f"=== COMPARISON ({title}): threshold (rows) x overhead alpha (cols) ===")
        print(f"{'T':>8} " + " ".join(f"a={a:.0%}".rjust(10) for a in ALPHAS)
              + f"{'chunks':>9}")
        for T in SWEEP:
            print(f"{T:>8,} " + " ".join(f"{cell(T, a):>10.4f}" for a in ALPHAS)
                  + f"{res[T]['avg_chunks']:>9.2f}")
        print()

    matrix("saved $/SESSION", lambda T, a: saved(T, a) / ns)
    matrix("saved $ TOTAL (all sessions)", lambda T, a: saved(T, a))
    matrix("saved % of full", lambda T, a: 100 * saved(T, a) / total_full)

    # ---- best threshold per alpha ----
    print(f"optimal threshold (max net savings) per alpha:")
    for alpha in ALPHAS:
        best = max(SWEEP, key=lambda T: saved(T, alpha))
        sv = saved(best, alpha)
        print(f"  alpha={alpha:.0%}: T={best:,} -> ${sv:,.0f} total | "
              f"${sv/ns:.4f}/session | {100*sv/total_full:.1f}% of full")

    # per-bucket breakdown via explicit per-call simulation
    print_per_bucket(it, ot, cw, cr, n, c_bar, alpha=0.10)


if __name__ == "__main__":
    main()
