"""Cost of splitting SWE-chat sessions into chunks -- EXACT model, SUMMARY overhead.

Scope: Claude Code sessions ONLY, from the `sessions` table.

WHERE EVERYTHING LIVES (map of this file)
-----------------------------------------
  SECTION A  config          -- dataset path, thresholds, alphas, sweep
  SECTION B  data loading     -- load_claude_code(): pull the 5 token columns
  SECTION C  analysis         -- analyze()          (exact split, all savings metrics)
                                 per_chunk_costs()  (price every individual chunk)
  SECTION D  reporting        -- pctile(), print_chunk_distribution(), main()

THE MATH MODEL ITSELF LIVES IN  chunk_split_model.py  (import `csm` below) and is
fully reusable -- for SEMANTIC splitting, feed your own chunk-height list to
csm.cache_read_after_split / csm.calls_per_chunk / csm.summary_overhead_cost.

MODEL (exact geometry): a session's context ramps ~linearly to a peak P over N API
calls; cache-read = area = N*P/2 (~70% of the bill). Capping context at threshold T
resets into k = ceil(P/T) chunks; each chunk re-reads only its own <=T content and
the LAST chunk climbs only its remainder R = P-(k-1)*T (partial). Every new chunk
(2..k) is seeded with a summary of size s = alpha*T, billed at REAL rates
(OUTPUT to make it + CACHE-WRITE to load it + CACHE-READ pedestal) -- see
chunk_split_model.summary_overhead_cost. Dollar figures are token-based ESTIMATES.
"""
import glob
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import chunk_split_model as csm

# ============================================================================
# SECTION A -- CONFIG
# ============================================================================

SESSIONS_PATH = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--SALT-NLP--SWE-chat/snapshots/*/sessions.parquet"))[0]

CLAUDE_CODE_AGENTS = ["Claude Code", "claude-code"]

RATES = csm.DEFAULT_RATES                           # $4.75/$23.75, write 1.25x, read 0.10x

ALPHAS = [0.05, 0.10, 0.15, 0.20]                   # summary size as fraction of T
THRESHOLDS = [100_000, 150_000]                     # detailed tables
SWEEP = list(range(100_000, 150_001, 10_000))       # 100k -> 150k sweep, 10k steps


# ============================================================================
# SECTION B -- DATA LOADING
# ============================================================================


def load_claude_code():
    """Return the five per-session token/count columns for Claude Code sessions:
    (input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
     api_call_count) -- each a list, nulls coerced to 0.
    """
    table = pq.read_table(SESSIONS_PATH, columns=[
        "agent", "input_tokens", "output_tokens",
        "cache_creation_tokens", "cache_read_tokens", "api_call_count"])
    is_claude = pc.is_in(table.column("agent"), value_set=pa.array(CLAUDE_CODE_AGENTS))
    table = table.filter(is_claude)

    def col(name):
        return [x or 0 for x in table.column(name).to_pylist()]

    return (col("input_tokens"), col("output_tokens"),
            col("cache_creation_tokens"), col("cache_read_tokens"),
            col("api_call_count"))


# ============================================================================
# SECTION C -- ANALYSIS  (loop sessions x thresholds x alphas; reuse csm math)
# per session: input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls
# per split:   threshold T, alpha, summary_tok s, peak P, chunks k, calls/chunk m
# ============================================================================


def analyze(input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls):
    """EXACT chunk-split model: each chunk billed as a fresh mini-session, the last
    chunk climbing only its remainder R = P-(k-1)*T. Returns
    (total_full_cost, total_cache_read_cost, results[(T, alpha)] -> metrics dict).
    """
    sessions = list(zip(input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls))
    total_full = sum(csm.full_session_cost(it, ot, cw, cr, RATES) for it, ot, cw, cr, _ in sessions)
    total_cr_cost = sum(cr * RATES.input_per_mtok * RATES.cache_read_mult / 1e6
                        for *_, cr, _ in sessions)

    results = {}
    for threshold in sorted(set(THRESHOLDS) | set(SWEEP)):
        for alpha in ALPHAS:
            summary_tok = csm.summary_tokens(threshold, alpha)
            split_cost = 0.0
            oh_total = oh_gen = oh_write = oh_reread = 0.0
            n_split = 0
            sum_chunks = 0
            for (it, ot, cw, cr, api) in sessions:
                # can't model (no cache-read / no calls) OR peak below T -> bill as-is
                if api <= 0 or cr <= 0:
                    split_cost += csm.full_session_cost(it, ot, cw, cr, RATES)
                    sum_chunks += 1
                    continue
                peak = csm.peak_context_tokens(cr, api)
                if peak <= threshold:
                    split_cost += csm.full_session_cost(it, ot, cw, cr, RATES)
                    sum_chunks += 1
                    continue
                heights = csm.uniform_chunk_heights(peak, threshold)   # [T]*(k-1) + [R]
                k = len(heights)
                chunk_calls = csm.calls_per_chunk(heights, api, peak)   # [m_full]*(k-1)+[m_last]
                cr_split = csm.cache_read_after_split(heights, api, peak)
                base = csm.base_cost_after_split(it, ot, cw, cr_split, RATES)
                # new chunks = chunks 2..k, each seeded with a summary
                oh, g, w, rr = csm.summary_overhead_cost(chunk_calls[1:], summary_tok, RATES)
                split_cost += base + oh
                oh_total += oh
                oh_gen += g
                oh_write += w
                oh_reread += rr
                sum_chunks += k
                n_split += 1
            results[(threshold, alpha)] = {
                "split": split_cost,
                "saved": total_full - split_cost,
                "oh_total": oh_total,
                "oh_gen": oh_gen,
                "oh_write": oh_write,
                "oh_reread": oh_reread,
                "n_split": n_split,
                "avg_chunks": sum_chunks / len(sessions),
            }
    return total_full, total_cr_cost, results


def per_chunk_costs(input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls,
                    threshold, alpha):
    """Cost of EVERY individual chunk, billed as a fresh mini-session (EXACT geometry).

    Each chunk owns its call-share of input/output/cache-write, plus its own
    cache-read triangle, plus (chunks 2..k) the summary overhead. Returns a list
    of (chunk_cost, chunk_calls, is_first_chunk).
    """
    summary_tok = csm.summary_tokens(threshold, alpha)
    out = []
    for it, ot, cw, cr, api in zip(input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls):
        if api <= 0 or cr <= 0:
            out.append((csm.full_session_cost(it, ot, cw, cr, RATES), api, True))
            continue
        peak = csm.peak_context_tokens(cr, api)
        if peak <= threshold:
            out.append((csm.full_session_cost(it, ot, cw, cr, RATES), api, True))
            continue
        heights = csm.uniform_chunk_heights(peak, threshold)
        chunk_calls = csm.calls_per_chunk(heights, api, peak)
        for j in range(len(heights)):
            height_j = heights[j]
            calls_j = chunk_calls[j]
            call_fraction = calls_j / api
            base = csm.fresh_chunk_base_cost(it, ot, cw, call_fraction, RATES)
            own_cr = csm.chunk_own_cache_read_cost(calls_j, height_j, RATES)
            overhead = 0.0 if j == 0 else csm.summary_overhead_cost([calls_j], summary_tok, RATES)[0]
            out.append((base + own_cr + overhead, calls_j, j == 0))
    return out


# ============================================================================
# SECTION D -- REPORTING
# ============================================================================


def pctile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(p / 100 * len(xs)))
    return xs[i]


def print_chunk_distribution(input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls):
    print("=== PER-CHUNK cost distribution -- chunks are NOT a fixed price ===")
    print("(every individual chunk across all sessions; alpha=10%)")
    print(f"{'T':>9} {'#chunks':>9}{'mean$':>9}{'p10':>8}{'median':>8}{'p90':>8}"
          f"{'max':>9}{'mean m(calls)':>15}")
    for threshold in THRESHOLDS:
        chunks = per_chunk_costs(input_tok, output_tok, cache_write_tok, cache_read_tok,
                                 api_calls, threshold, 0.10)
        costs = [c for c, _, _ in chunks]
        calls = [m for _, m, _ in chunks]
        mean = sum(costs) / len(costs)
        mean_calls = sum(calls) / len(calls)
        print(f"{threshold:>9,} {len(costs):>9,}{mean:>9.4f}{pctile(costs,10):>8.4f}"
              f"{pctile(costs,50):>8.4f}{pctile(costs,90):>8.4f}{max(costs):>9.2f}"
              f"{mean_calls:>15.1f}")
    print()


def main():
    it, ot, cw, cr, api = load_claude_code()
    n_sessions = len(it)
    total_full, total_cr_cost, res = analyze(it, ot, cw, cr, api)

    print(f"=== SWE-chat split-cost -- EXACT summary-overhead model -- Claude Code only, "
          f"{n_sessions:,} sessions ===")
    print(f"rates: ${RATES.input_per_mtok}/${RATES.output_per_mtok} per MTok "
          f"(write {RATES.cache_write_mult}x, read {RATES.cache_read_mult}x)")
    print(f"exact: each chunk a fresh mini-session; last chunk partial (height R = P-(k-1)*T)")
    print(f"overhead per new chunk = OUTPUT(s) + CACHE-WRITE(s) + CACHE-READ(s*(m-1)), s=alpha*T\n")
    print(f"cost AS-IS (full): ${total_full:,.0f} total | ${total_full/n_sessions:.4f}/session | 100%")
    print(f"  of which cache-read tax: ${total_cr_cost:,.0f} | ${total_cr_cost/n_sessions:.4f}/sess | "
          f"{100*total_cr_cost/total_full:.1f}%\n")

    # BIG TABLE: savings per alpha at the detailed thresholds
    for alpha in ALPHAS:
        print(f"--- overhead alpha = {alpha:.0%} of T (summary = {alpha:.0%}*T tokens) ---")
        print(f"{'T':>9} {'saved$tot':>10}{'saved/ss':>9}{'saved%':>8}  "
              f"{'OH$tot':>9}{'chunks':>8}")
        for threshold in THRESHOLDS:
            r = res[(threshold, alpha)]
            sv, oh = r["saved"], r["oh_total"]
            print(f"{threshold:>9,} {sv:>10,.0f}{sv/n_sessions:>9.4f}{100*sv/total_full:>7.1f}%  "
                  f"{oh:>9,.0f}{r['avg_chunks']:>8.2f}")
        print()

    # overhead component breakdown at the named thresholds (alpha=10%)
    print("=== overhead breakdown (alpha=10%): where the summary $ goes ===")
    print(f"{'T':>9} {'OH total':>10}{'gen(out)':>10}{'write':>10}{'reread(cr)':>12}")
    for threshold in THRESHOLDS:
        r = res[(threshold, 0.10)]
        print(f"{threshold:>9,} {r['oh_total']:>10,.0f}{r['oh_gen']:>10,.0f}"
              f"{r['oh_write']:>10,.0f}{r['oh_reread']:>12,.0f}")
    print()

    # SWEEP: saved % of full across 100k -> 150k
    print("=== SWEEP: saved % of full -- threshold (rows) x alpha (cols) ===")
    print(f"{'T':>9} " + " ".join(f"a={a:.0%}".rjust(9) for a in ALPHAS) + f"{'chunks':>9}")
    for threshold in SWEEP:
        print(f"{threshold:>9,} " + " ".join(
            f"{100*res[(threshold, a)]['saved']/total_full:>9.2f}" for a in ALPHAS)
            + f"{res[(threshold, ALPHAS[0])]['avg_chunks']:>9.2f}")
    print()

    # optimal threshold per alpha (savings rise as T shrinks -> smallest wins)
    print("optimal threshold (MAX net savings) per alpha:")
    for alpha in ALPHAS:
        best = max(SWEEP, key=lambda T: res[(T, alpha)]["saved"])
        r = res[(best, alpha)]
        print(f"  alpha={alpha:.0%}: T={best:,} -> ${r['saved']:,.0f} total | "
              f"${r['saved']/n_sessions:.4f}/session | {100*r['saved']/total_full:.1f}% of full")
    print()

    print_chunk_distribution(it, ot, cw, cr, api)


if __name__ == "__main__":
    main()
