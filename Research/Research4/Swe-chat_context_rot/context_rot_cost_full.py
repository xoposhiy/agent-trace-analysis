"""Context-rot cost analysis from local conversations.parquet, ALL sessions.

Key facts learned from the data:
  - Each ROW is an event; tokens live on ~371k "billed" API-call rows
    (role assistant/tool_use). 86% of rows are zero-token metadata/tool_result.
  - So "cost per turn" = cost per BILLED API call (a row with tokens > 0).
  - A call is HEAVY when its processed prompt (input + cache_write + cache_read)
    exceeds HEAVY_THRESHOLD tokens.
  - Context tax = cache-read $ / total $ (money spent re-reading old context).
Dollar figures are token-based ESTIMATES at per-model Anthropic rates.
"""
import glob, os
from collections import defaultdict

import pyarrow.parquet as pq

PATH = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--SALT-NLP--SWE-chat/snapshots/*/conversations.parquet"))[0]
HEAVY_THRESHOLD = 100_000

# per-MTok (input, output); cache write = 1.25x input, cache read = 0.10x input.
# Order matters: most-specific first. Modern Opus (4.5-4.8) = $5/$25; Opus 4.0/4.1 = $15/$75.
PRICING = [
    ("claude-fable-5", (10.0, 50.0)),
    ("opus-4-8", (5.0, 25.0)), ("opus-4-7", (5.0, 25.0)),
    ("opus-4-6", (5.0, 25.0)), ("opus-4-5", (5.0, 25.0)),
    ("opus-4-1", (15.0, 75.0)), ("opus-4-20", (15.0, 75.0)),  # 4.0 (claude-opus-4-20250514)
    ("3-opus", (15.0, 75.0)), ("opus", (15.0, 75.0)),
    ("sonnet-4", (3.0, 15.0)), ("3-5-sonnet", (3.0, 15.0)),
    ("3-7-sonnet", (3.0, 15.0)), ("sonnet", (3.0, 15.0)),
    ("haiku-4", (1.0, 5.0)), ("3-5-haiku", (0.80, 4.0)),
    ("3-haiku", (0.25, 1.25)), ("haiku", (1.0, 5.0)),
]
DEFAULT = (3.0, 15.0)  # None / non-Claude (glm, gpt, synthetic...) — 3.6% of tokens

def rate_for(m):
    m = (m or "").lower()
    for k, r in PRICING:
        if k in m:
            return r
    return DEFAULT

COLS = ["session_id", "turn_number", "model", "input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens"]

CTX_BINS = [(0, 10_000), (10_000, 25_000), (25_000, 50_000), (50_000, 100_000),
            (100_000, 200_000), (200_000, 10**12)]

def main():
    t = pq.read_table(PATH, columns=COLS)
    n = t.num_rows
    d = t.to_pydict(); del t
    sid = d["session_id"]; tn = d["turn_number"]; mdl = d["model"]
    it = d["input_tokens"]; ot = d["output_tokens"]
    cw = d["cache_creation_input_tokens"]; cr = d["cache_read_input_tokens"]

    sessions = defaultdict(list)   # sid -> list of billed calls
    for i in range(n):
        inp = it[i] or 0; out = ot[i] or 0; cwt = cw[i] or 0; crt = cr[i] or 0
        if inp + out + cwt + crt == 0:
            continue  # zero-token event (metadata/tool_result) — not a billed call
        ir, orr = rate_for(mdl[i])
        cost_cr = crt / 1e6 * ir * 0.10
        total = (inp/1e6*ir) + (out/1e6*orr) + (cwt/1e6*ir*1.25) + cost_cr
        sessions[sid[i]].append((tn[i] or 0, total, cost_cr, inp + cwt + crt))

    sess_costs = []
    all_calls = []; light = []; heavy = []
    tax_cr = 0.0; tax_tot = 0.0
    became_heavy = []
    ctx_bin = {b: [] for b in CTX_BINS}

    for s, calls in sessions.items():
        calls.sort(key=lambda x: x[0])
        sess_costs.append(sum(c[1] for c in calls))
        first_heavy = None
        for i, (_, c, ccr, ctx) in enumerate(calls):
            all_calls.append(c)
            (heavy if ctx > HEAVY_THRESHOLD else light).append(c)
            tax_cr += ccr; tax_tot += c
            for lo, hi in CTX_BINS:
                if lo <= ctx < hi:
                    ctx_bin[(lo, hi)].append(c); break
            if first_heavy is None and ctx > HEAVY_THRESHOLD:
                first_heavy = i
        if first_heavy is not None:
            before = sum(x[1] for x in calls[:first_heavy])
            after = sum(x[1] for x in calls[first_heavy:])
            became_heavy.append((before, after, first_heavy, len(calls)-first_heavy))

    def avg(x): return sum(x)/len(x) if x else 0.0
    def med(x): return sorted(x)[len(x)//2] if x else 0.0
    def pct(x, q): return sorted(x)[min(int(q*len(x)), len(x)-1)] if x else 0.0

    print(f"=== context-rot cost — ALL {len(sessions):,} sessions, "
          f"{len(all_calls):,} billed API calls ===")
    print(f"heavy call = prompt > {HEAVY_THRESHOLD:,} tokens\n")
    print(f"avg  cost / session      : ${avg(sess_costs):.4f}")
    print(f"median cost / session    : ${med(sess_costs):.4f}")
    print(f"p90  cost / session      : ${pct(sess_costs,0.9):.4f}")
    print(f"max  cost / session      : ${max(sess_costs):.2f}")
    print(f"avg  cost / call (all)   : ${avg(all_calls):.4f}")
    print(f"avg  cost / LIGHT call   : ${avg(light):.4f}  (n={len(light):,})")
    print(f"avg  cost / HEAVY call   : ${avg(heavy):.4f}  (n={len(heavy):,})")
    if light and heavy:
        print(f"   -> a heavy call costs {avg(heavy)/avg(light):.0f}x a light call")
    print(f"\ncontext tax (cache-read $ / total $): {100*tax_cr/tax_tot:.1f}%")
    print(f"\nsessions that go heavy: {len(became_heavy):,}/{len(sessions):,} "
          f"({100*len(became_heavy)/len(sessions):.0f}%)")
    if became_heavy:
        b = avg([x[0] for x in became_heavy]); a = avg([x[1] for x in became_heavy])
        nb = avg([x[2] for x in became_heavy]); na = avg([x[3] for x in became_heavy])
        print(f"  BEFORE heavy: ${b:.4f} over {nb:.0f} calls (${b/nb if nb else 0:.4f}/call)")
        print(f"  AFTER  heavy: ${a:.4f} over {na:.0f} calls (${a/na if na else 0:.4f}/call)")
        print(f"  -> {100*a/(a+b):.0f}% of a heavy session's cost lands AFTER it goes heavy")
    print(f"\nrot curve — avg $/call by context size of that call:")
    for (lo, hi) in CTX_BINS:
        xs = ctx_bin[(lo, hi)]
        label = f"{lo//1000}k-{hi//1000}k" if hi < 10**11 else f">{lo//1000}k"
        print(f"  ctx {label:>9}: ${avg(xs):.4f}  (n={len(xs):,})")


if __name__ == "__main__":
    main()
