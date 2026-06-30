"""Estimate the *money* cost of context rot from the SWE-chat dataset.

Question studied:
  - average $ cost per session
  - average $ cost per turn, and how it grows with turn position (the "rot curve")
  - cost BEFORE vs AFTER a session "goes heavy" (heavy = a turn whose full prompt
    exceeds HEAVY_THRESHOLD tokens), and how much of total spend is the
    "context tax" (cache-read of accumulated context).

Heavy definition (user choice): context-size threshold.
  A turn is "heavy" when its processed prompt
      ctx = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
  exceeds HEAVY_THRESHOLD (default 100_000 ≈ 50% of a 200k window).

Scope (user choice): a SAMPLE of the first N complete sessions seen while
streaming, so we validate the method cheaply before a full run.

Cost model: tokens x per-model Anthropic rates, 4 buckets each:
  input (full), output, cache write (1.25x input), cache read (0.10x input).
Dollar figures are token-based ESTIMATES, not billed amounts.
"""
from collections import defaultdict

from datasets import load_dataset

SAMPLE_SESSIONS = 300
HEAVY_THRESHOLD = 100_000
MAX_ROWS_SCAN = 2_000_000  # safety cap on the stream

# Per-MTok USD rates: (input, output). Cache write = 1.25x input, read = 0.10x input.
# Matched by substring against the lower-cased `model` field; first hit wins, so
# order most-specific first. Rates per Anthropic public pricing.
PRICING = [
    ("claude-fable-5",   (10.0, 50.0)),
    ("opus-4",           (15.0, 75.0)),  # Opus 4.0/4.1 era (SWE-chat vintage)
    ("3-opus",           (15.0, 75.0)),
    ("opus",             (15.0, 75.0)),
    ("3-5-sonnet",       (3.0, 15.0)),
    ("3-7-sonnet",       (3.0, 15.0)),
    ("sonnet-4",         (3.0, 15.0)),
    ("sonnet",           (3.0, 15.0)),
    ("3-5-haiku",        (0.80, 4.0)),
    ("haiku-4",          (1.0, 5.0)),
    ("3-haiku",          (0.25, 1.25)),
    ("haiku",            (1.0, 5.0)),
]
DEFAULT_RATE = (3.0, 15.0)  # fall back to Sonnet-class if unknown


def rate_for(model: str):
    m = (model or "").lower()
    for key, r in PRICING:
        if key in m:
            return r
    return DEFAULT_RATE


def turn_cost(row):
    """USD cost of a single turn from its 4 token buckets."""
    inp = row["input_tokens"] or 0
    out = row["output_tokens"] or 0
    cw = row["cache_creation_input_tokens"] or 0
    cr = row["cache_read_input_tokens"] or 0
    in_rate, out_rate = rate_for(row["model"])
    cost_in = inp / 1e6 * in_rate
    cost_out = out / 1e6 * out_rate
    cost_cw = cw / 1e6 * (in_rate * 1.25)
    cost_cr = cr / 1e6 * (in_rate * 0.10)
    return {
        "total": cost_in + cost_out + cost_cw + cost_cr,
        "in": cost_in, "out": cost_out, "cw": cost_cw, "cr": cost_cr,
        "ctx": inp + cw + cr,  # prompt size processed this turn
    }


def main():
    ds = load_dataset("SALT-NLP/SWE-chat", "conversations", split="train",
                      streaming=True)

    sessions = defaultdict(list)  # session_id -> list of (turn_number, cost dict)
    order = []
    seen = set()
    scanned = 0
    for row in ds:
        scanned += 1
        sid = row["session_id"]
        if sid not in seen:
            if len(seen) >= SAMPLE_SESSIONS:
                # we already have our sample; stop once the current sessions end
                break
            seen.add(sid)
            order.append(sid)
        if sid in seen:
            c = turn_cost(row)
            c["turn"] = row["turn_number"]
            c["role"] = row["role"]
            sessions[sid].append(c)
        if scanned >= MAX_ROWS_SCAN:
            break

    # ---- aggregate ----
    sess_costs = []
    all_turn_costs = []
    light_turns, heavy_turns = [], []          # per-turn $ split by heavy flag
    became_heavy = []                          # per-session before/after split
    tax = {"cr": 0.0, "total": 0.0}            # context-tax share

    for sid, turns in sessions.items():
        turns.sort(key=lambda t: (t["turn"] if t["turn"] is not None else 0))
        total = sum(t["total"] for t in turns)
        sess_costs.append(total)
        for t in turns:
            all_turn_costs.append(t["total"])
            (heavy_turns if t["ctx"] > HEAVY_THRESHOLD else light_turns).append(t["total"])
            tax["cr"] += t["cr"]
            tax["total"] += t["total"]

        # before/after going heavy
        first_heavy = next((i for i, t in enumerate(turns)
                            if t["ctx"] > HEAVY_THRESHOLD), None)
        if first_heavy is not None:
            before = sum(t["total"] for t in turns[:first_heavy])
            after = sum(t["total"] for t in turns[first_heavy:])
            became_heavy.append({
                "sid": sid, "before": before, "after": after,
                "n_before": first_heavy, "n_after": len(turns) - first_heavy,
            })

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"=== SWE-chat context-rot cost (sample) ===")
    print(f"rows scanned: {scanned:,}   sessions: {len(sessions)}   "
          f"turns: {len(all_turn_costs):,}")
    print(f"heavy threshold: {HEAVY_THRESHOLD:,} prompt tokens/turn\n")

    print(f"avg cost / session:            ${avg(sess_costs):.4f}")
    print(f"median cost / session:         ${sorted(sess_costs)[len(sess_costs)//2]:.4f}")
    print(f"avg cost / turn (all):         ${avg(all_turn_costs):.4f}")
    print(f"avg cost / LIGHT turn (<{HEAVY_THRESHOLD//1000}k): ${avg(light_turns):.4f}  (n={len(light_turns):,})")
    print(f"avg cost / HEAVY turn (>{HEAVY_THRESHOLD//1000}k): ${avg(heavy_turns):.4f}  (n={len(heavy_turns):,})")
    if light_turns and heavy_turns:
        print(f"  -> a heavy turn costs {avg(heavy_turns)/avg(light_turns):.1f}x a light turn")

    print(f"\ncontext tax (cache-read $ as share of all spend): "
          f"{100*tax['cr']/tax['total']:.1f}%")

    print(f"\nsessions that went heavy: {len(became_heavy)}/{len(sessions)} "
          f"({100*len(became_heavy)/len(sessions):.0f}%)")
    if became_heavy:
        b = avg([x["before"] for x in became_heavy])
        a = avg([x["after"] for x in became_heavy])
        nb = avg([x["n_before"] for x in became_heavy])
        na = avg([x["n_after"] for x in became_heavy])
        print(f"  avg $ spent BEFORE going heavy: ${b:.4f}  over {nb:.1f} turns "
              f"(${b/nb if nb else 0:.4f}/turn)")
        print(f"  avg $ spent AFTER  going heavy: ${a:.4f}  over {na:.1f} turns "
              f"(${a/na if na else 0:.4f}/turn)")
        print(f"  -> {100*a/(a+b):.0f}% of a heavy session's cost is spent "
              f"after it goes heavy")

    # rot curve: avg cost by turn-position bucket
    print(f"\nrot curve — avg $/turn by position:")
    buckets = defaultdict(list)
    for turns in sessions.values():
        for i, t in enumerate(turns):
            buckets[min(i // 10, 9)].append(t["total"])
    for b in sorted(buckets):
        lo = b * 10
        print(f"  turns {lo:>3}-{lo+9:<3}: ${avg(buckets[b]):.4f}  (n={len(buckets[b]):,})")


if __name__ == "__main__":
    main()
