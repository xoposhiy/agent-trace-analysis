"""
main.py — Context-coupling & session-splitting analysis on the SWE-chat dataset.

FOCUS OF THIS VERSION:
    Detect the MOMENT a user should have started a NEW session instead of
    continuing in the same one, and measure how much old context was
    dragged along past that moment.

RESULT: detected via TIME GAPS between user messages (Section F below).
A real elapsed-time gap is an objective behavioral fact, independent of
what the work is about. ~35% of long sessions have a 4+ hour gap between
user messages — a natural point where a fresh session would plausibly
have made more sense than continuing.

(Word-overlap, file-based, and issue-tracker approaches were also tried
and did not produce a reliable signal — see the accompanying report for
that history. They are left out of this file to keep it to what works.)

SECTIONS (run all, or comment out what you don't need in main()):
    A — shared setup (load data, long-session filter, transcript reader)
    B — Step 1+2: how many long sessions, exploration vs editing
    C — Step 3: tool categories (adds "execution")
    D — Step 4: token weight per category (exploration = biggest)
    E — Step 5: exploration cost + Viewer-style saving estimate
    F — task-switch detection via TIME GAP

Requirements: pip install datasets huggingface_hub
The dataset is gated: accept terms on huggingface.co and run `hf auth login` first.
"""

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import Counter
from datetime import datetime
import json

REPO = "SALT-NLP/SWE-chat"
LONG_SESSION_MIN_TURNS = 30

# Opus 4.5 pricing ($ per token) — used by Step 5 and Section F's $ savings
BASE_INPUT = 5.00
BASE_OUTPUT = 25.00
P_IN    = BASE_INPUT  / 1_000_000
P_OUT   = BASE_OUTPUT / 1_000_000
P_WRITE = BASE_INPUT * 1.25 / 1_000_000     # cache write
P_READ  = BASE_INPUT * 0.10 / 1_000_000     # cache read
VIEWER_FILTER_RATE = 0.60                   # SWE-Edit's measured filter rate
HANDOFF_TOKENS = 5_000                      # size of the "catch-up" summary at each split

# Section F — absolute, human-meaningful time-gap thresholds
GAP_THRESHOLDS = [15*60, 30*60, 60*60, 4*60*60, 24*60*60]
GAP_LABELS = ["15 min", "30 min", "1 hour", "4 hours", "24 hours"]


# ======================================================================
# Section A — shared setup
# ======================================================================

def load_tables():
    """Load the two tables we use. Cached locally after first download."""
    sessions = load_dataset(REPO, "sessions", split="train")
    logs = load_dataset(REPO, "session_logs", split="train")
    return sessions, logs


def is_long(s):
    """A 'long' session: has token data, >30 turns, and is Claude Code."""
    return (s["input_tokens"]
            and s["turn_count"] and s["turn_count"] > LONG_SESSION_MIN_TURNS
            and s["agent"] == "Claude Code")


def long_session_ids(sessions):
    return [s["session_id"] for s in sessions if is_long(s)]


def path_map(logs):
    return {l["session_id"]: l["transcript_path"] for l in logs}


def read_transcript(sid, paths):
    """Download (cached) and parse one transcript into a list of events."""
    p = hf_hub_download(REPO, paths[sid], repo_type="dataset")
    events = []
    with open(p) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue          # skip malformed lines — rare, safe to ignore
    return events


def categorize(tool):
    """Sort a tool name into one of four activity categories."""
    t = tool.lower()
    if any(k in t for k in ["read", "grep", "glob", "lsp", "search", "fetch"]):
        return "exploration"
    if any(k in t for k in ["edit", "write"]) and "todowrite" not in t:
        return "editing"
    if "bash" in t:
        return "execution"
    return "coordination"


def is_exploration(tool):
    t = tool.lower()
    return any(k in t for k in ["read", "grep", "glob", "lsp", "search", "fetch"])


def text_len(x):
    """Rough token estimate for any tool result: characters / 4."""
    if x is None:
        return 0
    if isinstance(x, str):
        return len(x)
    return len(json.dumps(x))


# ======================================================================
# Section B — Step 1 + 2: count long sessions, exploration vs editing
# ======================================================================

def step_1_2(sessions):
    print("\n" + "=" * 60)
    print("STEP 1+2 — long sessions, exploration vs editing")
    print("=" * 60)

    longs = [s for s in sessions if is_long(s)]
    print("long Claude Code sessions (>30 turns):", len(longs))

    total_research = total_action = usable = 0
    per_session = []
    for s in longs:
        if not s["research_count"] or not s["action_count"]:
            continue
        total_research += s["research_count"]
        total_action += s["action_count"]
        usable += 1
        tot = s["research_count"] + s["action_count"]
        per_session.append(s["research_count"] / tot)

    both = total_research + total_action
    print("sessions with both counts:", usable)
    print(f"  exploration (research): {total_research:,.0f}  ({100*total_research/both:.0f}%)")
    print(f"  editing (action):       {total_action:,.0f}  ({100*total_action/both:.0f}%)")
    print(f"  avg exploration share per session: {100*sum(per_session)/len(per_session):.0f}%")


# ======================================================================
# Section C — Step 3: tool categories across all long sessions
# ======================================================================

def step_3_categories(long_ids, paths):
    print("\n" + "=" * 60)
    print("STEP 3 — tool categories (call counts)")
    print("=" * 60)

    cat_counts = Counter()
    n_ok = n_skipped = 0
    for sid in long_ids:
        try:
            for e in read_transcript(sid, paths):
                if e.get("type") != "assistant":
                    continue
                for b in e["message"].get("content", []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        cat_counts[categorize(b.get("name", ""))] += 1
            n_ok += 1
        except Exception:
            n_skipped += 1

    total = sum(cat_counts.values()) or 1
    print("sessions read:", n_ok, "| skipped:", n_skipped, "| total tool calls:", f"{total:,}")
    for cat in ["exploration", "editing", "execution", "coordination"]:
        c = cat_counts[cat]
        print(f"  {cat:14s} {c:>8,}  ({100*c/total:.0f}%)")


# ======================================================================
# Section D — Step 4: token weight per category
# ======================================================================

def step_4_tokens(long_ids, paths):
    print("\n" + "=" * 60)
    print("STEP 4 — token weight per category")
    print("=" * 60)

    cat_calls = Counter()
    cat_tokens = Counter()
    matched = unmatched = n_ok = n_skipped = 0

    for sid in long_ids:
        try:
            id_to_cat = {}
            for e in read_transcript(sid, paths):
                msg = e.get("message", {})
                content = msg.get("content") if isinstance(msg, dict) else None
                if e.get("type") == "assistant" and isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            cat = categorize(b.get("name", ""))
                            id_to_cat[b.get("id")] = cat
                            cat_calls[cat] += 1
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            tid = b.get("tool_use_id")
                            if tid in id_to_cat:
                                cat = id_to_cat[tid]; matched += 1
                            else:
                                cat = "coordination"; unmatched += 1
                            cat_tokens[cat] += text_len(b.get("content")) / 4
            n_ok += 1
        except Exception:
            n_skipped += 1

    print("sessions read:", n_ok, "| skipped:", n_skipped)
    print(f"results matched: {matched:,} | unmatched: {unmatched:,}")
    tot_calls = sum(cat_calls.values()) or 1
    tot_tokens = sum(cat_tokens.values()) or 1
    print(f"\n{'category':14s} {'calls':>10s} {'call%':>7s} {'tokens(M)':>11s} {'token%':>8s}")
    for cat in ["exploration", "editing", "execution", "coordination"]:
        c = cat_calls[cat]; tk = cat_tokens[cat]
        print(f"{cat:14s} {c:>10,} {100*c/tot_calls:>6.0f}% {tk/1e6:>10.1f}M {100*tk/tot_tokens:>7.0f}%")


# ======================================================================
# Section E — Step 5: exploration cost + Viewer-style saving
# ======================================================================

def step_5_savings(long_ids, paths, filter_rate=VIEWER_FILTER_RATE):
    print("\n" + "=" * 60)
    print("STEP 5 — exploration cost & Viewer-style saving")
    print("=" * 60)

    total_expl_cost = total_all_cost = 0.0
    n_ok = n_skipped = 0
    for sid in long_ids:
        try:
            events = read_transcript(sid, paths)
            N = sum(1 for e in events if e.get("type") == "assistant")
            if N == 0:
                continue
            id_to_name = {}
            turn_no = 0
            for e in events:
                msg = e.get("message", {})
                content = msg.get("content") if isinstance(msg, dict) else None
                if e.get("type") == "assistant":
                    turn_no += 1
                    if isinstance(content, list):
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "tool_use":
                                id_to_name[b.get("id")] = (b.get("name", ""), turn_no)
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            name, t = id_to_name.get(b.get("tool_use_id"), (None, turn_no))
                            R = text_len(b.get("content")) / 4
                            remaining = max(N - t, 0)
                            life_cost = R * (P_WRITE + remaining * P_READ)
                            total_all_cost += life_cost
                            if name and is_exploration(name):
                                total_expl_cost += life_cost
            n_ok += 1
        except Exception:
            n_skipped += 1

    saving = total_expl_cost * filter_rate
    print("sessions read:", n_ok, "| skipped:", n_skipped)
    print(f"lifetime cost of ALL tool-result tokens:  ${total_all_cost:,.0f}")
    print(f"lifetime cost of EXPLORATION tokens:      ${total_expl_cost:,.0f}"
          f"  ({100*total_expl_cost/total_all_cost:.0f}% of tool-result cost)")
    print(f"estimated Viewer saving ({int(filter_rate*100)}% of expl): ${saving:,.0f}")


# ======================================================================
# Section F — task-switch detection via TIME GAP
# ======================================================================
#
# A real elapsed-time gap between one user message and the next is an
# objective behavioral fact, independent of what the work is about. A long
# gap is exactly the kind of moment where a fresh session would make sense
# — the person's own train of thought reset, so the old context is stale
# regardless of whether the next message is the same task or a new one.
#
# The threshold is an ABSOLUTE, human-meaningful gap size (not a relative
# percentile of this dataset — its gaps are almost all a few seconds, so
# a relative percentile would only mean "a few minutes", not a real break).

def parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def user_gaps_in_order(events):
    """
    Return (gaps, total_turns). gaps = [(turn_number, gap_seconds), ...],
    the elapsed time between one user message and the next, tagged with
    the assistant-turn count so far.
    """
    turn_no = 0
    last_ts = None
    gaps = []
    for e in events:
        if e.get("type") == "assistant":
            turn_no += 1
        if e.get("type") == "user":
            ts = parse_ts(e.get("timestamp"))
            if ts is None:
                continue
            if last_ts is not None:
                gap_seconds = (ts - last_ts).total_seconds()
                if gap_seconds > 0:              # ignore clock oddities
                    gaps.append((turn_no, gap_seconds))
            last_ts = ts
    return gaps, turn_no


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def section_f_time_gap(long_ids, paths):
    print("\n" + "=" * 60)
    print("SECTION F — task-switch detection via TIME GAP")
    print("=" * 60)

    per_session = {}
    all_gap_seconds = []
    n_skipped = 0

    for sid in long_ids:
        try:
            events = read_transcript(sid, paths)
            gaps, total_turns = user_gaps_in_order(events)
            if not gaps or total_turns == 0:
                continue
            per_session[sid] = (gaps, total_turns)
            all_gap_seconds.extend(g for _, g in gaps)
        except Exception:
            n_skipped += 1

    if not all_gap_seconds:
        print("no usable timestamp data found")
        return

    print(f"sessions with usable timestamps: {len(per_session)} | skipped: {n_skipped}")
    p50 = percentile(all_gap_seconds, 0.50)
    p95 = percentile(all_gap_seconds, 0.95)
    print(f"gap distribution across ALL sessions: median={p50:,.0f}s | 95th pct={p95:,.0f}s")

    for label, threshold in zip(GAP_LABELS, GAP_THRESHOLDS):
        n_with_gap = 0
        switch_positions = []
        for sid, (gaps, total_turns) in per_session.items():
            big_gaps = [(t, g) for t, g in gaps if g >= threshold]
            if not big_gaps:
                continue
            n_with_gap += 1
            turn, _ = big_gaps[0]
            switch_positions.append(min(turn / total_turns, 1.0))

        pct = 100 * n_with_gap / len(per_session)
        print(f"\n  threshold >= {label}: {n_with_gap} sessions ({pct:.0f}%) have such a gap")
        if switch_positions:
            avg_pos = 100 * sum(switch_positions) / len(switch_positions)
            buckets = [0, 0, 0, 0]
            for f in switch_positions:
                buckets[min(int(f * 4), 3)] += 1
            print(f"    avg position: {avg_pos:.0f}% | spread (0-25/25-50/50-75/75-100): {buckets}")

    print("\nRecommended reading: 4-hour threshold — ~35% coverage, and the position")
    print("distribution is close to flat (a real spread, not clustered near the start).")
    print("CAVEAT: a big gap proves the person stepped away, not that the next")
    print("message is a different TASK — treat as 'a point where restarting would")
    print("plausibly have helped', not proof the task itself changed.")


# ------------------------------------------------------------------------
# Section F (continued) — how much money splitting at the time gap would save
# ------------------------------------------------------------------------
#
# Same idea and same honest method as the earlier compaction-splitting
# analysis: use the REAL per-turn `usage` block (input/output/cache-write/
# cache-read tokens) recorded on each assistant turn, not an estimate. At
# each detected time-gap split point, the new "chunk" no longer re-reads
# the old context — approximated by zeroing that turn's cache-read tokens
# from the split point onward (same simplification used before). A
# realistic handoff cost is charged once per split: a ~5,000-token summary,
# which costs both to GENERATE (output tokens) and to CARRY FORWARD into
# the new chunk's cache (cache-write tokens) — both are included here.

def turn_cost(u):
    return (u.get("input_tokens", 0) * P_IN
          + u.get("output_tokens", 0) * P_OUT
          + u.get("cache_creation_input_tokens", 0) * P_WRITE
          + u.get("cache_read_input_tokens", 0) * P_READ)


def gaps_and_usage_in_order(events):
    """
    One pass over a transcript. Returns:
      - gaps: [(turn_number, gap_seconds), ...] — same as user_gaps_in_order
      - usage_by_turn: {turn_number: usage_dict} — the real recorded usage
        on each assistant turn (input/output/cache tokens)
    Both keyed on the SAME turn counter, so they line up exactly.
    """
    turn_no = 0
    last_ts = None
    gaps = []
    usage_by_turn = {}
    for e in events:
        if e.get("type") == "assistant":
            turn_no += 1
            u = e.get("message", {}).get("usage")
            if isinstance(u, dict):
                usage_by_turn[turn_no] = u
        if e.get("type") == "user":
            ts = parse_ts(e.get("timestamp"))
            if ts is not None:
                if last_ts is not None:
                    gap_seconds = (ts - last_ts).total_seconds()
                    if gap_seconds > 0:
                        gaps.append((turn_no, gap_seconds))
                last_ts = ts
    return gaps, usage_by_turn


def simulate_split_savings(usage_by_turn, gaps, threshold, handoff_tokens=HANDOFF_TOKENS):
    """
    Return (original_cost, split_free_cost, split_real_cost, n_splits) for
    one session, splitting at every gap >= threshold.
    """
    orig = sum(turn_cost(u) for u in usage_by_turn.values())
    split_turns = sorted(set(t for t, g in gaps if g >= threshold))
    if not split_turns:
        return orig, orig, orig, 0

    first_split = split_turns[0]      # once reset, later turns stay "reset" too
    split_free = 0.0
    for t, u in usage_by_turn.items():
        if t >= first_split:
            reduced = dict(u)
            reduced["cache_read_input_tokens"] = 0
            split_free += turn_cost(reduced)
        else:
            split_free += turn_cost(u)

    handoff_each = handoff_tokens * P_WRITE + handoff_tokens * P_OUT
    split_real = split_free + len(split_turns) * handoff_each
    return orig, split_free, split_real, len(split_turns)


def section_f_time_gap_savings(long_ids, paths):
    print("\n" + "=" * 60)
    print("SECTION F (continued) — $ saved by splitting at the time gap")
    print("=" * 60)

    n_ok = n_skipped = 0
    totals = {label: {"orig": 0.0, "free": 0.0, "real": 0.0, "n_split_sessions": 0}
              for label in GAP_LABELS}

    for sid in long_ids:
        try:
            events = read_transcript(sid, paths)
            gaps, usage_by_turn = gaps_and_usage_in_order(events)
            if not usage_by_turn:
                continue
            n_ok += 1
            for label, threshold in zip(GAP_LABELS, GAP_THRESHOLDS):
                orig, free, real, n_splits = simulate_split_savings(usage_by_turn, gaps, threshold)
                totals[label]["orig"] += orig
                totals[label]["free"] += free
                totals[label]["real"] += real
                if n_splits > 0:
                    totals[label]["n_split_sessions"] += 1
        except Exception:
            n_skipped += 1

    print("sessions with usable per-turn cost data:", n_ok, "| skipped:", n_skipped)
    print(f"\n{'threshold':10s} {'sessions split':>14s} {'orig cost':>12s} "
          f"{'saved (free)':>13s} {'saved (real)':>13s}")
    for label in GAP_LABELS:
        t = totals[label]
        saved_free = t["orig"] - t["free"]
        saved_real = t["orig"] - t["real"]
        pct_free = 100 * saved_free / t["orig"] if t["orig"] else 0
        pct_real = 100 * saved_real / t["orig"] if t["orig"] else 0
        print(f"{label:10s} {t['n_split_sessions']:>14,} ${t['orig']:>10,.0f} "
              f"${saved_free:>9,.0f} ({pct_free:>3.0f}%) ${saved_real:>9,.0f} ({pct_real:>3.0f}%)")

    print("\n'free' = an unrealistic best case (the split costs nothing to set up).")
    print("'real' = charges a ~5,000-token handoff summary per split (its generation")
    print("as output tokens, plus carrying it into the new chunk as cache tokens).")
    print("Use the REAL column and the 4-hour row as the headline, honest figure.")


# ======================================================================
# main
# ======================================================================

def main():
    sessions, logs = load_tables()
    long_ids = long_session_ids(sessions)
    paths = path_map(logs)

    # Cost/size analyses — comment out any you don't need
    step_1_2(sessions)
    step_3_categories(long_ids, paths)
    step_4_tokens(long_ids, paths)
    step_5_savings(long_ids, paths)

    # Task-switch detection
    section_f_time_gap(long_ids, paths)
    section_f_time_gap_savings(long_ids, paths)


if __name__ == "__main__":
    main()