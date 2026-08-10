"""
main.py — consolidated SWE-chat trace analysis.

Merges nine separate scripts into one runnable file. The `sessions` table is
loaded once and reused across every section; the sample transcript is downloaded
once and shared by the peek + context-growth sections.

Sections
  1. Long-session prevalence            (was data.py)
  2. Cost breakdown of long sessions    (was details_sonnet.py / money.py)
  3. Total spend + long-session share   (was long_sessions.py)
  4. Agent distribution (long sessions) (was model_and_growth.py, part 1)
  5. Research vs. action turns          (was exploration.py)
  6. Transcript structure peek          (was peek_transcript.py / peek_tokens.py)
  7. Context growth across turns        (was model_and_growth.py, part 2)
  8. Split-simulation savings           (was splitter.py)

Requirements: pip install datasets huggingface_hub
"""

import json
from collections import Counter

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REPO = "SALT-NLP/SWE-chat"
LONG_TURN_THRESHOLD = 30           # a session is "long" if turn_count > this
SAMPLE_TRANSCRIPT = "transcripts/2026-01-05-0bb9a2d9-a51e-4d5a-8275-566c766a0ff2.jsonl"

# Pricing per model, as (base_input $/M, base_output $/M).
# Cache write = 1.25x input, cache read = 0.10x input (Anthropic caching economics).
PRICING = {
    "Sonnet": (3.00, 15.00),
    "Opus":   (5.00, 25.00),
}

# Section toggles — the transcript-download sections need network + HF access.
RUN_TRANSCRIPT_SECTIONS = True     # sections 6 & 7 (one small download)
RUN_SPLIT_SIMULATION = True        # section 8 (downloads many transcripts — slow)

# Split-simulation params (section 8)
SPLIT_SAMPLE = 797                 # how many long Claude Code sessions to analyze
SPLIT_LIMIT = 120_000              # context ceiling that triggers a "split"
HANDOFF_TOKENS = 5_000             # summary re-fed to each new part on a split


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def make_prices(base_input, base_output):
    """Per-token prices derived from per-million base rates."""
    return {
        "in":    base_input  / 1_000_000,
        "out":   base_output / 1_000_000,
        "write": base_input * 1.25 / 1_000_000,
        "read":  base_input * 0.10 / 1_000_000,
    }


def is_long(s):
    return bool(s["turn_count"]) and s["turn_count"] > LONG_TURN_THRESHOLD


def session_cost(s, p):
    """Cost of one session row given a per-token price dict."""
    return (s["input_tokens"]          * p["in"]
          + s["output_tokens"]         * p["out"]
          + s["cache_creation_tokens"] * p["write"]
          + s["cache_read_tokens"]     * p["read"])


def turn_cost(usage, p):
    """Cost of one assistant turn from its raw transcript `usage` block."""
    return (usage.get("input_tokens", 0)                * p["in"]
          + usage.get("output_tokens", 0)               * p["out"]
          + usage.get("cache_creation_input_tokens", 0) * p["write"]
          + usage.get("cache_read_input_tokens", 0)     * p["read"])


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 1. Long-session prevalence
# --------------------------------------------------------------------------- #
def section_prevalence(sessions):
    header("1. LONG-SESSION PREVALENCE")
    total = len(sessions)
    long_turns   = sum(1 for x in sessions if x["turn_count"]   and x["turn_count"]   > 30)
    long_prompts = sum(1 for x in sessions if x["prompt_count"] and x["prompt_count"] > 30)
    print(f"total sessions:      {total:,}")
    print(f"by turn_count >30:   {long_turns:,}  ({100*long_turns/total:.1f}%)")
    print(f"by prompt_count >30: {long_prompts:,}  ({100*long_prompts/total:.1f}%)")


# --------------------------------------------------------------------------- #
# 2. Cost breakdown of long sessions (per pricing model)
# --------------------------------------------------------------------------- #
def section_cost_breakdown(sessions):
    header("2. COST BREAKDOWN OF LONG SESSIONS (>30 turns)")
    for name, (bi, bo) in PRICING.items():
        p = make_prices(bi, bo)
        c_in = c_out = c_write = c_read = 0.0
        for s in sessions:
            if not s["input_tokens"] or not is_long(s):
                continue
            c_in    += s["input_tokens"]          * p["in"]
            c_out   += s["output_tokens"]         * p["out"]
            c_write += s["cache_creation_tokens"] * p["write"]
            c_read  += s["cache_read_tokens"]     * p["read"]
        total = c_in + c_out + c_write + c_read
        if total == 0:
            continue
        print(f"\n[{name} pricing @ {bi}/{bo} per M]")
        print(f"  fresh input:  ${c_in:,.0f}  ({100*c_in/total:.0f}%)")
        print(f"  output:       ${c_out:,.0f}  ({100*c_out/total:.0f}%)")
        print(f"  cache writes: ${c_write:,.0f}  ({100*c_write/total:.0f}%)")
        print(f"  cache reads:  ${c_read:,.0f}  ({100*c_read/total:.0f}%)")
        print(f"  TOTAL:        ${total:,.0f}")


# --------------------------------------------------------------------------- #
# 3. Total spend + share attributable to long sessions
# --------------------------------------------------------------------------- #
def section_spend_share(sessions):
    header("3. TOTAL SPEND AND LONG-SESSION SHARE")
    for name, (bi, bo) in PRICING.items():
        p = make_prices(bi, bo)
        all_cost = long_cost = 0.0
        all_count = long_count = 0
        for s in sessions:
            if not s["input_tokens"]:
                continue
            c = session_cost(s, p)
            all_cost += c
            all_count += 1
            if is_long(s):
                long_cost += c
                long_count += 1
        if all_cost == 0:
            continue
        print(f"\n[{name} pricing @ {bi}/{bo} per M]")
        print(f"  sessions with token data: {all_count:,}")
        print(f"    of those, long (>30):   {long_count:,} "
              f"({100*long_count/all_count:.1f}%)")
        print(f"  total spend:              ${all_cost:,.2f}")
        print(f"    spent on long sessions: ${long_cost:,.2f} "
              f"({100*long_cost/all_cost:.1f}%)")


# --------------------------------------------------------------------------- #
# 4. Agent distribution for long sessions
# --------------------------------------------------------------------------- #
def section_agents(sessions):
    header("4. AGENT DISTRIBUTION (LONG SESSIONS)")
    agents = Counter()
    for s in sessions:
        if s["input_tokens"] and is_long(s):
            agents[s["agent"]] += 1
    for agent, count in agents.most_common():
        print(f"  {agent}: {count:,}")


# --------------------------------------------------------------------------- #
# 5. Research vs. action turns (long Claude Code sessions)
# --------------------------------------------------------------------------- #
def section_exploration(sessions):
    header("5. RESEARCH vs. ACTION TURNS (long Claude Code sessions)")
    total_research = total_action = n = 0
    for s in sessions:
        if not s["input_tokens"] or not is_long(s):
            continue
        if s["agent"] != "Claude Code":
            continue
        if not s["research_count"] or not s["action_count"]:
            continue
        total_research += s["research_count"]
        total_action += s["action_count"]
        n += 1
    both = total_research + total_action
    if both == 0:
        print("  no qualifying sessions found.")
        return
    print(f"  sessions analyzed:            {n:,}")
    print(f"  exploration (research) turns: {total_research:,}  ({100*total_research/both:.0f}%)")
    print(f"  editing (action) turns:       {total_action:,}  ({100*total_action/both:.0f}%)")


# --------------------------------------------------------------------------- #
# 6. Transcript structure peek (shares a downloaded transcript with section 7)
# --------------------------------------------------------------------------- #
def load_sample_transcript():
    path = hf_hub_download(repo_id=REPO, filename=SAMPLE_TRANSCRIPT, repo_type="dataset")
    with open(path) as f:
        events = [json.loads(ln) for ln in f]
    return path, events


def section_peek(path, events):
    header("6. TRANSCRIPT STRUCTURE PEEK")
    print(f"  file: {path}")
    print(f"  total events: {len(events):,}")
    print(f"  event types: {Counter(e.get('type') for e in events)}")

    keys = set()
    for e in events:
        keys.update(e.keys())
    print(f"  top-level keys across events: {sorted(keys)}")

    for e in events:
        if e.get("type") == "assistant":
            msg = e.get("message", {})
            print(f"\n  first assistant message keys: {sorted(msg.keys())}")
            print(f"  model: {msg.get('model', 'no model field')}")
            print("  usage block:")
            print("    " + json.dumps(msg.get("usage", "NO USAGE FIELD"), indent=2)
                  .replace("\n", "\n    "))
            break


# --------------------------------------------------------------------------- #
# 7. Context growth across turns (reuses the section-6 transcript)
# --------------------------------------------------------------------------- #
def section_context_growth(events):
    header("7. CONTEXT GROWTH ACROSS TURNS (cache_read + cache_creation)")
    turn = 0
    for e in events:
        if e.get("type") != "assistant":
            continue
        u = e["message"].get("usage", {})
        ctx = u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        turn += 1
        if turn <= 5 or turn % 20 == 0:   # first 5 turns, then every 20th
            print(f"  turn {turn:3d}: context ~ {ctx:,} tokens")


# --------------------------------------------------------------------------- #
# 8. Split-simulation savings (downloads many transcripts — slow)
# --------------------------------------------------------------------------- #
def analyze_split(events, p):
    orig = split_free = split_real = 0.0
    for e in events:
        if e.get("type") != "assistant":
            continue
        u = e["message"].get("usage", {})
        ctx = u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        orig += turn_cost(u, p)
        if ctx > SPLIT_LIMIT:
            reduced = dict(u)
            reduced["cache_read_input_tokens"] = 0
            base = turn_cost(reduced, p)
            split_free += base
            split_real += base + HANDOFF_TOKENS * p["write"]
        else:
            split_free += turn_cost(u, p)
            split_real += turn_cost(u, p)
    return orig, split_free, split_real


def section_split_simulation(sessions):
    header("8. SPLIT-SIMULATION SAVINGS (Opus pricing)")
    p = make_prices(*PRICING["Opus"])

    logs = load_dataset(REPO, "session_logs", split="train")
    long_ids = [s["session_id"] for s in sessions
                if s["input_tokens"] and is_long(s) and s["agent"] == "Claude Code"]
    path_by_id = {l["session_id"]: l["transcript_path"] for l in logs}

    o = sf = sr = 0.0
    n = 0
    for sid in long_ids[:SPLIT_SAMPLE]:
        try:
            p_file = hf_hub_download(REPO, path_by_id[sid], repo_type="dataset")
            with open(p_file) as f:
                ev = [json.loads(x) for x in f]
            a, b, c = analyze_split(ev, p)
            o += a; sf += b; sr += c; n += 1
        except Exception as err:
            print("  skip", sid, err)

    print(f"\n  sessions analyzed: {n}")
    if o == 0:
        print("  no cost accumulated.")
        return
    print(f"  original cost:        ${o:,.2f}")
    print(f"  split (free handoff): ${sf:,.2f}   saving ${o-sf:,.2f}  ({100*(o-sf)/o:.0f}%)")
    print(f"  split (real handoff): ${sr:,.2f}   saving ${o-sr:,.2f}  ({100*(o-sr)/o:.0f}%)")
    print("\n  free handoff = best-case ceiling (summary assumed free).")
    print("  real handoff = charges", f"{HANDOFF_TOKENS:,}", "tokens per split (closer to reality).")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    print("Loading sessions table (once)...")
    sessions = load_dataset(REPO, "sessions", split="train")

    section_prevalence(sessions)
    section_cost_breakdown(sessions)
    section_spend_share(sessions)
    section_agents(sessions)
    section_exploration(sessions)

    if RUN_TRANSCRIPT_SECTIONS:
        try:
            path, events = load_sample_transcript()
            section_peek(path, events)
            section_context_growth(events)
        except Exception as err:
            print("\n[transcript sections skipped]", err)

    if RUN_SPLIT_SIMULATION:
        try:
            section_split_simulation(sessions)
        except Exception as err:
            print("\n[split simulation skipped]", err)


if __name__ == "__main__":
    main()