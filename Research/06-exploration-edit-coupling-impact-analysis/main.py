"""
main.py — Context-coupling analysis on the SWE-chat dataset.

WHAT THIS FILE DOES:
    Measures, on real long Claude Code sessions, how much of the agent's
    work is exploration (reading), editing, or execution (running
    commands), and what that costs in tokens/money.

SECTIONS (run all, or comment out what you don't need in main()):
    A — shared setup (load data, long-session filter, transcript reader)
    B — Step 1+2: how many long sessions, exploration vs editing
    C — Step 3: tool categories (adds "execution")
    D — Step 4: token weight per category (exploration = biggest)
    E — Step 5: exploration cost + Viewer-style saving estimate

Requirements: pip install datasets huggingface_hub
The dataset is gated: accept terms on huggingface.co and run `hf auth login` first.
"""

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import Counter
import json

REPO = "SALT-NLP/SWE-chat"
LONG_SESSION_MIN_TURNS = 30

# Opus 4.5 pricing ($ per token) — used by Step 5
BASE_INPUT = 5.00
P_WRITE = BASE_INPUT * 1.25 / 1_000_000     # cache write
P_READ  = BASE_INPUT * 0.10 / 1_000_000     # cache read
VIEWER_FILTER_RATE = 0.60                   # SWE-Edit's measured filter rate

# Bash commands that only READ (don't change anything) — used to tell
# "bash grep" or "bash cat" apart from a real build/test/execute command.
# See categorize() below — item 4 of the feedback.
READ_ONLY_BASH_COMMANDS = {
    "cat", "grep", "find", "ls", "head", "tail", "wc",
    "less", "more", "tree", "pwd", "which", "file", "diff",
}


# ======================================================================
# Section A — shared setup
# ======================================================================

def load_tables():
    """Load the two tables we use. Cached locally after first download."""
    sessions = load_dataset(REPO, "sessions", split="train")
    logs = load_dataset(REPO, "session_logs", split="train")
    return sessions, logs


def is_long(session):
    """A 'long' session: has token data, >30 turns, and is Claude Code.

    NOTE (only Claude Code): the parsing below (message.content shape,
    tool_use/tool_result blocks, the usage field) matches Claude Code's
    transcript format specifically. Verified by sampling real transcripts
    from every agent in the dataset:
      - Codex and Copilot CLI use completely different event schemas (not
        assistant/user/tool_use at all).
      - Cursor's events are missing their "type" field entirely.
      - OpenCode and Gemini CLI mostly fail to parse as JSON in this
        format: 7 of 10 sampled Gemini CLI sessions threw a JSON error;
        the one that looked Claude-Code-shaped was not representative.
    So including these agents would silently misread most of their
    transcripts, not just skip them. Kept as Claude-Code-only — confirmed,
    not assumed.
    """
    return (session["input_tokens"]
            and session["turn_count"] and session["turn_count"] > LONG_SESSION_MIN_TURNS
            and session["agent"] == "Claude Code")


def long_session_ids(sessions):
    return [session["session_id"] for session in sessions if is_long(session)]


def path_map(logs):
    return {log["session_id"]: log["transcript_path"] for log in logs}


def read_transcript(session_id, paths):
    """Download (cached) and parse one transcript into a list of events."""
    local_path = hf_hub_download(REPO, paths[session_id], repo_type="dataset")
    events = []
    with open(local_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue          # skip malformed lines — rare, safe to ignore
    return events


def is_read_only_bash_command(tool_input):
    """
    A Bash call can run a read-only command (cat, grep, find, ls, ...),
    which is really exploration, not execution — or a real build/test/git
    command, which is execution. Judging by the tool NAME alone ("Bash")
    can't tell these apart, so this looks at the actual command text.
    (Feedback item 4: "bash grep is also exploration.")
    """
    if not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command", "")
    if not isinstance(command, str) or not command.strip():
        return False
    first_word = command.strip().split()[0]
    return first_word in READ_ONLY_BASH_COMMANDS


def categorize(tool_name, tool_input=None):
    """
    Sort a tool call into one of four activity categories. For Bash calls,
    look at the actual command (see is_read_only_bash_command) instead of
    assuming every Bash call is "execution" — a plain `grep`/`cat`/`find`
    run through Bash is exploration, not execution.
    """
    name = tool_name.lower()
    if any(keyword in name for keyword in ["read", "grep", "glob", "lsp", "search", "fetch"]):
        return "exploration"
    if any(keyword in name for keyword in ["edit", "write"]) and "todowrite" not in name:
        return "editing"
    if "bash" in name:
        if is_read_only_bash_command(tool_input):
            return "exploration"
        return "execution"
    return "coordination"


def is_exploration(tool_name, tool_input=None):
    """
    Whether a tool call counts as exploration. Reuses categorize() instead
    of keeping a second copy of the keyword list (feedback item 2: the two
    lists could silently drift apart).
    """
    return categorize(tool_name, tool_input) == "exploration"


def estimate_tokens(value):
    """
    Rough token estimate for any tool result: characters / 4. (Feedback
    item 1: the old function name/docstring said "tokens" but returned
    characters, and callers divided by 4 outside the function — the /4 is
    now done here, so the function's name matches what it returns.)
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value) / 4
    return len(json.dumps(value)) / 4


# ======================================================================
# Section B — Step 1 + 2: count long sessions, exploration vs editing
# ======================================================================

def step_1_2(sessions):
    print("\n" + "=" * 60)
    print("STEP 1+2 — long sessions, exploration vs editing")
    print("=" * 60)

    longs = [session for session in sessions if is_long(session)]
    print("long Claude Code sessions (>30 turns):", len(longs))

    total_research = total_action = usable = 0
    per_session = []
    for session in longs:
        if not session["research_count"] or not session["action_count"]:
            continue
        total_research += session["research_count"]
        total_action += session["action_count"]
        usable += 1
        total = session["research_count"] + session["action_count"]
        per_session.append(session["research_count"] / total)

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
    for session_id in long_ids:
        try:
            for event in read_transcript(session_id, paths):
                if event.get("type") != "assistant":
                    continue
                for block in event["message"].get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        cat = categorize(block.get("name", ""), block.get("input", {}))
                        cat_counts[cat] += 1
            n_ok += 1
        except Exception:
            n_skipped += 1

    total = sum(cat_counts.values()) or 1
    print("sessions read:", n_ok, "| skipped:", n_skipped, "| total tool calls:", f"{total:,}")
    for cat in ["exploration", "editing", "execution", "coordination"]:
        count = cat_counts[cat]
        print(f"  {cat:14s} {count:>8,}  ({100*count/total:.0f}%)")


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

    for session_id in long_ids:
        try:
            id_to_cat = {}
            for event in read_transcript(session_id, paths):
                message = event.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if event.get("type") == "assistant" and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            cat = categorize(block.get("name", ""), block.get("input", {}))
                            id_to_cat[block.get("id")] = cat
                            cat_calls[cat] += 1
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_use_id = block.get("tool_use_id")
                            if tool_use_id in id_to_cat:
                                cat = id_to_cat[tool_use_id]; matched += 1
                            else:
                                cat = "coordination"; unmatched += 1
                            cat_tokens[cat] += estimate_tokens(block.get("content"))
            n_ok += 1
        except Exception:
            n_skipped += 1

    print("sessions read:", n_ok, "| skipped:", n_skipped)
    print(f"results matched: {matched:,} | unmatched: {unmatched:,}")
    tot_calls = sum(cat_calls.values()) or 1
    tot_tokens = sum(cat_tokens.values()) or 1
    print(f"\n{'category':14s} {'calls':>10s} {'call%':>7s} {'tokens(M)':>11s} {'token%':>8s}")
    for cat in ["exploration", "editing", "execution", "coordination"]:
        count = cat_calls[cat]; tokens = cat_tokens[cat]
        print(f"{cat:14s} {count:>10,} {100*count/tot_calls:>6.0f}% "
              f"{tokens/1e6:>10.1f}M {100*tokens/tot_tokens:>7.0f}%")


# ======================================================================
# Section E — Step 5: exploration cost + Viewer-style saving
# ======================================================================

def step_5_savings(long_ids, paths, filter_rate=VIEWER_FILTER_RATE):
    print("\n" + "=" * 60)
    print("STEP 5 — exploration cost & Viewer-style saving")
    print("=" * 60)

    total_expl_cost = total_all_cost = 0.0
    n_ok = n_skipped = 0
    for session_id in long_ids:
        try:
            events = read_transcript(session_id, paths)
            total_turns = sum(1 for event in events if event.get("type") == "assistant")
            if total_turns == 0:
                continue
            id_to_call = {}     # tool_use id -> (name, input, turn_number)
            turn_no = 0
            for event in events:
                message = event.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if event.get("type") == "assistant":
                    turn_no += 1
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                id_to_call[block.get("id")] = (
                                    block.get("name", ""), block.get("input", {}), turn_no
                                )
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            name, tool_input, turn = id_to_call.get(
                                block.get("tool_use_id"), (None, None, turn_no)
                            )
                            result_tokens = estimate_tokens(block.get("content"))
                            remaining = max(total_turns - turn, 0)
                            life_cost = result_tokens * (P_WRITE + remaining * P_READ)
                            total_all_cost += life_cost
                            if name and is_exploration(name, tool_input):
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
# main
# ======================================================================

def main():
    sessions, logs = load_tables()
    long_ids = long_session_ids(sessions)
    paths = path_map(logs)

    step_1_2(sessions)
    step_3_categories(long_ids, paths)
    step_4_tokens(long_ids, paths)
    step_5_savings(long_ids, paths)


if __name__ == "__main__":
    main()