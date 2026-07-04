"""
main.py — Context-coupling analysis + LLM-judged task-switch detection
          on the SWE-chat dataset.

WHAT THIS FILE DOES:
    Part 1 (Sections A-E): measures, on real long Claude Code sessions, how
    much of the agent's work is exploration (reading), editing, or
    execution (running commands), and what that costs in tokens/money.

    Part 2 (Section F): detects whether a session contains a genuine task
    switch, using an LLM as a judge instead of a hand-written heuristic.
    Earlier attempts (word overlap, file overlap, time gaps — see the
    project report) all failed to reliably tell apart "one big task
    spanning many files/words" from "a real new task," because none of
    them understand MEANING. An LLM can actually read the conversation and
    judge intent, which is what this section does. It does NOT calculate
    any dollar savings — that will come from a separate math model.

SECTIONS (run all, or comment out what you don't need in main()):
    A — shared setup (load data, long-session filter, transcript reader)
    B — Step 1+2: how many long sessions, exploration vs editing
    B2 — raw tool-name counts (not categorized), per tool and total
    C — Step 3: tool categories (adds "execution")
    D — Step 4: token weight per category (exploration = biggest)
    E — Step 5: exploration cost + Viewer-style saving estimate
    F — LLM-as-judge task-switch detection (no dollar savings)

Requirements: pip install datasets huggingface_hub openai
The dataset is gated: accept terms on huggingface.co and run `hf auth login` first.
Section F calls Claude through JetBrains' internal LiteLLM proxy
(https://litellm.labs.jb.gg), using the OpenAI-compatible API. Set your
LiteLLM key as the OPENAI_API_KEY environment variable (the SDK reads it
automatically) before running.
"""

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import Counter
import json
from openai import OpenAI

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

# Section F — LLM-as-judge settings.
# Haiku is used because this runs once per SESSION (not per turn), so cost
# stays low, but it is still a real API call that costs money — default to
# a small SAMPLE while testing, then raise it once you trust the results.
JUDGE_MODEL = "anthropic/claude-haiku-4-5"   # confirmed working via LiteLLM
JUDGE_SAMPLE = None
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"   # confirmed working base URL


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
# Section B2 — raw tool-name counts (not categorized), per tool and total
# ======================================================================

def step_tool_name_counts(long_ids, paths):
    """
    Prints how many times each individual tool was called (Read, Edit,
    Bash, Grep, TodoWrite, ...) across all long sessions, most common
    first, plus the grand total. This is the raw count BEFORE any
    exploration/editing/execution categorization — useful for spotting
    tools the categorizer doesn't know about yet.
    """
    print("\n" + "=" * 60)
    print("STEP (tool names) — raw tool call counts, per tool and total")
    print("=" * 60)

    tool_counts = Counter()
    n_ok = n_skipped = 0
    for session_id in long_ids:
        try:
            for event in read_transcript(session_id, paths):
                if event.get("type") != "assistant":
                    continue
                for block in event["message"].get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_counts[block.get("name", "unknown")] += 1
            n_ok += 1
        except Exception:
            n_skipped += 1

    total = sum(tool_counts.values())
    print("sessions read:", n_ok, "| skipped:", n_skipped)
    print(f"\n{'tool name':30s} {'count':>10s}")
    for name, count in tool_counts.most_common():
        print(f"{name:30s} {count:>10,}")
    print(f"{'-'*30} {'-'*10}")
    print(f"{'TOTAL':30s} {total:>10,}")


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
# Section F — LLM-as-judge task-switch detection (NO dollar savings)
# ======================================================================
#
# Earlier heuristics (word overlap, file overlap, time gaps) all failed or
# were rejected, because none of them understand MEANING — they can't tell
# "one big task spanning many files/words" apart from "a real new task."
# An LLM can actually read the conversation and judge intent.
#
# Cost control: this makes ONE real API call per session (on the ordered
# list of user prompts), not one call per turn-pair — much cheaper than
# judging every point in every session. Still, it is a real, billed API
# call, so JUDGE_SAMPLE defaults to a small number for testing.
#
# This section deliberately does NOT calculate any dollar savings — a
# separate math model for that is coming.

def user_prompts_with_turns(events):
    """
    Return (prompts, total_turns). prompts = [(turn_number, text), ...],
    the user's typed messages in order, each tagged with how many
    assistant turns had happened so far (for position-in-session stats).
    """
    turn_no = 0
    prompts = []
    for event in events:
        if event.get("type") == "assistant":
            turn_no += 1
        if event.get("type") == "user":
            message = event.get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [block.get("text", "") for block in content
                         if isinstance(block, dict) and block.get("type") == "text"]
                text = " ".join(parts)
            if text.strip():
                prompts.append((turn_no, text.strip()))
    return prompts, turn_no


def build_judge_prompt(prompt_texts):
    numbered = "\n".join(f"{i+1}. {text}" for i, text in enumerate(prompt_texts))
    return (
        "Here are the user's messages from one coding session, in order:\n\n"
        f"{numbered}\n\n"
        "Question: does the user switch to a clearly different, unrelated task "
        "at any point in this list?\n\n"
        "Do NOT count these as a switch (they are the same task, just a "
        "different phase of it):\n"
        "- moving from discussing/designing to writing a plan\n"
        "- moving from planning to implementation\n"
        "- moving from writing code to testing or fixing bugs IN THAT SAME "
        "feature/change\n"
        "- using different files or different words for the same underlying goal\n\n"
        "Only answer yes if the underlying GOAL genuinely changes to something "
        "unrelated — e.g. the user was fixing one bug and now asks about a "
        "completely different feature or problem area.\n\n"
        "Answer with ONLY a JSON object, no other text:\n"
        '{"has_switch": true or false, "switch_message_number": <number of the '
        'first message that starts the new task, or null>, "reason": '
        '"<one short sentence>"}'
    )


def judge_session(client, prompt_texts, model=JUDGE_MODEL):
    """Ask the LLM to judge one session's user prompts. Returns a dict or None."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=200,
            temperature=0,      # reduce run-to-run randomness in the judgment
            messages=[{"role": "user", "content": build_judge_prompt(prompt_texts)}],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def section_f_llm_judge(long_ids, paths, sample=JUDGE_SAMPLE, verbose=True):
    print("\n" + "=" * 60)
    print("SECTION F — LLM-as-judge task-switch detection")
    print("=" * 60)
    print(f"(testing on a sample of {sample} sessions — raise `sample` once trusted)")

    client = OpenAI(base_url=LITELLM_BASE_URL)   # reads OPENAI_API_KEY from the environment

    ids = long_ids if sample is None else long_ids[:sample]
    n_ok = n_skipped = n_judged = n_switch = 0
    switch_positions = []

    for session_id in ids:
        try:
            events = read_transcript(session_id, paths)
            prompts, total_turns = user_prompts_with_turns(events)
            if len(prompts) < 4 or total_turns == 0:
                continue
            n_ok += 1

            result = judge_session(client, [text for _, text in prompts])
            if not result or "error" in result:
                n_skipped += 1
                continue
            n_judged += 1

            if result.get("has_switch"):
                n_switch += 1
                switch_num = result.get("switch_message_number")
                if isinstance(switch_num, int) and 1 <= switch_num <= len(prompts):
                    switch_turn = prompts[switch_num - 1][0]
                    switch_positions.append(min(switch_turn / total_turns, 1.0))
                if verbose:
                    print(f"  [{session_id}] switch at msg {switch_num}: {result.get('reason')}")
        except Exception:
            n_skipped += 1

    print(f"\nsessions with enough prompts to judge: {n_ok} | skipped (API/parse errors): {n_skipped}")
    print(f"sessions successfully judged: {n_judged}")
    if n_judged == 0:
        print("no successful judgments — check ANTHROPIC_API_KEY and the `anthropic` package")
        return

    pct = 100 * n_switch / n_judged
    print(f"sessions with a detected task switch: {n_switch}  ({pct:.0f}%)")
    if switch_positions:
        avg_pos = 100 * sum(switch_positions) / len(switch_positions)
        buckets = [0, 0, 0, 0]
        for frac in switch_positions:
            buckets[min(int(frac * 4), 3)] += 1
        print(f"  average switch position: {avg_pos:.0f}% of the way through the session")
        print("  distribution (0-25/25-50/50-75/75-100):", buckets)
    print("\nNOTE: no dollar savings are calculated here — that comes from a")
    print("separate math model. This section only reports whether/where a real")
    print("task switch was found.")


# ======================================================================
# Section G — Sequential analysis: detect missed plan-mode opportunities
# ======================================================================
#
# Steps 3/4 count HOW MUCH exploration happens in total. This section
# looks at WHEN it happens — the order of tool calls matters, because a
# long block of pure exploration at the start of a session could have been
# run in "plan mode" (a cheaper model doing the reading), while scattered
# exploration mixed throughout cannot be cleanly separated.
#
# Three patterns to detect (from mentor feedback):
#
#   Pattern A — "front-loaded exploration": a long exploration phase at the
#   START, followed by a shift to editing/execution. The exploration phase
#   is a missed plan-mode opportunity — it could have been offloaded.
#
#   Pattern B — "mixed throughout": exploration and editing/execution are
#   interleaved with no clean separation. Cannot be split into plan mode
#   without breaking the workflow.
#
#   Pattern C — "mid-session exploration burst": a new user prompt triggers
#   a long exploration phase in the MIDDLE of the session, after editing
#   has already started. A candidate for a sub-agent (a cheaper reader
#   invoked mid-session) rather than a full session split.
#
# This section does NOT calculate dollar savings — a separate math model
# will use the phase data for that. It only detects and reports the
# patterns and their sizes (call counts, token estimates).

# Tunable: how many consecutive calls of one category count as a "phase"
MIN_PHASE_LENGTH = 5           # at least this many calls to be a real phase
EXPLORATION_DOMINANCE = 0.70   # a phase is "exploration" if >= 70% of its calls are exploration


def extract_tool_sequence(events):
    """
    Walk a transcript in order and return:
      - sequence: [(turn_number, category, tool_name, estimated_result_tokens), ...]
        one entry per tool call, in order
      - total_turns: total assistant turns (same counter as everywhere else)

    This is the raw, ordered list that the phase detector segments.
    """
    sequence = []
    turn_no = 0
    id_to_call = {}     # tool_use id -> (category, turn_number)
    pending_tokens = {}  # tool_use id -> estimated result tokens (filled when result arrives)

    for event in events:
        message = event.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None

        if event.get("type") == "assistant":
            turn_no += 1
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        cat = categorize(tool_name, tool_input)
                        tool_id = block.get("id")
                        id_to_call[tool_id] = (cat, turn_no, tool_name)
                        # placeholder — token count filled when result arrives
                        sequence.append({
                            "turn": turn_no,
                            "category": cat,
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                            "result_tokens": 0,
                        })

        # fill in result tokens when we see the matching tool_result
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    tokens = estimate_tokens(block.get("content"))
                    # find the matching entry in sequence and update it
                    for entry in reversed(sequence):
                        if entry["tool_id"] == tool_use_id:
                            entry["result_tokens"] = tokens
                            break

    return sequence, turn_no


def detect_phases(sequence, min_length=MIN_PHASE_LENGTH):
    """
    Segment the ordered tool-call sequence into consecutive phases. A phase
    is a maximal run where the same category appears >= EXPLORATION_DOMINANCE
    of the time. Returns a list of phase dicts:
      {"category": str, "start_idx": int, "end_idx": int, "length": int,
       "total_tokens": float, "start_turn": int, "end_turn": int}
    """
    if not sequence:
        return []

    phases = []
    current_cat = sequence[0]["category"]
    start_idx = 0

    for i in range(1, len(sequence) + 1):
        # end of sequence or category changed
        if i == len(sequence) or sequence[i]["category"] != current_cat:
            length = i - start_idx
            if length >= min_length:
                total_tokens = sum(entry["result_tokens"] for entry in sequence[start_idx:i])
                phases.append({
                    "category": current_cat,
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "length": length,
                    "total_tokens": total_tokens,
                    "start_turn": sequence[start_idx]["turn"],
                    "end_turn": sequence[i - 1]["turn"],
                })
            if i < len(sequence):
                current_cat = sequence[i]["category"]
                start_idx = i

    return phases


def classify_session_pattern(phases, sequence):
    """
    Classify a session into Pattern A, B, or C based on its phases.
    Returns (pattern, exploration_phases) where:
      - pattern: "A" (front-loaded), "B" (mixed), "C" (mid-burst), or
                 "none" (no significant exploration phase found)
      - exploration_phases: list of exploration-phase dicts from detect_phases
    """
    if not phases or not sequence:
        return "none", []

    expl_phases = [p for p in phases if p["category"] == "exploration"]
    if not expl_phases:
        return "none", []

    total_calls = len(sequence)
    first_expl = expl_phases[0]

    # Pattern A: first significant phase is exploration, starting near the
    # beginning of the session (within the first 20% of calls)
    if first_expl["start_idx"] < total_calls * 0.20:
        # check if editing/execution follows after this exploration
        later_phases = [p for p in phases
                        if p["start_idx"] > first_expl["end_idx"]
                        and p["category"] in ("editing", "execution")]
        if later_phases:
            return "A", expl_phases

    # Pattern C: an exploration phase appears AFTER editing/execution has
    # already started (not at the very beginning)
    edit_exec_phases = [p for p in phases if p["category"] in ("editing", "execution")]
    if edit_exec_phases:
        first_action = edit_exec_phases[0]
        mid_explorations = [p for p in expl_phases if p["start_idx"] > first_action["end_idx"]]
        if mid_explorations:
            return "C", mid_explorations

    # Pattern B: exploration exists but is interleaved with editing/execution
    # without a clean separation
    if expl_phases and edit_exec_phases:
        return "B", expl_phases

    return "none", expl_phases


def section_g_sequential(long_ids, paths):
    print("\n" + "=" * 60)
    print("SECTION G — Sequential analysis: missed plan-mode opportunities")
    print("=" * 60)

    n_ok = n_skipped = 0
    pattern_counts = Counter()       # A / B / C / none
    pattern_a_sessions = []          # details for Pattern A sessions
    pattern_c_sessions = []          # details for Pattern C sessions
    all_expl_phase_lengths = []      # length of every exploration phase found
    all_expl_phase_tokens = []       # tokens of every exploration phase found

    for session_id in long_ids:
        try:
            events = read_transcript(session_id, paths)
            sequence, total_turns = extract_tool_sequence(events)
            if len(sequence) < 10:
                continue
            n_ok += 1

            phases = detect_phases(sequence)
            pattern, expl_phases = classify_session_pattern(phases, sequence)
            pattern_counts[pattern] += 1

            for phase in expl_phases:
                all_expl_phase_lengths.append(phase["length"])
                all_expl_phase_tokens.append(phase["total_tokens"])

            if pattern == "A":
                first_expl = expl_phases[0]
                pattern_a_sessions.append({
                    "session_id": session_id,
                    "expl_calls": first_expl["length"],
                    "expl_tokens": first_expl["total_tokens"],
                    "total_calls": len(sequence),
                    "position_pct": 100 * first_expl["start_idx"] / len(sequence),
                })
            elif pattern == "C":
                for phase in expl_phases:
                    pattern_c_sessions.append({
                        "session_id": session_id,
                        "expl_calls": phase["length"],
                        "expl_tokens": phase["total_tokens"],
                        "total_calls": len(sequence),
                        "position_pct": 100 * phase["start_idx"] / len(sequence),
                    })

        except Exception:
            n_skipped += 1

    print("sessions analysed:", n_ok, "| skipped:", n_skipped)
    total_classified = sum(pattern_counts.values())

    print(f"\nPattern classification (min phase length = {MIN_PHASE_LENGTH} calls):")
    for pattern, label in [
        ("A", "front-loaded exploration → editing (plan-mode opportunity)"),
        ("C", "mid-session exploration burst (sub-agent opportunity)"),
        ("B", "mixed throughout (not cleanly splittable)"),
        ("none", "no significant exploration phase found"),
    ]:
        count = pattern_counts[pattern]
        pct = 100 * count / total_classified if total_classified else 0
        print(f"  Pattern {pattern:4s}: {count:>5,}  ({pct:>4.0f}%)  — {label}")

    if pattern_a_sessions:
        avg_calls = sum(s["expl_calls"] for s in pattern_a_sessions) / len(pattern_a_sessions)
        avg_tokens = sum(s["expl_tokens"] for s in pattern_a_sessions) / len(pattern_a_sessions)
        print(f"\nPattern A detail ({len(pattern_a_sessions)} sessions):")
        print(f"  avg exploration phase length: {avg_calls:.0f} calls")
        print(f"  avg exploration phase tokens: {avg_tokens:,.0f} (estimated)")
        print("  these are the 'missed plan-mode' opportunities — the initial reading")
        print("  phase could have been done by a cheaper model before the main agent")
        print("  started editing.")

    if pattern_c_sessions:
        avg_calls = sum(s["expl_calls"] for s in pattern_c_sessions) / len(pattern_c_sessions)
        avg_tokens = sum(s["expl_tokens"] for s in pattern_c_sessions) / len(pattern_c_sessions)
        print(f"\nPattern C detail ({len(pattern_c_sessions)} mid-session exploration bursts):")
        print(f"  avg burst length: {avg_calls:.0f} calls")
        print(f"  avg burst tokens: {avg_tokens:,.0f} (estimated)")
        print("  these are candidates for a sub-agent: a cheaper reader could handle")
        print("  the exploration burst mid-session instead of the main model.")

    if all_expl_phase_lengths:
        print(f"\nAll exploration phases across all sessions:")
        print(f"  total phases found: {len(all_expl_phase_lengths)}")
        print(f"  avg length: {sum(all_expl_phase_lengths)/len(all_expl_phase_lengths):.0f} calls")
        print(f"  avg tokens: {sum(all_expl_phase_tokens)/len(all_expl_phase_tokens):,.0f} (estimated)")

    print("\nNOTE: no dollar savings calculated here — a separate math model will")
    print("use the phase data (call counts, token estimates, positions) for that.")
    print("This section only detects and classifies the structural patterns.")


# ======================================================================
# Section H — Apply the chunk-split math model to F and G split points
# ======================================================================
#
# This is where the detections become dollar/token savings, using the
# external math model in chunk_split_model.py (imported as `csm`). That
# module prices a "split a session at a boundary" scenario from a
# session's real token buckets. We feed it TWO kinds of semantic split:
#
#   - Section F style: split at the LLM-detected task-switch position
#     (2 chunks: before / after the switch).
#   - Section G style: split at the end of the initial exploration phase
#     (2 chunks: the plan-mode reading phase / the rest).
#
# Both reduce to "one semantic split -> 2 chunk heights" which is exactly
# the interface the math model exposes. We report the aggregate saving in
# tokens and dollars for each.
#
# NOTE: the math model assumes context grows as a roughly linear ramp and
# derives the peak from cache-read + call count. Sessions that don't fit
# that shape (very little cache-read) are skipped rather than forced.

import chunk_split_model as csm

SUMMARY_ALPHA = 0.10     # carried summary is ~10% of the first chunk's height


def session_token_buckets(events):
    """
    Sum a session's real billed token buckets from the per-turn usage blocks,
    plus count the API calls (assistant turns with usage). Returns
    (input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls).
    """
    input_tok = output_tok = cache_write_tok = cache_read_tok = 0
    api_calls = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        usage = event.get("message", {}).get("usage")
        if not isinstance(usage, dict):
            continue
        api_calls += 1
        input_tok += usage.get("input_tokens", 0)
        output_tok += usage.get("output_tokens", 0)
        cache_write_tok += usage.get("cache_creation_input_tokens", 0)
        cache_read_tok += usage.get("cache_read_input_tokens", 0)
    return input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls


def saving_for_split(buckets, split_fraction):
    """
    Given a session's token buckets and a split point (fraction 0..1 of the
    way through the session), return a dict with the dollar and token
    savings from the chunk-split math model, or None if the session doesn't
    fit the model (no cache-read / too few calls / degenerate split).

    Returned dict keys: as_is_cost, split_cost, dollar_saving,
    cache_read_before, cache_read_after, token_saving.
    """
    input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls = buckets
    if cache_read_tok <= 0 or api_calls < 2:
        return None
    if not (0.05 < split_fraction < 0.95):
        return None      # a split at the very start/end saves ~nothing; skip

    peak = csm.peak_context_tokens(cache_read_tok, api_calls)
    if peak <= 0:
        return None

    heights = [peak * split_fraction, peak * (1.0 - split_fraction)]
    calls = csm.calls_per_chunk(heights, api_calls, peak)
    cache_read_split = csm.cache_read_after_split(heights, api_calls, peak)

    summary_tok = csm.summary_tokens(heights[0], SUMMARY_ALPHA)
    overhead, *_ = csm.summary_overhead_cost(calls[1:], summary_tok)

    as_is = csm.full_session_cost(input_tok, output_tok, cache_write_tok, cache_read_tok)
    split = csm.base_cost_after_split(input_tok, output_tok, cache_write_tok,
                                      cache_read_split) + overhead
    return {
        "as_is_cost": as_is,
        "split_cost": split,
        "dollar_saving": as_is - split,
        "cache_read_before": cache_read_tok,
        "cache_read_after": cache_read_split,
        "token_saving": cache_read_tok - cache_read_split,
    }


def section_h_apply_math_model(long_ids, paths, judge_sample=JUDGE_SAMPLE):
    print("\n" + "=" * 60)
    print("SECTION H — dollar/token savings from splitting (uses math model)")
    print("=" * 60)

    client = OpenAI(base_url=LITELLM_BASE_URL)

    # accumulators for the two split strategies
    f_as_is = f_split = f_tokens = 0.0
    f_count = 0
    g_as_is = g_split = g_tokens = 0.0
    g_count = 0
    n_ok = n_skipped = 0

    ids = long_ids if judge_sample is None else long_ids[:judge_sample]

    for session_id in ids:
        try:
            events = read_transcript(session_id, paths)
            buckets = session_token_buckets(events)
            _, _, _, cache_read_tok, api_calls = buckets
            if cache_read_tok <= 0 or api_calls < 2:
                continue
            n_ok += 1

            # ----- Section F style: split at LLM-detected task switch -----
            prompts, total_turns = user_prompts_with_turns(events)
            if len(prompts) >= 4 and total_turns > 0:
                result = judge_session(client, [text for _, text in prompts])
                if result and "error" not in result and result.get("has_switch"):
                    switch_num = result.get("switch_message_number")
                    if isinstance(switch_num, int) and 1 <= switch_num <= len(prompts):
                        switch_turn = prompts[switch_num - 1][0]
                        frac = switch_turn / total_turns
                        out = saving_for_split(buckets, frac)
                        if out:
                            f_as_is += out["as_is_cost"]
                            f_split += out["split_cost"]
                            f_tokens += out["token_saving"]
                            f_count += 1

            # ----- Section G style: split at end of initial exploration -----
            sequence, seq_turns = extract_tool_sequence(events)
            if len(sequence) >= 10 and seq_turns > 0:
                phases = detect_phases(sequence)
                pattern, expl_phases = classify_session_pattern(phases, sequence)
                if pattern == "A" and expl_phases:
                    # split where the initial exploration phase ends
                    end_turn = expl_phases[0]["end_turn"]
                    frac = end_turn / seq_turns
                    out = saving_for_split(buckets, frac)
                    if out:
                        g_as_is += out["as_is_cost"]
                        g_split += out["split_cost"]
                        g_tokens += out["token_saving"]
                        g_count += 1

        except Exception:
            n_skipped += 1

    print("sessions processed:", n_ok, "| skipped:", n_skipped)
    print(f"(sample = {judge_sample} sessions; Section F uses real API calls)")

    def report(label, count, as_is, split, tokens):
        print(f"\n--- {label} ---")
        print(f"sessions with a usable split: {count}")
        if not count:
            return
        saving = as_is - split
        pct = 100 * saving / as_is if as_is else 0
        print(f"  as-is cost:        ${as_is:,.2f}")
        print(f"  cost after split:  ${split:,.2f}")
        print(f"  dollar saving:     ${saving:,.2f}  ({pct:.0f}%)")
        print(f"  cache-read tokens saved: {tokens/1e6:,.1f}M")

    report("Section F split (at task switch)", f_count, f_as_is, f_split, f_tokens)
    report("Section G split (initial exploration, plan-mode)", g_count, g_as_is, g_split, g_tokens)


# ======================================================================
# main
# ======================================================================

def main():
    sessions, logs = load_tables()
    long_ids = long_session_ids(sessions)
    paths = path_map(logs)

    step_1_2(sessions)
    step_tool_name_counts(long_ids, paths)
    step_3_categories(long_ids, paths)
    step_4_tokens(long_ids, paths)
    step_5_savings(long_ids, paths)

    # LLM-as-judge task-switch detection
    section_f_llm_judge(long_ids, paths)

    # Sequential analysis — detect missed plan-mode opportunities
    section_g_sequential(long_ids, paths)

    # Apply the chunk-split math model to F and G split points -> $ savings
    section_h_apply_math_model(long_ids, paths)


if __name__ == "__main__":
    main()