"""
session_core.py — dependency-light analysis core shared by the dataset script
(`main.py`) and the local CLI (`split_advisor.py`).

Everything here is PURE: it takes a parsed transcript (a list of JSONL events, the
Claude Code shape) and returns numbers / structures. It imports only the stdlib and
the pricing math in `chunk_split_model.py` — NO `datasets` / `huggingface_hub` /
`openai` — so it can run fully offline against local `~/.claude/projects/**/*.jsonl`
transcripts as well as against the SWE-chat dataset.

What lives here:
  - tool-call categorisation (exploration / editing / execution / coordination),
    including the "bash grep is really reading" rule
  - a rough token estimator for tool results
  - ordered tool-call sequence extraction + phase detection + A/B/C pattern
    classification (the "front-loaded reading = plan-mode opportunity" detector)
  - the per-session billed-token buckets reader
  - `saving_for_split`: price a single semantic split via `chunk_split_model`

Both callers parse a transcript into `events` their own way (HF download vs local
file) and then hand the events to these functions.
"""
import json

import chunk_split_model as csm


# ======================================================================
# Tool categorisation
# ======================================================================

# Bash commands that only READ (don't change anything) — used to tell
# "bash grep" or "bash cat" apart from a real build/test/execute command.
READ_ONLY_BASH_COMMANDS = {
    "cat", "grep", "find", "ls", "head", "tail", "wc",
    "less", "more", "tree", "pwd", "which", "file", "diff",
}


def is_read_only_bash_command(tool_input):
    """
    A Bash call can run a read-only command (cat, grep, find, ls, ...),
    which is really exploration, not execution — or a real build/test/git
    command, which is execution. Judging by the tool NAME alone ("Bash")
    can't tell these apart, so this looks at the actual command text.
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
    """Whether a tool call counts as exploration. Reuses categorize()."""
    return categorize(tool_name, tool_input) == "exploration"


def estimate_tokens(value):
    """Rough token estimate for any tool result: characters / 4."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value) / 4
    return len(json.dumps(value)) / 4


# ======================================================================
# User prompts (for LLM-judge task-switch detection)
# ======================================================================

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
    """Build the LLM-as-judge prompt asking whether the user switches to a
    clearly unrelated task at any point (pure string; no API call)."""
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


# ======================================================================
# Ordered tool-call sequence + phase detection + pattern classification
# ======================================================================

# Tunable: how many consecutive calls of one category count as a "phase"
MIN_PHASE_LENGTH = 5           # at least this many calls to be a real phase
EXPLORATION_DOMINANCE = 0.70   # a phase is "exploration" if >= 70% of its calls are exploration


def extract_tool_sequence(events):
    """
    Walk a transcript in order and return:
      - sequence: [{turn, category, tool_name, tool_id, result_tokens}, ...]
        one entry per tool call, in order
      - total_turns: total assistant turns

    This is the raw, ordered list that the phase detector segments.
    """
    sequence = []
    turn_no = 0
    id_to_call = {}     # tool_use id -> (category, turn_number)

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
                    for entry in reversed(sequence):
                        if entry["tool_id"] == tool_use_id:
                            entry["result_tokens"] = tokens
                            break

    return sequence, turn_no


def detect_phases(sequence, min_length=MIN_PHASE_LENGTH):
    """
    Segment the ordered tool-call sequence into consecutive phases. A phase
    is a maximal run of the same category with length >= min_length.
    Returns a list of phase dicts:
      {"category", "start_idx", "end_idx", "length", "total_tokens",
       "start_turn", "end_turn"}
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
        later_phases = [p for p in phases
                        if p["start_idx"] > first_expl["end_idx"]
                        and p["category"] in ("editing", "execution")]
        if later_phases:
            return "A", expl_phases

    # Pattern C: an exploration phase appears AFTER editing/execution started
    edit_exec_phases = [p for p in phases if p["category"] in ("editing", "execution")]
    if edit_exec_phases:
        first_action = edit_exec_phases[0]
        mid_explorations = [p for p in expl_phases if p["start_idx"] > first_action["end_idx"]]
        if mid_explorations:
            return "C", mid_explorations

    # Pattern B: exploration exists but is interleaved without clean separation
    if expl_phases and edit_exec_phases:
        return "B", expl_phases

    return "none", expl_phases


# ======================================================================
# Billed-token buckets + single-split pricing (uses chunk_split_model)
# ======================================================================

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


def saving_for_split(buckets, split_fraction, alpha=SUMMARY_ALPHA):
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

    summary_tok = csm.summary_tokens(heights[0], alpha)
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
