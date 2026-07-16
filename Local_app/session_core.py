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


def build_task_forest_prompt(prompt_texts):
    """Build the LLM-as-judge prompt that segments a session into a HIERARCHICAL
    forest of tasks (pure string; no API call).

    Every user message is assigned a task id. Ids are hierarchical:
      * top-level ids (T1, T2, T3, …) are GENUINELY INDEPENDENT tasks (unrelated
        goals); a top-level id may RECUR (T1 … T2 … T1) when the user returns to it;
      * dotted ids (T1.1, T1.2) are SUB-AGENT children of their parent task — related
        but tangential, self-contained work that could be offloaded to a sub-agent.
    The same call also returns a one-line session summary (no extra request)."""
    numbered = "\n".join(f"{i+1}. {text}" for i, text in enumerate(prompt_texts))
    return (
        "Here are the user's messages from one coding session, in order:\n\n"
        f"{numbered}\n\n"
        "Assign EVERY message to a task, in order, using HIERARCHICAL task ids.\n\n"
        "Top-level tasks — `T1`, `T2`, `T3`, … — are GENUINELY INDEPENDENT goals "
        "(unrelated to each other). Number them in the order they first appear, and "
        "reuse the SAME id when the user RETURNS to that task later (e.g. T1 … T2 … "
        "T1).\n\n"
        "Sub-agent children — `T1.1`, `T1.2`, … — are work that is RELATED to task "
        "`T1` but is a self-contained tangent ('similar but not quite the main goal'): "
        "a big read-only investigation, a side change, a detour. Crucially: if the "
        "user LEAVES a task to do some related work and then COMES BACK to that task, "
        "the in-between work is a CHILD of it, NOT a new independent task — because "
        "returning proves it was a side-quest that a sub-agent could have handled.\n\n"
        "Do NOT create a new task (top-level OR child) for a different PHASE of the "
        "same work: design→plan, plan→implement, implement→test/fix of the SAME "
        "feature, or just different files/words for the same goal.\n\n"
        "Only use a new TOP-LEVEL id when the goal genuinely changes to something "
        "unrelated the user does NOT return from.\n\n"
        "Answer with ONLY a JSON object, no other text:\n"
        '{"tasks": [{"id": "T1", "label": "<short concrete name>"}, '
        '{"id": "T1.1", "label": "<the tangent/side-task>"}, …], '
        '"assignments": ["T1", "T1.1", "T1", "T2", …], '
        '"summary": "<one short, concrete sentence naming the session\'s overall '
        'work>"}\n\n'
        '"assignments" MUST have exactly one id per message, in the same order and '
        "length as the numbered list above, and every id must appear in \"tasks\"."
    )


def build_summary_prompt(prompt_texts):
    """Build a prompt asking ONLY for a one-sentence task summary — used for
    sessions too short for the task-switch judge (pure string; no API call)."""
    numbered = "\n".join(f"{i+1}. {text}" for i, text in enumerate(prompt_texts))
    return (
        "Here are the user's messages from one coding session, in order:\n\n"
        f"{numbered}\n\n"
        "In ONE short, concrete sentence, describe what task the user worked on "
        "in this session (name the feature/bug/goal, not just 'coding').\n\n"
        "Answer with ONLY a JSON object, no other text:\n"
        '{"summary": "<one sentence>"}'
    )


# ======================================================================
# Task forest — timeline, interleaving, boundaries (from judge assignments)
# ======================================================================

def top_level(task_id):
    """The top-level (independent) task an id belongs to: 'T1.2' -> 'T1', 'T1' -> 'T1'."""
    return str(task_id).split(".", 1)[0]


def is_subtask(task_id):
    """True if the id is a sub-agent child (dotted), e.g. 'T1.2'."""
    return "." in str(task_id)


def compress_timeline(assignments):
    """Collapse a per-message task-id list into consecutive runs.

    `assignments` = ["T1","T1","T2","T1"] -> [("T1",1,2),("T2",3,3),("T1",4,4)],
    each run being (task_id, start_message, end_message) with 1-based message
    numbers. This is what renders the forest string T1 -> T2 -> T1.
    """
    runs = []
    for i, tid in enumerate(assignments, start=1):
        if runs and runs[-1][0] == tid:
            runs[-1][2] = i
        else:
            runs.append([tid, i, i])
    return [tuple(run) for run in runs]


def detect_interleaving(assignments):
    """Summarise the forest's shape from its per-message assignments.

    Returns {"runs", "num_switches", "recurring"}:
      - runs: the compressed timeline (see compress_timeline)
      - num_switches: number of task changes (transitions between runs)
      - recurring: sorted TOP-LEVEL task ids the user returned to (appear in more than
        one run once child ids are folded into their parent) — the interleaving signal.
        Children (T1.1) don't count; returning to a main task T1 does.
    """
    runs = compress_timeline(assignments)
    # collapse consecutive runs that share a top-level id (T1 -> T1.1 -> T1 is still
    # "in T1"), then count how many distinct top-level stretches each main task has.
    counts, prev = {}, None
    for tid, _, _ in runs:
        tl = top_level(tid)
        if tl != prev:
            counts[tl] = counts.get(tl, 0) + 1
        prev = tl
    return {
        "runs": runs,
        "num_switches": max(len(runs) - 1, 0),
        "recurring": sorted(tl for tl, n in counts.items() if n > 1),
    }


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


def saving_for_multi_split(buckets, fractions, alpha=SUMMARY_ALPHA):
    """Price splitting a session into k+1 chunks at ALL of `fractions` at once
    (the "split the whole forest into separate sessions" headline).

    Generalises saving_for_split to many cut points: the chunk_split_model
    primitives already accept arbitrary chunk heights, so we build the height
    list from the (deduped, sorted) fractions along the linear context ramp and
    charge a carried summary into every chunk after the first (sized off the
    PRECEDING chunk, matching the single-split convention). Returns the same dict
    shape as saving_for_split (plus "num_chunks"), or None if it can't be priced.
    """
    input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls = buckets
    if cache_read_tok <= 0 or api_calls < 2:
        return None

    # keep interior cuts only, dedup (a rounded key avoids zero-width chunks)
    seen, fracs = set(), []
    for f in sorted(fractions):
        if not (0.05 < f < 0.95):
            continue
        key = round(f, 4)
        if key not in seen:
            seen.add(key)
            fracs.append(f)
    if not fracs:
        return None

    peak = csm.peak_context_tokens(cache_read_tok, api_calls)
    if peak <= 0:
        return None

    cuts = [0.0] + [peak * f for f in fracs] + [peak]
    heights = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
    if any(h <= 0 for h in heights):
        return None

    calls = csm.calls_per_chunk(heights, api_calls, peak)
    cache_read_split = csm.cache_read_after_split(heights, api_calls, peak)

    overhead = 0.0
    for j in range(1, len(heights)):
        summary_tok = csm.summary_tokens(heights[j - 1], alpha)
        oh, *_ = csm.summary_overhead_cost([calls[j]], summary_tok)
        overhead += oh

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
        "num_chunks": len(heights),
    }


def saving_for_subagent(buckets, f_start, f_end, alpha=SUMMARY_ALPHA):
    """Price offloading the segment occupying turns [f_start, f_end] to a SUB-AGENT.

    The whole COST model lives in the math file (csm.subagent_cost): the segment runs
    isolated, the main thread continues afterwards dropping the segment's tokens, and
    the sub-agent's output is carried back and re-read to the end. This function only
    reads the buckets, guards the degenerate cases, and packages the saving dict (same
    shape as saving_for_split). Returns None if unpriceable.
    """
    input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls = buckets
    if cache_read_tok <= 0 or api_calls < 2:
        return None
    if not (0.0 <= f_start < f_end <= 1.0):
        return None
    peak = csm.peak_context_tokens(cache_read_tok, api_calls)
    if peak <= 0 or (f_end - f_start) * peak <= 0:
        return None

    as_is = csm.full_session_cost(input_tok, output_tok, cache_write_tok, cache_read_tok)
    split, cache_read_after = csm.subagent_cost(
        input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls,
        f_start, f_end, alpha)
    return {
        "as_is_cost": as_is,
        "split_cost": split,
        "dollar_saving": as_is - split,
        "cache_read_before": cache_read_tok,
        "cache_read_after": cache_read_after,
        "token_saving": cache_read_tok - cache_read_after,
    }
