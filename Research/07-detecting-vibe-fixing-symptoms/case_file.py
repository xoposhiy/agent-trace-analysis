"""
case_file.py — loads SWE-Chat sessions and turns raw Claude Code JSONL events
into a condensed, chronologically-ordered "case file" for the judge.

Runnable standalone to inspect one session's rendered transcript:

    python case_file.py <session_id>
    python case_file.py --first        # first parseable session

Key decisions:
    - No regex, no keyword heuristics. The literal harness tags in
      HARNESS_TAGS are a syntactic test, not a semantic one.
    - Rendering follows VCC (https://github.com/lllyasviel/VCC): one
      chronological, typed-block transcript, never reordered. Every block is
      addressed by a two-part coordinate `P.S` — prompt number, step inside
      the prompt. `total_turns` (assistant messages only) is separate and
      feeds only the scope_turns_too_long check.
    - Roughly 40% of text-carrying user events are harness wrappers — slash
      commands, compaction summaries, interruption markers. They are
      classified as `system_event` and kept out of `user_messages`, the
      denominator of every "% of prompts" rate. They are still rendered in
      full and labelled: a `/clear` explains why the context forgot
      everything, an interruption is quotable evidence.
    - Numbering is PURELY POSITIONAL: the counter advances on every
      text-carrying user event, real or wrapper, so numbering has gaps
      between real prompts and changing _is_system_event never renumbers a
      session.
    - Every tool_use becomes exactly one timeline entry; its tool_result is
      patched into that entry once found later in the stream.
    - Thinking and assistant replies are capped PER BLOCK (no session-wide
      cap), keeping head AND tail — the conclusion sits at the end.
    - A user message with an attachment but no text is kept: that attachment
      is the evidence no_visual_reference depends on.
    - Assistant text is rendered as "[reply-to-user P.S]", a distinct block
      type from thinking. Three subcategories are defined as things the agent
      says out loud, and reasoning the human never saw is not a statement
      made to the human.
    - The repo-root .env is loaded at import time, so HF_TOKEN and
      OPENAI_API_KEY are available to every script importing this module.
"""

import json
import time
from collections import Counter

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# Credentials live in a gitignored .env at the repo root — HF_TOKEN here,
# OPENAI_API_KEY for the judge in classify.py. Loaded in this module because
# every entry point imports it. find_dotenv walks up from THIS file, not the
# cwd, so it resolves whatever directory the script is launched from.
load_dotenv(find_dotenv(usecwd=False))

REPO = "SALT-NLP/SWE-chat"

SCOPE_FILES_TOO_MANY_THRESHOLD = 8
SCOPE_TURNS_TOO_LONG_THRESHOLD = 150

# Per-block cap only, so one pathological block can't blow up the prompt.
MAX_THINKING_CHARS_PER_BLOCK = 4000
THINKING_HEAD_CHARS = 2000
THINKING_TAIL_CHARS = 2000

# Same for what the agent writes BACK to the user. Three subcategories are
# defined as things the agent says out loud, and none of that lives in a
# thinking block — unrendered, those checks would be judging silence.
MAX_ASSISTANT_TEXT_CHARS = 4000
ASSISTANT_TEXT_HEAD_CHARS = 2000
ASSISTANT_TEXT_TAIL_CHARS = 2000

MAX_TOOL_DETAIL_CHARS = 400     # command / file path / generic input summary
MAX_TOOL_RESULT_CHARS = 600     # raw tool_result text, any tool
# Applies to system_event blocks too: their content is rendered unchanged,
# just capped like any user message.
MAX_USER_MESSAGE_CHARS = 2000


# ----------------------------------------------------------------------
# Real prompts vs. harness wrappers
# ----------------------------------------------------------------------

# Structural fields the harness sets on an event it generated itself.
# isVisibleInTranscriptOnly is excluded: all 25 of its occurrences across the
# 111 cached transcripts also carry isCompactSummary, so it filters nothing.
SYSTEM_EVENT_FLAGS = ("isMeta", "isCompactSummary")

# Literal markup the Claude Code harness wraps around its own machinery. Not a
# keyword heuristic: "did the harness emit this literal tag?" is a syntactic
# yes/no, unlike the semantic guesses (is this file a spec? is this call a
# test?) the no-heuristics rule bans and the judge exists to make.
HARNESS_TAGS = (
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "[Request interrupted by user",
)

# Scanned as a WINDOW over the start of the text, not as a prefix test: the
# tags arrive in pairs in arbitrary order and <command-name> is regularly the
# second one, e.g.
#   <command-message>bmad-bmb-edit-module</command-message> <command-name>/bmad-...
# 200 chars covers the pair, and is short enough that a real prompt quoting a
# tag further down still counts as real.
HARNESS_TAG_SCAN_CHARS = 200


def _is_system_event(event, text):
    """True when this text-carrying user event is harness machinery rather
    than a request from the human. Two independent signals, either sufficient:
    a structural flag (SYSTEM_EVENT_FLAGS), or a literal tag near the start of
    the text (HARNESS_TAGS). Deliberately NOT used: promptId — present on only
    70 of 460 text user messages, since older Claude Code versions lack it."""
    for flag in SYSTEM_EVENT_FLAGS:
        if event.get(flag):
            return True
    head = text[:HARNESS_TAG_SCAN_CHARS]
    return any(tag in head for tag in HARNESS_TAGS)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_tables():
    sessions = load_dataset(REPO, "sessions", split="train")
    logs = load_dataset(REPO, "session_logs", split="train")
    return sessions, logs


def claude_code_session_ids(sessions):
    """All sessions we can actually parse. Restricted to agent == 'Claude
    Code' because other agents use different event schemas entirely."""
    return [s["session_id"] for s in sessions if s["agent"] == "Claude Code"]


def path_map(logs):
    return {log["session_id"]: log["transcript_path"] for log in logs}


def read_transcript(session_id, paths):
    local_path = hf_hub_download(REPO, paths[session_id], repo_type="dataset")
    events = []
    with open(local_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


# ----------------------------------------------------------------------
# Case-file extraction
# ----------------------------------------------------------------------

def _extract_text_from_content(content):
    """Content on a tool_result / message block can be a plain string OR a
    list of structured blocks like [{"type": "text", "text": "..."}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif isinstance(b.get("content"), str):
                    parts.append(b["content"])
        return "\n".join(parts)
    return str(content) if content else ""


def _tool_call_detail(tool_input):
    """Short human-readable summary of a tool call's input, generic across
    ALL tool types."""
    for key in ("file_path", "path", "pattern"):
        if tool_input.get(key):
            return str(tool_input[key])[:MAX_TOOL_DETAIL_CHARS]
    if "command" in tool_input:
        return str(tool_input["command"])[:MAX_TOOL_DETAIL_CHARS]
    if not tool_input:
        return ""
    return ", ".join(f"{k}={str(v)[:120]}" for k, v in tool_input.items())[:MAX_TOOL_DETAIL_CHARS]


def _truncate_middle(text, max_chars, head_chars, tail_chars, label):
    """Per-block cap. Under it, unchanged; over it, keep head_chars and
    tail_chars with a note in between, flagged as truncated.

    Shared by thinking and assistant replies: both end in the payload — a
    decision, or "done, here's what I changed" — so a pure head-cut loses
    exactly the part the verification checks turn on."""
    if len(text) <= max_chars:
        return text, False
    combined = (
        f"{text[:head_chars]}\n\n"
        f"[... TRUNCATED — {len(text) - head_chars - tail_chars:,} "
        f"characters omitted from the middle of this {label} ...]\n\n"
        f"{text[-tail_chars:]}"
    )
    return combined, True


def _truncate_thinking(text):
    return _truncate_middle(
        text, MAX_THINKING_CHARS_PER_BLOCK,
        THINKING_HEAD_CHARS, THINKING_TAIL_CHARS, "thinking block",
    )


def _truncate_assistant_text(text):
    return _truncate_middle(
        text, MAX_ASSISTANT_TEXT_CHARS,
        ASSISTANT_TEXT_HEAD_CHARS, ASSISTANT_TEXT_TAIL_CHARS, "assistant reply",
    )


def build_case_file(events):
    """
    Builds a condensed case file. `timeline` is a single list of entries in
    STRICT event order. Each entry carries THREE numbers:

      - "prompt": which user prompt's work this entry belongs to. Increments
        on every text-carrying user event, real prompt or harness wrapper
        alike; assistant/tool entries inherit the most recent one. This is the
        unit the judge cites and the statistics are computed over.
      - "step": position INSIDE that prompt, starting at 1 on the opening
        [USER ...] / [SYSTEM ...] block. Stays in the 1..50 range where a
        global message index runs to 3000, which is what makes it citable.
      - "message": the raw message index, for jumping to the exact event.
        Deliberately kept out of the judge's view — two unrelated numbers per
        block is what produced the earlier soup of free-text locations
        ("messages 322, 351-322").

    Entries before the first user event get prompt 0, which is not addressable
    and is excluded from `prompts`. `total_turns` (assistant messages only)
    feeds scope_turns_too_long.

    Returns, beyond the timeline: `user_messages` (REAL prompts only — the
    denominator), `system_events`, `max_prompt_number` (the range a cited
    prompt number is validated against), and `prompts`, a
    {number: {"kind", "steps"}} index classify.py's validator uses to tell a
    real prompt from a wrapper and to bound a cited step.
    """
    timeline = []
    user_messages = []         # [(text, has_image)] — REAL prompts only
    files_touched = set()
    tool_call_by_id = {}
    prompts = {}               # prompt number -> {"kind": ..., "steps": ...}
    message_no = 0
    prompt_no = 0
    step_no = 1
    system_events = 0
    total_turns = 0

    def add(entry):
        """Stamp the current coordinate onto an entry and file it. Single
        choke point, so the per-prompt step ceiling stays in sync with what
        was actually emitted."""
        nonlocal step_no
        entry["prompt"] = prompt_no
        entry["step"] = step_no
        timeline.append(entry)
        if prompt_no in prompts:
            prompts[prompt_no]["steps"] = step_no
        step_no += 1

    for event in events:
        etype = event.get("type")
        if etype not in ("assistant", "user"):
            continue
        message_no += 1

        message = event.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None

        if etype == "assistant":
            total_turns += 1
            # An assistant message whose content is a bare string is all reply
            # text — no blocks to walk.
            if isinstance(content, str) and content.strip():
                snippet, truncated = _truncate_assistant_text(content.strip())
                add({
                    "kind": "assistant_text", "message": message_no,
                    "text": snippet, "truncated": truncated,
                })
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue

                    if b.get("type") == "text":
                        reply_text = (b.get("text", "") or "").strip()
                        if reply_text:
                            snippet, truncated = _truncate_assistant_text(reply_text)
                            add({
                                "kind": "assistant_text", "message": message_no,
                                "text": snippet, "truncated": truncated,
                            })

                    if b.get("type") == "thinking":
                        thinking_text = b.get("thinking", "") or ""
                        if thinking_text:
                            snippet, truncated = _truncate_thinking(thinking_text)
                            add({
                                "kind": "thinking", "message": message_no,
                                "text": snippet, "truncated": truncated,
                            })

                    if b.get("type") == "tool_use":
                        name = b.get("name", "") or "(unnamed tool)"
                        tool_input = b.get("input", {}) or {}
                        name_lower = name.lower()

                        if any(k in name_lower for k in ("edit", "write")) and "todowrite" not in name_lower:
                            fp = tool_input.get("file_path")
                            if fp:
                                files_touched.add(fp)

                        entry = {
                            "kind": "tool_call", "message": message_no,
                            "id": b.get("id"),
                            "name": name, "detail": _tool_call_detail(tool_input),
                            "result": None,
                        }
                        add(entry)
                        if entry["id"]:
                            tool_call_by_id[entry["id"]] = entry

        if etype == "user":
            has_image = False
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                has_image = any(isinstance(b, dict) and b.get("type") == "image" for b in content)
                text = " ".join(parts).strip()
            else:
                text = ""
            # `text or has_image`: a message that is ONLY a pasted screenshot
            # has no text blocks, and dropping it would hide the attachment
            # no_visual_reference turns on.
            if text or has_image:
                trimmed_text = text[:MAX_USER_MESSAGE_CHARS]
                # One counter for both kinds — the number is positional. What
                # the event IS shows up only in the rendered header.
                prompt_no += 1
                step_no = 1
                is_system = _is_system_event(event, text)
                prompts[prompt_no] = {
                    "kind": "system" if is_system else "user", "steps": 1,
                }
                if is_system:
                    system_events += 1
                    add({
                        "kind": "system_event", "message": message_no,
                        "text": trimmed_text, "has_image": has_image,
                    })
                else:
                    user_messages.append((trimmed_text, has_image))
                    add({
                        "kind": "user_message", "message": message_no,
                        "text": trimmed_text, "has_image": has_image,
                    })

            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        entry = tool_call_by_id.get(b.get("tool_use_id"))
                        if entry is not None and entry["result"] is None:
                            result_text = _extract_text_from_content(b.get("content", ""))
                            entry["result"] = result_text[:MAX_TOOL_RESULT_CHARS]

    return {
        "timeline": timeline,
        "user_messages": user_messages,
        "system_events": system_events,
        "prompts": prompts,
        "max_prompt_number": prompt_no,
        "files_touched": sorted(files_touched),
        "total_turns": total_turns,
    }


def build_case_file_with_timing(session_id, paths):
    t0 = time.perf_counter()
    events = read_transcript(session_id, paths)
    t1 = time.perf_counter()
    case_file = build_case_file(events)
    t2 = time.perf_counter()
    return case_file, (t1 - t0), (t2 - t1)


def compute_scope_flags(case_file):
    """Direct metadata check — no LLM. scope_turns_too_long is based on
    assistant-message count (total_turns), independent of the numbering
    scheme used for judge locations."""
    return {
        "scope_files_too_many": len(case_file["files_touched"]) >= SCOPE_FILES_TOO_MANY_THRESHOLD,
        "scope_turns_too_long": case_file["total_turns"] >= SCOPE_TURNS_TOO_LONG_THRESHOLD,
    }


def timeline_stats(case_file):
    """How big this session is, counted the way the judge sees it. The
    several "sizes" are NOT interchangeable:
      - user_prompts_real and system_events stay SEPARATE, never summed. Only
        the first is a denominator.
      - max_prompt_number is the highest coordinate handed out over both
        kinds, and it — not user_prompts_real — is the range a cited prompt
        number has to fall inside.
      - assistant_turns is assistant MESSAGES (what scope_turns_too_long
        thresholds on), not exchanges: 8 user messages can mean 150 turns.
      - timeline_blocks is how many rendered blocks the judge sees, smaller
        than the message-number space because message numbers also advance on
        events that render nothing.
    """
    counts = Counter(entry["kind"] for entry in case_file["timeline"])
    return {
        "user_prompts_real": counts.get("user_message", 0),
        "system_events": counts.get("system_event", 0),
        "max_prompt_number": case_file["max_prompt_number"],
        "assistant_replies": counts.get("assistant_text", 0),
        "thinking_blocks": counts.get("thinking", 0),
        "tool_calls": counts.get("tool_call", 0),
        "timeline_blocks": len(case_file["timeline"]),
        "assistant_turns": case_file["total_turns"],
        "files_touched": len(case_file["files_touched"]),
    }


# ----------------------------------------------------------------------
# VCC-inspired rendering: one chronological, typed-block transcript
# ----------------------------------------------------------------------

def _coord(entry):
    """The `P.S` coordinate printed in every header. One place, so the two
    numbers can never disagree between block types."""
    return f"{entry['prompt']}.{entry['step']}"


# Header suffixes are terse on purpose: one bit of information each, repeated
# on hundreds of blocks, on a header that has to stay scannable.
TRUNCATED_TAG = " (truncated)"
IMAGE_TAG = " (image)"


def render_timeline(case_file):
    """Chronological, typed-block rendering: every user message, harness
    wrapper, agent reply, thinking block and tool call — with its raw result —
    in the exact order they occurred.

    Every header has ONE shape: `[<kind> <prompt>.<step> <detail>]`, so the
    coordinate always sits in the same place whatever the block type.

    USER and SYSTEM are the only kinds in capitals: they are the only two that
    open a new numbered section, and prompt boundaries have to be findable by
    eye in an unbroken wall of text."""
    lines = []
    for entry in case_file["timeline"]:
        kind = entry["kind"]
        if kind == "user_message":
            tag = IMAGE_TAG if entry["has_image"] else ""
            lines.append(f"[USER {_coord(entry)}]{tag}")
            lines.append(entry["text"] or "(no text in this message)")
        elif kind == "system_event":
            # Rendered in full, exactly like a user message. The label is the
            # whole difference: tooling output, no request from the human, but
            # still quotable as evidence.
            tag = IMAGE_TAG if entry["has_image"] else ""
            lines.append(f"[SYSTEM {_coord(entry)}]{tag}")
            lines.append(entry["text"] or "(no text in this message)")
        elif kind == "thinking":
            trunc_note = TRUNCATED_TAG if entry.get("truncated") else ""
            lines.append(f"[thinking {_coord(entry)}]{trunc_note}")
            lines.append(entry["text"])
        elif kind == "assistant_text":
            trunc_note = TRUNCATED_TAG if entry.get("truncated") else ""
            # "reply-to-user", not "reply", so the judge can tell it apart
            # from thinking: ask-for-manual-testing turns on exactly that
            # distinction.
            lines.append(f"[reply-to-user {_coord(entry)}]{trunc_note}")
            lines.append(entry["text"])
        elif kind == "tool_call":
            lines.append(f"[tool {_coord(entry)} {entry['name']}]")
            if entry["detail"]:
                lines.append(f"input: {entry['detail']}")
            lines.append(f"result: {entry['result'] if entry['result'] else '(no result captured)'}")
        lines.append("")
    return "\n".join(lines).strip() if lines else "(empty session)"


# ----------------------------------------------------------------------
# Standalone inspection mode
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading dataset tables from HuggingFace...")
    sessions, logs = load_tables()
    ids = claude_code_session_ids(sessions)
    paths = path_map(logs)
    print(f"{len(ids):,} Claude Code sessions available.\n")

    if len(sys.argv) > 1 and sys.argv[1] != "--first":
        target_id = sys.argv[1]
    else:
        target_id = ids[0]
        print(f"(no session id given, using the first one: {target_id})\n")

    case_file, download_s, parse_s = build_case_file_with_timing(target_id, paths)
    rendered = render_timeline(case_file)

    n_real = len(case_file["user_messages"])
    n_system = case_file["system_events"]
    n_text_events = n_real + n_system
    filtered_pct = (100 * n_system / n_text_events) if n_text_events else 0.0

    print(f"session_id: {target_id}")
    print(f"download time: {download_s:.2f}s | parse time: {parse_s:.2f}s")
    print(f"real user prompts: {n_real} | system events: {n_system} "
          f"({filtered_pct:.1f}% of {n_text_events} text user events filtered) | "
          f"numbering runs up to: {case_file['max_prompt_number']}")
    print(f"files touched: {len(case_file['files_touched'])} | "
          f"total assistant turns: {case_file['total_turns']}")
    print(f"rendered transcript size: {len(rendered):,} chars\n")
    print("=" * 70)
    print(rendered)
    print("=" * 70)