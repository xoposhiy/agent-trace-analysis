"""
case_file.py — loads SWE-Chat sessions and turns raw Claude Code JSONL events
into a condensed, chronologically-ordered "case file" for the judge.

Runnable standalone to inspect one session's rendered transcript directly:

    python case_file.py <session_id>
    python case_file.py --first        # just grabs the first parseable session

This prints the exact text that would go into a judge prompt's system
message, plus its character count and how long it took to download/parse —
useful for "how big are these traces actually" / "where's the time going"
questions without needing to run the full experiment or call the LLM.

DESIGN NOTES (this revision):
    - No regex, no keyword heuristics anywhere in this file. We do not try
      to decide "is this a test command," "did it pass," or "is this a spec
      file" ourselves — every tool call and its raw result are captured
      verbatim and handed to the judge (in classify.py) to interpret itself.
    - Rendering is inspired by VCC (https://github.com/lllyasviel/VCC) — a
      single chronological, typed-block transcript (one global turn-number
      scheme, blocks for [user] / [assistant thinking] / [tool_call], never
      reordered) rather than several disconnected sections. We use turn
      numbers (already assigned per-assistant-turn) instead of raw line
      numbers, since that's the addressing unit the downstream application
      actually needs (to know "which turn(s) does this symptom occur at"),
      not a byte/line offset into the original file.
    - EVERY tool_use call becomes exactly one timeline entry, and its
      matching tool_result (found later in the event stream) gets patched
      into that same entry once it arrives — nothing is silently dropped,
      and nothing is duplicated.
"""

import json
import time

from huggingface_hub import hf_hub_download
from datasets import load_dataset

REPO = "SALT-NLP/SWE-chat"

SCOPE_FILES_TOO_MANY_THRESHOLD = 8
SCOPE_TURNS_TOO_LONG_THRESHOLD = 150

# Caps that exist ONLY to stop a single pathological session (e.g. a giant
# generated file diff, or a runaway chain-of-thought, or a tool result that
# dumps megabytes of output) from blowing up the prompt. These are not
# content filters — everything under the cap is passed through untouched.
MAX_THINKING_CHARS_PER_BLOCK = 4000
MAX_TOTAL_THINKING_CHARS = 20000
MAX_TOOL_DETAIL_CHARS = 400     # command / file path / generic input summary
MAX_TOOL_RESULT_CHARS = 600     # raw tool_result text, any tool, not just tests
MAX_USER_MESSAGE_CHARS = 2000


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_tables():
    sessions = load_dataset(REPO, "sessions", split="train")
    logs = load_dataset(REPO, "session_logs", split="train")
    return sessions, logs


def claude_code_session_ids(sessions):
    """
    All sessions we can actually parse. Restricted to agent == 'Claude Code'
    because other agents (Codex, Copilot CLI, Cursor, OpenCode, Gemini CLI)
    use different event schemas entirely.
    """
    return [s["session_id"] for s in sessions if s["agent"] == "Claude Code"]


def path_map(logs):
    return {log["session_id"]: log["transcript_path"] for log in logs}


def read_transcript(session_id, paths):
    local_path = hf_hub_download(REPO, paths[session_id], repo_type="dataset")
    events = []
    # Explicit encoding + errors="replace" so odd/invalid bytes in a
    # transcript line get replaced rather than silently corrupting the read
    # or throwing a hard decode error that skips the whole session.
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
    list of structured blocks like [{"type": "text", "text": "..."}]. Doing
    str(content) on the list case stringifies the Python repr (braces,
    quotes, key names and all) instead of the actual text — this avoids that."""
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


def _tool_call_detail(name_lower, tool_input):
    """Short human-readable summary of a tool call's input, generic across
    ALL tool types — no special-casing of 'test' commands or 'spec' files."""
    for key in ("file_path", "path", "pattern"):
        if tool_input.get(key):
            return str(tool_input[key])[:MAX_TOOL_DETAIL_CHARS]
    if "command" in tool_input:
        return str(tool_input["command"])[:MAX_TOOL_DETAIL_CHARS]
    if not tool_input:
        return ""
    return ", ".join(f"{k}={str(v)[:120]}" for k, v in tool_input.items())[:MAX_TOOL_DETAIL_CHARS]


def build_case_file(events):
    """
    Builds a condensed case file. The key structural piece is `timeline`:
    a single list of entries in the STRICT order events actually occurred
    (user message / agent thinking / tool call), each tagged with the
    assistant-turn number it happened at. Every tool_use produces exactly
    one entry; its tool_result (arriving later in the event stream) gets
    patched into that same entry in place once found, so nothing is
    duplicated and nothing is dropped.
    """
    timeline = []              # [{"kind": ..., "turn": int, ...}]
    user_messages = []         # [(text, has_image)]
    files_touched = set()
    tool_call_by_id = {}        # tool_use_id -> the timeline entry dict (mutable, patched in place)
    total_thinking_chars = 0
    thinking_truncated = False
    turn_no = 0

    for event in events:
        etype = event.get("type")
        message = event.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None

        if etype == "assistant":
            turn_no += 1
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue

                    if b.get("type") == "thinking":
                        thinking_text = b.get("thinking", "") or ""
                        if thinking_text and total_thinking_chars < MAX_TOTAL_THINKING_CHARS:
                            snippet = thinking_text[:MAX_THINKING_CHARS_PER_BLOCK]
                            timeline.append({"kind": "thinking", "turn": turn_no, "text": snippet})
                            total_thinking_chars += len(snippet)
                        elif thinking_text:
                            thinking_truncated = True

                    if b.get("type") == "tool_use":
                        name = b.get("name", "") or "(unnamed tool)"
                        tool_input = b.get("input", {}) or {}
                        name_lower = name.lower()

                        if any(k in name_lower for k in ("edit", "write")) and "todowrite" not in name_lower:
                            fp = tool_input.get("file_path")
                            if fp:
                                files_touched.add(fp)

                        entry = {
                            "kind": "tool_call", "turn": turn_no, "id": b.get("id"),
                            "name": name, "detail": _tool_call_detail(name_lower, tool_input),
                            "result": None,
                        }
                        timeline.append(entry)
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
            if text:
                trimmed_text = text[:MAX_USER_MESSAGE_CHARS]
                user_messages.append((trimmed_text, has_image))
                timeline.append({
                    "kind": "user_message", "turn": turn_no, "text": trimmed_text,
                    "has_image": has_image,
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
        "files_touched": sorted(files_touched),
        "thinking_truncated": thinking_truncated,
        "total_turns": turn_no,
    }


def build_case_file_with_timing(session_id, paths):
    """Same as build_case_file, but times the download and parse stages
    separately so callers can report where time is actually going."""
    t0 = time.perf_counter()
    events = read_transcript(session_id, paths)
    t1 = time.perf_counter()
    case_file = build_case_file(events)
    t2 = time.perf_counter()
    return case_file, (t1 - t0), (t2 - t1)


def compute_scope_flags(case_file):
    """Direct metadata check — no LLM needed. Two independent flags."""
    return {
        "scope_files_too_many": len(case_file["files_touched"]) >= SCOPE_FILES_TOO_MANY_THRESHOLD,
        "scope_turns_too_long": case_file["total_turns"] >= SCOPE_TURNS_TOO_LONG_THRESHOLD,
    }


def is_single_message_session(case_file):
    """True if the session has exactly one user message (no back-and-forth
    at all). Purely structural — no text matching involved."""
    return len(case_file["user_messages"]) == 1


# ----------------------------------------------------------------------
# VCC-inspired rendering: one chronological, typed-block transcript
# ----------------------------------------------------------------------

def render_timeline(case_file):
    """Chronological, typed-block rendering (inspired by VCC's "Full View"):
    every user message, every piece of agent thinking, and every tool call
    — including its raw result — in the exact order they occurred, each
    tagged with the turn number it happened at. This is the ONE canonical
    rendering used both for judge prompts and for human-readable dumps."""
    lines = []
    for entry in case_file["timeline"]:
        if entry["kind"] == "user_message":
            tag = " (image attached)" if entry["has_image"] else ""
            lines.append(f"[user] (turn {entry['turn']}){tag}")
            lines.append(entry["text"])
        elif entry["kind"] == "thinking":
            lines.append(f"[assistant thinking] (turn {entry['turn']})")
            lines.append(entry["text"])
        elif entry["kind"] == "tool_call":
            lines.append(f"[tool_call] {entry['name']} (turn {entry['turn']})")
            if entry["detail"]:
                lines.append(f"input: {entry['detail']}")
            lines.append(f"result: {entry['result'] if entry['result'] else '(no result captured)'}")
        lines.append("")
    if case_file["thinking_truncated"]:
        lines.append("(additional thinking beyond the size cap omitted)")
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

    print(f"session_id: {target_id}")
    print(f"download time: {download_s:.2f}s | parse time: {parse_s:.2f}s")
    print(f"user messages: {len(case_file['user_messages'])} | "
          f"files touched: {len(case_file['files_touched'])} | "
          f"total turns: {case_file['total_turns']}")
    print(f"rendered transcript size: {len(rendered):,} chars\n")
    print("=" * 70)
    print(rendered)
    print("=" * 70)