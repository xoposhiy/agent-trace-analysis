"""
case_file.py — loads SWE-Chat sessions and turns raw Claude Code JSONL events
into a condensed, chronologically-ordered "case file" for the judge.

Runnable standalone to inspect one session's rendered transcript directly:

    python case_file.py <session_id>
    python case_file.py --first        # just grabs the first parseable session

DESIGN NOTES (this revision):
    - No regex, no keyword heuristics anywhere in this file.
    - Rendering is inspired by VCC (https://github.com/lllyasviel/VCC) — a
      single chronological, typed-block transcript, one global numbering
      scheme, never reordered. We use MESSAGE numbers (incrementing once per
      message — every user message AND every assistant message, not just
      assistant turns) as the addressing unit, since that's what the judge
      now cites back in its "location" field. `total_turns` (assistant
      messages only) is tracked separately and unchanged — it only feeds
      the scope_turns_too_long metadata check, which keeps its original
      definition untouched.
    - EVERY tool_use call becomes exactly one timeline entry, and its
      matching tool_result gets patched into that same entry in place once
      found later in the event stream — nothing dropped, nothing duplicated.
    - Thinking capture has no total session-wide cap anymore (we were
      capping total thinking chars per session; that's removed — larger
      context models don't need it). Each individual thinking block is
      still capped at 4000 chars, but instead of just keeping the first
      4000 chars, we now keep the first 2000 AND the last 2000 chars (with
      a note the block was truncated) — the end of a thinking block often
      contains the actual conclusion, which a pure head-cut was losing.
    - A user message with an attachment but NO text (a bare pasted
      screenshot) is kept, not dropped. It counts as a user message and is
      rendered with its "(image attached)" tag, since that attachment is
      precisely the evidence no_visual_reference depends on.
"""

import json
import time

from huggingface_hub import hf_hub_download
from datasets import load_dataset

REPO = "SALT-NLP/SWE-chat"

SCOPE_FILES_TOO_MANY_THRESHOLD = 8
SCOPE_TURNS_TOO_LONG_THRESHOLD = 150

# Per-block cap only — no total/session-wide cap anymore. Exists purely so a
# single pathological thinking block can't blow up the prompt on its own.
MAX_THINKING_CHARS_PER_BLOCK = 4000
THINKING_HEAD_CHARS = 2000
THINKING_TAIL_CHARS = 2000

MAX_TOOL_DETAIL_CHARS = 400     # command / file path / generic input summary
MAX_TOOL_RESULT_CHARS = 600     # raw tool_result text, any tool
MAX_USER_MESSAGE_CHARS = 2000


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


def _truncate_thinking(text):
    """Per-block cap only. If under the cap, return unchanged. If over,
    keep the first THINKING_HEAD_CHARS and last THINKING_TAIL_CHARS (the
    conclusion at the end is often the important part) with a note in
    between, and flag it as truncated so the renderer can tell the judge
    this isn't the full block."""
    if len(text) <= MAX_THINKING_CHARS_PER_BLOCK:
        return text, False
    head = text[:THINKING_HEAD_CHARS]
    tail = text[-THINKING_TAIL_CHARS:]
    combined = (
        f"{head}\n\n"
        f"[... TRUNCATED — {len(text) - THINKING_HEAD_CHARS - THINKING_TAIL_CHARS:,} "
        f"characters omitted from the middle of this thinking block ...]\n\n"
        f"{tail}"
    )
    return combined, True


def build_case_file(events):
    """
    Builds a condensed case file. `timeline` is a single list of entries in
    STRICT event order, each tagged with the MESSAGE number it occurred at
    (message_no increments once per user OR assistant message — every
    message gets a unique number). `total_turns` is tracked separately
    (assistant messages only, unchanged) purely for scope_turns_too_long.
    """
    timeline = []
    user_messages = []         # [(text, has_image)]
    files_touched = set()
    tool_call_by_id = {}
    message_no = 0
    total_turns = 0

    for event in events:
        etype = event.get("type")
        if etype not in ("assistant", "user"):
            continue
        message_no += 1

        message = event.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None

        if etype == "assistant":
            total_turns += 1
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue

                    if b.get("type") == "thinking":
                        thinking_text = b.get("thinking", "") or ""
                        if thinking_text:
                            snippet, truncated = _truncate_thinking(thinking_text)
                            timeline.append({
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
                            "kind": "tool_call", "message": message_no, "id": b.get("id"),
                            "name": name, "detail": _tool_call_detail(tool_input),
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
            # `text or has_image`, not just `text`: a message that is ONLY a
            # pasted screenshot has no text blocks, and dropping it would hide
            # the attachment from the judge — which is exactly the evidence
            # no_visual_reference turns on.
            if text or has_image:
                trimmed_text = text[:MAX_USER_MESSAGE_CHARS]
                user_messages.append((trimmed_text, has_image))
                timeline.append({
                    "kind": "user_message", "message": message_no, "text": trimmed_text,
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
    """Direct metadata check — no LLM needed. Unchanged definitions:
    scope_turns_too_long is still based on assistant-message count
    (total_turns), independent of the message-numbering scheme used for
    judge locations."""
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
    """Chronological, typed-block rendering: every user message, every
    piece of agent thinking, and every tool call — including its raw
    result — in the exact order they occurred, each tagged with the
    message number it happened at."""
    lines = []
    for entry in case_file["timeline"]:
        if entry["kind"] == "user_message":
            tag = " (image attached)" if entry["has_image"] else ""
            lines.append(f"[user] (message {entry['message']}){tag}")
            lines.append(entry["text"] or "(no text in this message)")
        elif entry["kind"] == "thinking":
            trunc_note = (
                " [TRUNCATED — showing the first/last portion only, not the full text]"
                if entry.get("truncated") else ""
            )
            lines.append(f"[assistant thinking] (message {entry['message']}){trunc_note}")
            lines.append(entry["text"])
        elif entry["kind"] == "tool_call":
            lines.append(f"[tool_call] {entry['name']} (message {entry['message']})")
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

    print(f"session_id: {target_id}")
    print(f"download time: {download_s:.2f}s | parse time: {parse_s:.2f}s")
    print(f"user messages: {len(case_file['user_messages'])} | "
          f"files touched: {len(case_file['files_touched'])} | "
          f"total assistant turns: {case_file['total_turns']}")
    print(f"rendered transcript size: {len(rendered):,} chars\n")
    print("=" * 70)
    print(rendered)
    print("=" * 70)