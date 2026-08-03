"""Claude Code adapter: raw ``~/.claude/projects/**/*.jsonl`` -> TraceLens IR.

Layout on disk (verified against real transcripts, 2026-08-03):

    ~/.claude/projects/
      <project-slug>/
        <session-id>.jsonl                       <- main transcript
        <session-id>/subagents/agent-<id>.jsonl  <- one file per subagent

Four pieces of line-handling logic follow Entire CLI's compact transcript
package (``cli/cmd/entire/cli/transcript/compact/compact.go``, MIT licensed,
(c) Entire Inc.), reimplemented here in Python:

  1. merge streamed assistant fragments sharing ``message.id``  (compact.go:432)
  2. inline tool results into their ``tool_use`` block           (compact.go:453)
  3. drop ``thinking`` / ``redacted_thinking`` blocks            (compact.go:685)
  4. enrich results from the per-tool ``toolUseResult`` envelope (compact.go:552)

Everything else — subagent files, ``agent_id``, cache tokens — is ours; Entire's
IR has no representation for it (see DESIGN.md §2).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from Final_app.ir.models import (
    EV_ASSISTANT,
    EV_COMPACTION,
    EV_TOOL_USE,
    EV_USER,
    Event,
    Session,
    Tokens,
    ToolCall,
    ToolResult,
)

VENDOR = "claude-code"

# The tool that spawns a subagent. Claude Code renamed Task -> Agent; both are
# accepted because transcripts on disk predate the rename. Keying on "Task"
# alone finds zero subagents in current transcripts.
SUBAGENT_TOOLS = {"Agent", "Task"}

# Line types that carry no conversational content. Claude Code writes a lot of
# these (attachment alone outnumbers real messages 2:1 in a typical session).
SKIP_TYPES = {
    "attachment",
    "mode",
    "permission-mode",
    "last-prompt",
    "queue-operation",
    "file-history-snapshot",
    "file-history-delta",
    "progress",
    "ai-title",
}


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def projects_root() -> Path:
    """Where Claude Code keeps transcripts. Honours ``CLAUDE_CONFIG_DIR``."""
    base = os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude"))
    return Path(base) / "projects"


# Directories that carry no information about *which* project this is.
_NOISE_PREFIXES = ("Desktop", "Documents", "Downloads", "src", "code", "dev",
                   "work", "projects", "repos", "git")


def unslug_project(slug: str) -> str:
    """Turn ``-Users-me-Desktop-my-repo`` back into something readable.

    The slug is a lossy encoding — Claude Code replaces every ``/`` **and**
    every ``-`` in the real path with ``-``, so ``my-repo`` and ``my/repo``
    encode identically and the original genuinely cannot be recovered. So this
    is a display heuristic, never authoritative: the UI keeps the raw slug in a
    tooltip.

    We strip the home prefix and any leading scaffolding directory, then keep
    what remains. Taking only the final segment loses too much — three separate
    projects here all ended in ``analysis``/``app`` and were indistinguishable
    in the filter.
    """
    cleaned = slug.strip("-")
    if not cleaned:
        return slug

    home = os.path.expanduser("~").lstrip("/").replace("/", "-")
    if cleaned == home:
        return "~"
    if cleaned.startswith(home + "-"):
        cleaned = cleaned[len(home) + 1:]

    parts = cleaned.split("-")
    while len(parts) > 1 and parts[0] in _NOISE_PREFIXES:
        parts = parts[1:]

    return "-".join(parts) or slug


def discover_sessions() -> list[tuple[str, Path]]:
    """Every session transcript on disk, as ``(project_slug, path)``.

    Only top-level ``*.jsonl`` files are sessions; ``<session-id>/subagents/``
    holds children, which are loaded as part of their parent.
    """
    root = projects_root()
    if not root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        for path in sorted(project_dir.glob("*.jsonl")):
            if path.is_file():
                found.append((project_dir.name, path))
    return found


def subagent_files(main_path: Path) -> list[Path]:
    """Subagent transcripts belonging to ``main_path``, if any."""
    sub_dir = main_path.with_suffix("") / "subagents"
    if not sub_dir.is_dir():
        return []
    return sorted(p for p in sub_dir.glob("agent-*.jsonl") if p.is_file())


# ----------------------------------------------------------------------
# Line-level helpers
# ----------------------------------------------------------------------

def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed objects, skipping malformed lines.

    Transcripts are appended to live; the last line of an in-flight session can
    be a partial write. One bad line must not lose the session.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _extract_tokens(usage: Any) -> Tokens:
    if not isinstance(usage, dict):
        return Tokens()
    return Tokens(
        input=int(usage.get("input_tokens") or 0),
        output=int(usage.get("output_tokens") or 0),
        cache_read=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation=int(usage.get("cache_creation_input_tokens") or 0),
    )


def _result_from_envelope(envelope: Any, fallback_output: str, is_error: bool) -> ToolResult:
    """Build a ToolResult from Claude Code's ``toolUseResult``.

    The envelope's shape differs per tool — verified shapes:
      Read   -> {file: {filePath, numLines, ...}, type}
      Edit   -> {filePath, oldString, newString, structuredPatch, ...}
      Bash   -> {stdout, stderr, interrupted, isImage, ...}
      Agent  -> {agentId, description, prompt, status, ...}
      (also a bare string for some tools)
    """
    result = ToolResult(output=fallback_output, is_error=is_error)

    if isinstance(envelope, str):
        result.output = envelope or fallback_output
        return result
    if not isinstance(envelope, dict):
        return result

    # Bash and friends.
    stdout = envelope.get("stdout")
    stderr = envelope.get("stderr")
    if isinstance(stdout, str) or isinstance(stderr, str):
        parts = [p for p in (stdout, stderr) if isinstance(p, str) and p]
        if parts:
            result.output = "\n".join(parts)
        if stderr:
            result.is_error = result.is_error or bool(envelope.get("interrupted"))

    # Read: nested file object.
    file_obj = envelope.get("file")
    if isinstance(file_obj, dict):
        result.file_path = str(file_obj.get("filePath") or "")
        try:
            result.num_lines = int(file_obj.get("numLines") or 0)
        except (TypeError, ValueError):
            result.num_lines = 0

    # Edit/Write: flat filePath.
    if not result.file_path and isinstance(envelope.get("filePath"), str):
        result.file_path = envelope["filePath"]

    # Agent/Task spawn: the child's id. This is the parent -> subagent link.
    if isinstance(envelope.get("agentId"), str):
        result.spawned_agent_id = envelope["agentId"]

    return result


def _tool_results_from_user_line(line: dict) -> dict[str, ToolResult]:
    """Map ``tool_use_id`` -> ToolResult for a user line carrying results.

    Claude Code puts one ``tool_result`` per user line, so the ``toolUseResult``
    envelope (which is per-line, not per-result) applies unambiguously.
    """
    message = line.get("message")
    if not isinstance(message, dict):
        return {}
    content = message.get("content")
    if not isinstance(content, list):
        return {}

    envelope = line.get("toolUseResult")
    out: dict[str, ToolResult] = {}
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tool_use_id = str(block.get("tool_use_id") or "")
        raw = block.get("content")
        if isinstance(raw, list):
            text = "\n".join(
                b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text"
            )
        elif isinstance(raw, str):
            text = raw
        else:
            text = ""
        out[tool_use_id] = _result_from_envelope(
            envelope, text, bool(block.get("is_error"))
        )
    return out


def _user_text(line: dict) -> str:
    message = line.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _is_tool_result_line(line: dict) -> bool:
    message = line.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _is_compaction(line: dict) -> bool:
    if line.get("isCompactSummary"):
        return True
    subtype = line.get("subtype")
    return isinstance(subtype, str) and "compact" in subtype.lower()


# ----------------------------------------------------------------------
# Transcript -> events
# ----------------------------------------------------------------------

def parse_transcript(
    path: Path,
    agent_id: Optional[str] = None,
    depth: int = 0,
) -> tuple[list[Event], dict]:
    """Parse one transcript file into Events plus a metadata bag.

    Returns ``(events, meta)`` where meta carries ``ai_title``, ``git_branch``,
    ``model``, ``vendor_version``, ``user_prompts``, ``compaction_points``, and
    ``spawned`` (a list of ``{agent_id, description}`` for subagents launched
    from this transcript).
    """
    meta: dict = {
        "ai_title": "",
        "git_branch": "",
        "model": "",
        "vendor_version": "",
        "user_prompts": [],
        "compaction_points": [],
        "spawned": [],
    }

    # Pass 1: collect tool results and the ai-title, both of which are needed
    # before (or independently of) the message they attach to.
    results_by_id: dict[str, ToolResult] = {}
    for line in _read_jsonl(path):
        if line.get("type") == "ai-title":
            title = line.get("aiTitle")
            if isinstance(title, str) and title.strip():
                meta["ai_title"] = title.strip()
            continue
        if line.get("type") == "user":
            results_by_id.update(_tool_results_from_user_line(line))
        envelope = line.get("toolUseResult")
        if isinstance(envelope, dict) and envelope.get("agentId"):
            meta["spawned"].append({
                "agent_id": str(envelope["agentId"]),
                "description": str(envelope.get("description") or ""),
            })

    # Pass 2: build events, merging streamed assistant fragments by message id.
    events: list[Event] = []
    # message_id -> index into `events` of that message's prose event, so a
    # later fragment appends to it instead of creating a phantom block.
    prose_index: dict[str, int] = {}
    seen_tool_ids: set[str] = set()

    for line in _read_jsonl(path):
        line_type = line.get("type")
        if line_type in SKIP_TYPES:
            continue

        ts = _parse_ts(line.get("timestamp"))
        if ts is None:
            continue

        if isinstance(line.get("gitBranch"), str) and line["gitBranch"]:
            meta["git_branch"] = line["gitBranch"]
        if isinstance(line.get("version"), str) and line["version"]:
            meta["vendor_version"] = line["version"]

        uuid = str(line.get("uuid") or "")
        parent_uuid = line.get("parentUuid")
        parent_uuid = str(parent_uuid) if isinstance(parent_uuid, str) else None

        if _is_compaction(line):
            meta["compaction_points"].append(ts)
            events.append(Event(
                uuid=uuid, ts=ts, type=EV_COMPACTION, vendor=VENDOR,
                parent_uuid=parent_uuid, agent_id=agent_id, depth=depth,
            ))
            continue

        if line_type == "user":
            # A user line is either a real prompt or a tool-result carrier.
            # Only the former is a "chatting with user" block.
            if _is_tool_result_line(line):
                continue
            text = _user_text(line)
            if not text.strip():
                continue
            meta["user_prompts"].append(text)
            events.append(Event(
                uuid=uuid, ts=ts, type=EV_USER, vendor=VENDOR,
                parent_uuid=parent_uuid, text=text,
                agent_id=agent_id, depth=depth,
            ))
            continue

        if line_type != "assistant":
            continue

        message = line.get("message")
        if not isinstance(message, dict):
            continue

        message_id = str(message.get("id") or "")
        model = str(message.get("model") or "")
        if model:
            meta["model"] = model
        tokens = _extract_tokens(message.get("usage"))

        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            # (3) thinking never reaches the IR.
            if block_type in ("thinking", "redacted_thinking"):
                continue

            if block_type == "text":
                text = block.get("text") or ""
                if not text.strip():
                    continue
                # (1) merge streamed fragments sharing a message id.
                idx = prose_index.get(message_id)
                if message_id and idx is not None:
                    existing = events[idx]
                    existing.text += text
                    existing.ts = ts
                    if tokens.total:
                        existing.tokens = tokens  # cumulative; last wins
                    continue
                events.append(Event(
                    uuid=uuid, ts=ts, type=EV_ASSISTANT, vendor=VENDOR,
                    parent_uuid=parent_uuid, message_id=message_id,
                    model=model, tokens=tokens, text=text,
                    agent_id=agent_id, depth=depth,
                ))
                if message_id:
                    prose_index[message_id] = len(events) - 1
                continue

            if block_type == "tool_use":
                tool_id = str(block.get("id") or "")
                # A re-sent fragment can repeat a tool_use block verbatim.
                if tool_id and tool_id in seen_tool_ids:
                    continue
                if tool_id:
                    seen_tool_ids.add(tool_id)
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    tool_input = {}
                events.append(Event(
                    uuid=uuid, ts=ts, type=EV_TOOL_USE, vendor=VENDOR,
                    parent_uuid=parent_uuid, message_id=message_id,
                    model=model,
                    # Tokens belong to the message, not the call. Attributing
                    # them to prose (above) keeps session totals exact; a tool
                    # call carries none of its own.
                    tokens=Tokens(),
                    tool=ToolCall(
                        id=tool_id,
                        name=str(block.get("name") or ""),
                        input=tool_input,
                        # (2) inline the result onto the call.
                        result=results_by_id.get(tool_id),
                    ),
                    agent_id=agent_id, depth=depth,
                ))

    events.sort(key=lambda e: e.ts)
    return events, meta


# ----------------------------------------------------------------------
# Session assembly
# ----------------------------------------------------------------------

def load_session(project_slug: str, path: Path, with_subagents: bool = True) -> Optional[Session]:
    """Load one session, including its subagent transcripts."""
    events, meta = parse_transcript(path)

    subagent_ids: list[str] = []
    if with_subagents:
        described = {s["agent_id"]: s["description"] for s in meta["spawned"]}
        for sub_path in subagent_files(path):
            # agent-<id>.jsonl
            agent_id = sub_path.stem[len("agent-"):]
            sub_events, sub_meta = parse_transcript(sub_path, agent_id=agent_id, depth=1)
            if not sub_events:
                continue
            subagent_ids.append(agent_id)
            events.extend(sub_events)
            if not meta["model"] and sub_meta["model"]:
                meta["model"] = sub_meta["model"]
            described.setdefault(agent_id, "")
        events.sort(key=lambda e: e.ts)

    if not events:
        return None

    session = Session(
        session_id=path.stem,
        project=project_slug,
        path=str(path),
        git_branch=meta["git_branch"],
        model=meta["model"],
        vendor=VENDOR,
        vendor_version=meta["vendor_version"],
        title=meta["ai_title"],
        first_ts=events[0].ts,
        last_ts=events[-1].ts,
        events=events,
        compaction_points=meta["compaction_points"],
        user_prompts=meta["user_prompts"],
        subagent_ids=subagent_ids,
    )
    return session


def load_all_sessions() -> list[Session]:
    """Every session on disk, newest last activity first."""
    sessions: list[Session] = []
    for project_slug, path in discover_sessions():
        try:
            session = load_session(project_slug, path)
        except Exception:
            # One unparseable transcript must not take down the whole list.
            continue
        if session is not None:
            sessions.append(session)
    sessions.sort(
        key=lambda s: s.last_ts or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return sessions
