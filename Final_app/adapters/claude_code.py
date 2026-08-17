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

    The slug is a lossy encoding — Claude Code replaces every ``/``, ``_`` **and**
    ``-`` in the real path with ``-``, so ``my-repo``, ``my_repo`` and ``my/repo``
    all encode identically and the original genuinely cannot be recovered.
    Verified against this machine (2026-08-17): the directory
    ``~/Desktop/Agent_traces/agent-trace-analysis/Final_app`` is stored as
    ``-Users-andreistan-Desktop-Agent-traces-agent-trace-analysis-Final-app``,
    in which one ``-`` was a separator, one an underscore and one a real hyphen.
    So this is a display heuristic, never authoritative: the UI keeps the raw
    slug in a tooltip.

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


# ----------------------------------------------------------------------
# Token attribution
# ----------------------------------------------------------------------

# ``usage`` is reported once per assistant *message*, but a message becomes
# several Events (prose plus one per tool call). Attributing the whole thing to
# the prose Event — as this adapter did until 2026-08-05 — has two consequences,
# both measured against a real 259-message transcript
# (51db4d3e, ~/.claude/projects/...-Local-app):
#
#   * every tool call reported exactly 0 tokens, so the bar's "tokens" Y-axis
#     sized read/write/execute blocks at nothing unless prose happened to be
#     merged into them;
#   * 105 of the 259 messages emitted tool calls with no prose at all, and their
#     usage had nowhere to land, so it was dropped: 476,169 of 853,096 working
#     tokens (55.8%) never reached the IR.
#
# So usage is now spread over the Events the message actually produced.


def _content_weight(text: str) -> int:
    """How much of a message's output a piece of content accounts for.

    Character count, not a tokeniser: this only has to divide one already-known
    output figure between siblings, and a tokeniser would be a dependency and a
    per-line cost for a number that is a proportion either way.
    """
    return max(1, len(text))


def _split_by_weight(amount: int, weights: list[int]) -> list[int]:
    """Divide ``amount`` across ``weights``, exactly — no token invented or lost.

    Largest-remainder: floor every share, then hand the rounding shortfall to
    the largest fractional parts. Plain rounding would drift, and over a
    300-message session the drift is what makes a session total stop matching
    the sum of its blocks.
    """
    if not weights:
        return []
    total = sum(weights)
    if total <= 0 or amount <= 0:
        # Nothing to weigh by (or nothing to give): spread evenly so the sum
        # still holds exactly.
        base, extra = divmod(max(0, amount), len(weights))
        return [base + (1 if i < extra else 0) for i in range(len(weights))]

    shares = [amount * weight // total for weight in weights]
    remainders = sorted(
        range(len(weights)),
        key=lambda i: (amount * weights[i]) % total,
        reverse=True,
    )
    for i in remainders[: amount - sum(shares)]:
        shares[i] += 1
    return shares


def _attribute_tokens(
    events: list[Event],
    message_events: dict[str, list[int]],
    event_weights: dict[int, int],
    message_usage: dict[str, Tokens],
) -> Tokens:
    """Push each message's usage onto its own Events. Returns what was orphaned.

    ``output`` is split across the message's Events by content weight, because
    output really was produced piece by piece. The prompt-side figures
    (``input``, ``cache_read``, ``cache_creation``) are charged once, to the
    message's first Event: they price the context the message was *given*, which
    no individual content block is responsible for, and splitting them would
    invite reading "this Edit cost 40k input tokens" off a block.

    Orphaned usage — a message whose content was entirely thinking, so it
    produced no Event — is returned rather than discarded. Rare but real: 1 of
    106 messages in session 5f12bde3, 2,497 tokens.
    """
    orphaned = Tokens()
    for message_key, usage in message_usage.items():
        indices = message_events.get(message_key) or []
        if not indices:
            orphaned = orphaned + usage
            continue

        shares = _split_by_weight(
            usage.output, [event_weights.get(i, 1) for i in indices]
        )
        for index, output in zip(indices, shares):
            events[index].tokens = Tokens(output=output)

        first = events[indices[0]]
        first.tokens = Tokens(
            input=usage.input,
            output=first.tokens.output,
            cache_read=usage.cache_read,
            cache_creation=usage.cache_creation,
        )
    return orphaned


# Keys of ``toolUseResult`` whose contents actually go back to the model. The
# envelope also carries Claude Code's own bookkeeping, which never reaches the
# prompt and must not be sized as if it did — across every transcript here:
#
#     file             7,086,778   the model sees it (a Read's contents)
#     stdout             872,649   the model sees it
#     content            550,888   the model sees it
#     originalFile     1,236,808   local: the pre-edit file, for diffing
#     structuredPatch    960,061   local: the computed diff
#     newString/oldString 653,925  part of the Edit *call*, already counted there
#
# Counting the whole envelope inflated every Edit and made the estimated
# content of a session exceed what it was billed — in session 15d437fe, 25,623
# estimated against 20,644 fresh tokens, which starves the attribution ledger.
_VISIBLE_RESULT_KEYS = ("file", "stdout", "stderr", "content")


def _visible_result_size(envelope: Any, fallback_output: str) -> int:
    """Characters of a tool result that the model actually reads back.

    ``max`` rather than a sum, because the same payload is often present twice:
    a Bash result appears both as the ``tool_result`` block's text and as
    ``stdout`` in the envelope, while a Read's contents appear *only* in the
    envelope. Taking the larger picks up the file body without double-counting
    the command output.
    """
    if isinstance(envelope, str):
        return max(len(envelope), len(fallback_output))
    if not isinstance(envelope, dict):
        return len(fallback_output)

    visible = 0
    for key in _VISIBLE_RESULT_KEYS:
        value = envelope.get(key)
        if value is None:
            continue
        try:
            visible += len(value) if isinstance(value, str) else len(
                json.dumps(value, default=str))
        except (TypeError, ValueError):
            continue
    return max(visible, len(fallback_output))


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

    result.size_chars = _visible_result_size(envelope, fallback_output)

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


# Claude Code injects a great deal into the *user* role that the human never
# typed: a skill's body, IDE context, slash-command echoes, local command
# output. All of it is a ``type: "user"`` line with real text, so counting
# every such line as "chatting with user" put skill activations on the bar as
# if the human had said them.
#
# Two signals, both verified against every transcript here:
#
#   ``isMeta: true``  — image metadata, the local-command caveat, and the
#                       skill body ("Base directory for this skill: ...").
#   a machinery tag   — the text opens with one of the wrappers below.
#
# ``promptSource`` looks tempting and is *not* usable: real typed prompts
# ("hi", "create new branch 10-local-app-tool") are recorded as ``sdk``, so
# filtering on it would delete genuine human turns. ``origin.kind`` is absent
# on 58 of 344 real prompts, so it cannot be required either.
_MACHINE_PROMPT_TAGS = (
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<ide_opened_file>",
    "<ide_selection>",
    "<task-notification>",
    "<system-reminder>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
)


def is_human_prompt(line: dict, text: str, is_subagent: bool = False) -> bool:
    """Did a person actually type this, or did Claude Code inject it?

    ``is_subagent`` settles it on its own, and categorically rather than by
    signal: a subagent has no human in its loop at any point. Its transcript
    opens with a user-role line carrying the task, but the *parent agent* wrote
    that. Checked against a real child transcript
    (``agent-a0b679820a32f88c4.jsonl``): the line has neither ``promptSource``
    nor ``origin``, both of which a typed prompt carries — but neither absence
    is reliable enough to test for (see the note above), so the containment is
    what decides.
    """
    if is_subagent:
        return False
    if line.get("isMeta"):
        return False
    if line.get("promptSource") == "system":
        return False
    stripped = text.lstrip()
    return not stripped.startswith(_MACHINE_PROMPT_TAGS)


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
        # Usage from messages that produced no Event (thinking-only). Kept so
        # the session total stays exact — see ``_attribute_tokens``.
        "orphaned_tokens": Tokens(),
        # Message ids that emitted a thinking block. Thinking is billed as
        # output but its text is not stored (verified: the ``thinking`` field
        # is empty in every transcript here), so its size is unknowable — and
        # any message containing one is useless for calibrating characters per
        # token, because part of what was billed is invisible.
        "thinking_messages": set(),
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

    # Bookkeeping for token attribution, resolved in one pass at the end (it
    # cannot be done inline: a streamed message's final usage only arrives with
    # its last fragment, by which time its earlier Events already exist).
    message_events: dict[str, list[int]] = {}
    event_weights: dict[int, int] = {}
    message_usage: dict[str, Tokens] = {}

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
            human = is_human_prompt(line, text, is_subagent=agent_id is not None)
            if human:
                meta["user_prompts"].append(text)
            events.append(Event(
                uuid=uuid, ts=ts, type=EV_USER, vendor=VENDOR,
                parent_uuid=parent_uuid, text=text,
                agent_id=agent_id, depth=depth,
                is_human_prompt=human,
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

        # Un-ided messages (rare, but they exist in older transcripts) get their
        # own bucket per line so they are never conflated with each other.
        message_key = message_id or f"uuid:{uuid}"
        if tokens.total:
            message_usage[message_key] = tokens  # cumulative; last fragment wins
        else:
            message_usage.setdefault(message_key, Tokens())

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
                meta["thinking_messages"].add(message_key)
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
                    event_weights[idx] += _content_weight(text)
                    continue
                events.append(Event(
                    uuid=uuid, ts=ts, type=EV_ASSISTANT, vendor=VENDOR,
                    parent_uuid=parent_uuid, message_id=message_id,
                    model=model, tokens=Tokens(), text=text,
                    agent_id=agent_id, depth=depth,
                ))
                index = len(events) - 1
                message_events.setdefault(message_key, []).append(index)
                event_weights[index] = _content_weight(text)
                if message_id:
                    prose_index[message_id] = index
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
                tool_name = str(block.get("name") or "")
                events.append(Event(
                    uuid=uuid, ts=ts, type=EV_TOOL_USE, vendor=VENDOR,
                    parent_uuid=parent_uuid, message_id=message_id,
                    model=model,
                    # Filled in by ``_attribute_tokens`` once the whole message
                    # is known; a tool call is a real share of the output.
                    tokens=Tokens(),
                    tool=ToolCall(
                        id=tool_id,
                        name=tool_name,
                        input=tool_input,
                        # (2) inline the result onto the call.
                        result=results_by_id.get(tool_id),
                    ),
                    agent_id=agent_id, depth=depth,
                ))
                index = len(events) - 1
                message_events.setdefault(message_key, []).append(index)
                # The call as the model emitted it: name plus arguments. The
                # result is excluded — it is the tool's output, not the model's.
                event_weights[index] = _content_weight(
                    tool_name + json.dumps(tool_input, default=str)
                )

    meta["orphaned_tokens"] = _attribute_tokens(
        events, message_events, event_weights, message_usage
    )

    events.sort(key=lambda e: e.ts)
    return events, meta


# ----------------------------------------------------------------------
# Session assembly
# ----------------------------------------------------------------------

def load_session(project_slug: str, path: Path, with_subagents: bool = True) -> Optional[Session]:
    """Load one session, including its subagent transcripts."""
    events, meta = parse_transcript(path)

    orphaned_tokens = meta["orphaned_tokens"]
    thinking_messages = set(meta["thinking_messages"])

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
            orphaned_tokens = orphaned_tokens + sub_meta["orphaned_tokens"]
            thinking_messages |= sub_meta["thinking_messages"]
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
        orphaned_tokens=orphaned_tokens,
        thinking_message_ids=thinking_messages,
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
