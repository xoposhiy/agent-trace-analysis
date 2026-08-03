"""Shared fixtures: a synthetic ``~/.claude/projects`` tree on disk.

Every test builds its own transcripts under ``tmp_path`` and points
``CLAUDE_CONFIG_DIR`` at it. Nothing here reads the developer's real
``~/.claude`` — a test whose result depends on whose machine it runs on is
worthless (CLAUDE.md §6).

The line shapes below are copied from real Claude Code transcripts, so the
fixtures drift only when the real format drifts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# Line builders — the shapes Claude Code actually writes
# ----------------------------------------------------------------------

PROJECT_SLUG = "-Users-tester-Desktop-my-repo"
SESSION_ID = "11111111-2222-3333-4444-555555555555"


def ai_title_line(title: str, session_id: str = SESSION_ID) -> dict:
    """The line Claude Code writes with its own generated session title."""
    return {"type": "ai-title", "aiTitle": title, "sessionId": session_id}


def user_line(uuid: str, text: str, timestamp: str, parent_uuid: str | None = None) -> dict:
    """A real user prompt."""
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": timestamp,
        "gitBranch": "main",
        "version": "2.1.78",
        "sessionId": SESSION_ID,
        "message": {"role": "user", "content": text},
    }


def assistant_text_line(
    uuid: str,
    message_id: str,
    text: str,
    timestamp: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> dict:
    """An assistant message carrying prose."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {
            "id": message_id,
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
        },
    }


def assistant_thinking_line(uuid: str, message_id: str, timestamp: str) -> dict:
    """A thinking block, which must never reach the IR."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {
            "id": message_id,
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [{"type": "thinking", "thinking": "secret reasoning",
                         "signature": "abc"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def assistant_tool_line(
    uuid: str,
    message_id: str,
    tool_id: str,
    tool_name: str,
    tool_input: dict,
    timestamp: str,
) -> dict:
    """An assistant message invoking a tool."""
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {
            "id": message_id,
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [{
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def tool_result_line(
    uuid: str,
    tool_id: str,
    output: str,
    timestamp: str,
    envelope: dict | str | None = None,
    is_error: bool = False,
) -> dict:
    """The user-role line carrying a tool's result back to the model."""
    line = {
        "type": "user",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": output,
                "is_error": is_error,
            }],
        },
    }
    if envelope is not None:
        line["toolUseResult"] = envelope
    return line


# ----------------------------------------------------------------------
# Filesystem fixtures
# ----------------------------------------------------------------------

def write_transcript(path: Path, lines: list[dict]) -> Path:
    """Write JSONL, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty fake ``~/.claude``, wired up via ``CLAUDE_CONFIG_DIR``.

    Returns the ``projects/`` directory, so tests write straight into it.
    """
    home = tmp_path / "claude-home"
    projects = home / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    return projects


@pytest.fixture
def simple_session(claude_home: Path) -> Path:
    """One ordinary session: title, two prompts, prose, and three tool calls.

    Composition is fixed and asserted on by name elsewhere:
      - 2 user prompts
      - 3 tool calls (Read, Bash, Edit)
      - 1 thinking block, which must be dropped
    """
    path = claude_home / PROJECT_SLUG / f"{SESSION_ID}.jsonl"
    write_transcript(path, [
        ai_title_line("Fix the login bug"),
        user_line("u1", "The login page is broken", "2026-08-01T10:00:00.000Z"),
        assistant_thinking_line("a0", "msg1", "2026-08-01T10:00:01.000Z"),
        assistant_text_line("a1", "msg1", "Let me look.", "2026-08-01T10:00:02.000Z",
                            input_tokens=100, output_tokens=50,
                            cache_read=9000, cache_creation=200),
        assistant_tool_line("a2", "msg1", "t1", "Read",
                            {"file_path": "/repo/login.py"},
                            "2026-08-01T10:00:03.000Z"),
        tool_result_line("r1", "t1", "def login(): ...", "2026-08-01T10:00:04.000Z",
                         envelope={"type": "text",
                                   "file": {"filePath": "/repo/login.py",
                                            "numLines": 42}}),
        assistant_tool_line("a3", "msg2", "t2", "Bash",
                            {"command": "pytest -k login",
                             "description": "Run the login tests"},
                            "2026-08-01T10:00:05.000Z"),
        tool_result_line("r2", "t2", "1 failed", "2026-08-01T10:00:06.000Z",
                         envelope={"stdout": "1 failed", "stderr": "",
                                   "interrupted": False}),
        user_line("u2", "Please fix it", "2026-08-01T10:01:00.000Z"),
        assistant_tool_line("a4", "msg3", "t3", "Edit",
                            {"file_path": "/repo/login.py"},
                            "2026-08-01T10:01:05.000Z"),
        tool_result_line("r3", "t3", "ok", "2026-08-01T10:01:06.000Z",
                         envelope={"filePath": "/repo/login.py",
                                   "oldString": "a", "newString": "b"}),
    ])
    return path


@pytest.fixture
def session_with_subagent(claude_home: Path) -> Path:
    """A session that spawns one subagent, whose transcript is a separate file.

    Composition:
      - main thread: 1 prompt, 1 ``Agent`` tool call
      - subagent:    2 tool calls (Grep, Read)
    """
    session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    agent_id = "a0b679820a32f88c4"
    project_dir = claude_home / PROJECT_SLUG

    write_transcript(project_dir / f"{session_id}.jsonl", [
        ai_title_line("Audit the codebase", session_id),
        user_line("u1", "Audit this repo", "2026-08-02T09:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Agent",
                            {"subagent_type": "Explore",
                             "description": "Find all TODOs",
                             "prompt": "Search for TODO comments"},
                            "2026-08-02T09:00:01.000Z"),
        # The spawn result carries the child's id — the parent -> child link.
        tool_result_line("r1", "t1", "done", "2026-08-02T09:00:30.000Z",
                         envelope={"agentId": agent_id,
                                   "description": "Find all TODOs",
                                   "status": "completed",
                                   "prompt": "Search for TODO comments"}),
    ])

    # Subagent lines all carry isSidechain + agentId, and start at parentUuid null.
    sub_path = project_dir / session_id / "subagents" / f"agent-{agent_id}.jsonl"
    write_transcript(sub_path, [
        {"type": "user", "uuid": "s-u1", "parentUuid": None, "isSidechain": True,
         "agentId": agent_id, "timestamp": "2026-08-02T09:00:02.000Z",
         "message": {"role": "user", "content": "Search for TODO comments"}},
        {"type": "assistant", "uuid": "s-a1", "isSidechain": True,
         "agentId": agent_id, "timestamp": "2026-08-02T09:00:10.000Z",
         "message": {"id": "s-msg1", "role": "assistant", "model": "claude-opus-4-6",
                     "content": [{"type": "tool_use", "id": "s-t1", "name": "Grep",
                                  "input": {"pattern": "TODO"}}],
                     "usage": {"input_tokens": 5, "output_tokens": 5}}},
        {"type": "assistant", "uuid": "s-a2", "isSidechain": True,
         "agentId": agent_id, "timestamp": "2026-08-02T09:00:20.000Z",
         "message": {"id": "s-msg2", "role": "assistant", "model": "claude-opus-4-6",
                     "content": [{"type": "tool_use", "id": "s-t2", "name": "Read",
                                  "input": {"file_path": "/repo/a.py"}}],
                     "usage": {"input_tokens": 5, "output_tokens": 5}}},
    ])
    return project_dir / f"{session_id}.jsonl"
