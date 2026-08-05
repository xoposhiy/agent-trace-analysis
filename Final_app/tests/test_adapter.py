"""Adapter tests: raw Claude Code JSONL -> IR.

Each format claim made in ``adapters/claude_code.py`` gets a test here, so
format drift surfaces as a failure rather than as silently empty output. The
``Task`` -> ``Agent`` rename is the motivating example: it would have been
caught instantly by ``test_task_named_tool_is_still_recognised_as_subagent``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Final_app.adapters import claude_code
from Final_app.ir.models import EV_TOOL_USE, EV_USER

from .conftest import (
    PROJECT_SLUG,
    assistant_text_line,
    assistant_tool_line,
    tool_result_line,
    user_line,
    write_transcript,
)


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def test_discovery_finds_top_level_transcripts(claude_home: Path, simple_session: Path):
    found = claude_code.discover_sessions()

    assert len(found) == 1
    project_slug, path = found[0]
    assert project_slug == PROJECT_SLUG
    assert path == simple_session


def test_discovery_ignores_subagent_files(claude_home: Path, session_with_subagent: Path):
    """Subagents are children, not sessions — they must not appear as rows."""
    found = claude_code.discover_sessions()

    assert len(found) == 1
    assert found[0][1] == session_with_subagent


def test_discovery_returns_empty_when_projects_dir_is_missing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "does-not-exist"))

    assert claude_code.discover_sessions() == []


# ----------------------------------------------------------------------
# Session assembly
# ----------------------------------------------------------------------

def test_session_carries_the_claude_code_title(claude_home: Path, simple_session: Path):
    """The ``ai-title`` line is Claude Code's own title — free, no LLM."""
    session = claude_code.load_session(PROJECT_SLUG, simple_session)

    assert session.title == "Fix the login bug"


def test_session_metadata_comes_from_the_transcript(claude_home: Path, simple_session: Path):
    session = claude_code.load_session(PROJECT_SLUG, simple_session)

    assert session.session_id == simple_session.stem
    assert session.project == PROJECT_SLUG
    assert session.git_branch == "main"
    assert session.model == "claude-opus-4-6"
    assert session.vendor_version == "2.1.78"


def test_user_prompts_exclude_tool_result_carriers(claude_home: Path, simple_session: Path):
    """Tool results arrive on user-role lines but are not things the user said."""
    session = claude_code.load_session(PROJECT_SLUG, simple_session)

    assert session.user_prompts == ["The login page is broken", "Please fix it"]
    assert len([e for e in session.events if e.type == EV_USER]) == 2


def test_tool_calls_are_one_event_each(claude_home: Path, simple_session: Path):
    session = claude_code.load_session(PROJECT_SLUG, simple_session)

    assert [e.tool.name for e in session.tool_calls] == ["Read", "Bash", "Edit"]


def test_bash_command_text_survives_for_the_judge(claude_home: Path, simple_session: Path):
    """The classifier needs the command itself to tell read from execute."""
    session = claude_code.load_session(PROJECT_SLUG, simple_session)
    bash = next(e for e in session.tool_calls if e.tool.name == "Bash")

    assert bash.tool.input["command"] == "pytest -k login"
    assert bash.tool.input["description"] == "Run the login tests"


def test_events_are_ordered_by_timestamp(claude_home: Path, simple_session: Path):
    session = claude_code.load_session(PROJECT_SLUG, simple_session)
    timestamps = [e.ts for e in session.events]

    assert timestamps == sorted(timestamps)


# ----------------------------------------------------------------------
# The four behaviours borrowed from Entire's compact package
# ----------------------------------------------------------------------

def test_thinking_blocks_never_reach_the_ir(claude_home: Path, simple_session: Path):
    """Thinking is dropped outright — not merely absent from ``text``.

    Asserting only "the words aren't in ``e.text``" passes even if the drop is
    removed, because an unhandled block type falls through every branch and
    vanishes by accident. Pinning the exact event count is what makes this
    fail if a future fall-through starts emitting thinking as prose.

    The fixture's 6 events: 2 user prompts, 1 assistant prose, 3 tool calls.
    The thinking block and the 3 tool-result carriers contribute none.
    """
    session = claude_code.load_session(PROJECT_SLUG, simple_session)

    assert len(session.events) == 6
    assert all("secret reasoning" not in e.text for e in session.events)


def test_streamed_fragments_sharing_a_message_id_merge(claude_home: Path):
    """Without merging, one turn becomes several phantom blocks."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
        assistant_text_line("a1", "msg1", "Hello ", "2026-08-01T10:00:01.000Z",
                            output_tokens=5),
        assistant_text_line("a2", "msg1", "there", "2026-08-01T10:00:02.000Z",
                            output_tokens=9),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    prose = [e for e in session.events if e.text and e.type != EV_USER]

    assert len(prose) == 1
    assert prose[0].text == "Hello there"
    # Streaming usage is cumulative, so the final fragment wins outright.
    assert prose[0].tokens.output == 9


def test_tool_results_are_inlined_onto_their_call(claude_home: Path, simple_session: Path):
    session = claude_code.load_session(PROJECT_SLUG, simple_session)
    read = next(e for e in session.tool_calls if e.tool.name == "Read")

    assert read.tool.result is not None
    assert read.tool.result.output == "def login(): ..."


def test_read_result_envelope_yields_file_metadata(claude_home: Path, simple_session: Path):
    """Read's envelope nests the path under ``file``."""
    session = claude_code.load_session(PROJECT_SLUG, simple_session)
    read = next(e for e in session.tool_calls if e.tool.name == "Read")

    assert read.tool.result.file_path == "/repo/login.py"
    assert read.tool.result.num_lines == 42


def test_edit_result_envelope_yields_file_metadata(claude_home: Path, simple_session: Path):
    """Edit's envelope puts the path at the top level instead."""
    session = claude_code.load_session(PROJECT_SLUG, simple_session)
    edit = next(e for e in session.tool_calls if e.tool.name == "Edit")

    assert edit.tool.result.file_path == "/repo/login.py"


def test_bash_result_envelope_combines_stdout_and_stderr(claude_home: Path):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "run it", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "m1", "t1", "Bash", {"command": "make"},
                            "2026-08-01T10:00:01.000Z"),
        tool_result_line("r1", "t1", "", "2026-08-01T10:00:02.000Z",
                         envelope={"stdout": "building", "stderr": "warning: x",
                                   "interrupted": False}),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.tool_calls[0].tool.result.output == "building\nwarning: x"


# ----------------------------------------------------------------------
# Subagents — absent from Entire's IR, central to ours
# ----------------------------------------------------------------------

def test_subagent_events_are_merged_into_the_parent_session(
    claude_home: Path, session_with_subagent: Path
):
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)

    assert session.subagent_ids == ["a0b679820a32f88c4"]
    subagent_events = [e for e in session.events if e.is_subagent]
    assert len(subagent_events) == 3  # 1 prompt + 2 tool calls


def test_subagent_tool_calls_are_tagged_with_their_agent_id(
    claude_home: Path, session_with_subagent: Path
):
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)
    inner = [e for e in session.tool_calls if e.is_subagent]

    assert [e.tool.name for e in inner] == ["Grep", "Read"]
    assert all(e.agent_id == "a0b679820a32f88c4" for e in inner)
    assert all(e.depth == 1 for e in inner)


def test_main_thread_events_have_no_agent_id(
    claude_home: Path, session_with_subagent: Path
):
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)
    outer = [e for e in session.events if not e.is_subagent]

    assert outer, "expected main-thread events"
    assert all(e.agent_id is None and e.depth == 0 for e in outer)


def test_subagent_and_parent_events_interleave_by_time(
    claude_home: Path, session_with_subagent: Path
):
    """The bar renders one timeline, so children must sort in with the parent."""
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)
    timestamps = [e.ts for e in session.events]

    assert timestamps == sorted(timestamps)
    assert any(e.is_subagent for e in session.events)


def test_subagents_can_be_skipped(claude_home: Path, session_with_subagent: Path):
    session = claude_code.load_session(
        PROJECT_SLUG, session_with_subagent, with_subagents=False
    )

    assert session.subagent_ids == []
    assert not any(e.is_subagent for e in session.events)


def test_task_named_tool_is_still_recognised_as_subagent():
    """Claude Code renamed ``Task`` -> ``Agent``; old transcripts still say Task.

    Keying on either name alone silently finds nothing. This pins both.
    """
    assert "Agent" in claude_code.SUBAGENT_TOOLS
    assert "Task" in claude_code.SUBAGENT_TOOLS


# ----------------------------------------------------------------------
# Degenerate input
# ----------------------------------------------------------------------

def test_malformed_line_does_not_lose_the_session(claude_home: Path):
    """Transcripts are appended to live; the last line can be a partial write."""
    path = claude_home / PROJECT_SLUG / "s.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"type":"user","uuid":"u1","timestamp":"2026-08-01T10:00:00.000Z",'
        '"message":{"role":"user","content":"hello"}}\n'
        '{"type":"assistant","uuid":"a1","timestam\n',  # truncated mid-write
        encoding="utf-8",
    )
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session is not None
    assert session.user_prompts == ["hello"]


def test_session_with_no_events_is_dropped(claude_home: Path):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        {"type": "attachment", "uuid": "x", "timestamp": "2026-08-01T10:00:00.000Z"},
    ])

    assert claude_code.load_session(PROJECT_SLUG, path) is None


def test_lines_without_a_timestamp_are_skipped(claude_home: Path):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "kept", "2026-08-01T10:00:00.000Z"),
        {"type": "user", "uuid": "u2",
         "message": {"role": "user", "content": "no timestamp"}},
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.user_prompts == ["kept"]


def test_load_all_sessions_orders_newest_first(claude_home: Path):
    for name, hour in (("old", "09"), ("new", "11"), ("mid", "10")):
        write_transcript(claude_home / PROJECT_SLUG / f"{name}.jsonl", [
            user_line("u1", name, f"2026-08-01T{hour}:00:00.000Z"),
        ])
    sessions = claude_code.load_all_sessions()

    assert [s.session_id for s in sessions] == ["new", "mid", "old"]


# ----------------------------------------------------------------------
# Project slug decoding
# ----------------------------------------------------------------------

def test_unslug_keeps_enough_to_tell_projects_apart(monkeypatch):
    """Taking only the last segment collapsed three real projects into one."""
    monkeypatch.setattr("os.path.expanduser", lambda p: "/Users/tester")
    base = "-Users-tester-Desktop-Agent-traces-analysis"

    assert claude_code.unslug_project(base) != claude_code.unslug_project(base + "-Local-app")


def test_unslug_strips_home_and_scaffolding_directories(monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: "/Users/tester")

    assert claude_code.unslug_project("-Users-tester-Desktop-my-repo") == "my-repo"
    assert claude_code.unslug_project("-Users-tester-src-my-repo") == "my-repo"


def test_unslug_renders_the_home_directory_itself(monkeypatch):
    monkeypatch.setattr("os.path.expanduser", lambda p: "/Users/tester")

    assert claude_code.unslug_project("-Users-tester") == "~"


# ----------------------------------------------------------------------
# Human prompts vs what the harness injects into the user role
# ----------------------------------------------------------------------
# Claude Code writes a skill's body, IDE context, slash-command echoes and
# local command output as ``type: "user"`` lines with real text. Counting those
# as human turns put skill activations on the bar as if the user had typed
# them. ``promptSource`` cannot be used to tell them apart — real typed prompts
# ("hi", "create new branch ...") are recorded as ``sdk`` in real transcripts.

def test_an_injected_skill_body_is_not_a_human_prompt(claude_home: Path):
    """The reported bug. Claude Code marks the injection with ``isMeta``."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "make me a chart", "2026-08-01T10:00:00.000Z"),
        user_line("u2", "Base directory for this skill: /tmp/skills/dataviz\n"
                        "# Data Visualization\nUse this skill when...",
                  "2026-08-01T10:00:01.000Z", is_meta=True),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    typed, injected = [e for e in session.events if e.type == EV_USER]

    assert typed.is_human_prompt is True
    assert injected.is_human_prompt is False


def test_a_subagents_task_prompt_is_not_a_human_turn(
    claude_home: Path, session_with_subagent: Path
):
    """A subagent has no human in its loop — ever.

    Its transcript opens with a user-role line holding the task, but the
    parent agent wrote that, not a person: verified against a real subagent
    file, the line carries neither ``promptSource`` nor ``origin``, both of
    which a typed prompt has. Counted as human it put a purple "chatting with
    user" marker at the top of every subagent's bar, claiming an interruption
    that never happened.
    """
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)
    child_prompts = [event for event in session.events
                     if event.type == EV_USER and event.is_subagent]

    assert child_prompts, "fixture must contain the child's task prompt"
    assert all(not event.is_human_prompt for event in child_prompts)


def test_the_parents_own_prompt_is_still_human(
    claude_home: Path, session_with_subagent: Path
):
    """The subagent rule must not reach the main thread."""
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)
    main_prompts = [event for event in session.events
                    if event.type == EV_USER and not event.is_subagent]

    assert [event.is_human_prompt for event in main_prompts] == [True]


def test_a_subagents_task_prompt_stays_out_of_user_prompts(
    claude_home: Path, session_with_subagent: Path
):
    """Otherwise the session's prompt list gains turns nobody typed."""
    session = claude_code.load_session(PROJECT_SLUG, session_with_subagent)

    assert session.user_prompts == ["Audit this repo"]


@pytest.mark.parametrize("text", [
    "<command-name>/model</command-name>",
    "<local-command-stdout>Set model to Haiku</local-command-stdout>",
    "<local-command-caveat>Caveat: the messages below...</local-command-caveat>",
    "<ide_opened_file>The user opened /repo/a.py</ide_opened_file>",
    "<ide_selection>The user selected lines 29 to 29</ide_selection>",
    "<task-notification><task-id>abc</task-id></task-notification>",
    "<system-reminder>remember the thing</system-reminder>",
])
def test_harness_machinery_is_not_a_human_prompt(claude_home: Path, text: str):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", text, "2026-08-01T10:00:00.000Z"),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.events[0].is_human_prompt is False


def test_a_prompt_recorded_as_sdk_is_still_human(claude_home: Path):
    """``promptSource`` is not the signal — real typed prompts show as ``sdk``."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "create new branch 10-local-app-tool",
                  "2026-08-01T10:00:00.000Z", prompt_source="sdk"),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.events[0].is_human_prompt is True


def test_only_human_prompts_reach_user_prompts(claude_home: Path):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "the real question", "2026-08-01T10:00:00.000Z"),
        user_line("u2", "<command-name>/model</command-name>",
                  "2026-08-01T10:00:01.000Z"),
        user_line("u3", "Base directory for this skill: /tmp/s\nbody",
                  "2026-08-01T10:00:02.000Z", is_meta=True),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.user_prompts == ["the real question"]
