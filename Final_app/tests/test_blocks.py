"""Block-building tests: classification, the merge rule, subagent nesting.

The merge rule under test is the sketch's, verbatim:

    neighbouring blocks of the same type (with coordination inside)
    - join them in one block

No test here calls the LLM — ``use_judge=False`` throughout, so classification
falls back to rules plus the shell heuristic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from Final_app.analysis import classify
from Final_app.analysis.blocks import (
    Y_METRICS,
    block_metric,
    build_blocks,
    merge_neighbours,
)
from Final_app.ir.models import (
    COORDINATION,
    EV_ASSISTANT,
    EV_TOOL_USE,
    EV_USER,
    EXECUTE,
    READ,
    SUBAGENT,
    USER_CHAT,
    WRITE,
    Block,
    Event,
    Session,
    Tokens,
    ToolCall,
    ToolResult,
)

BASE_TIME = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------

def tool_event(
    offset_seconds: int,
    name: str,
    tool_input: dict | None = None,
    agent_id: str | None = None,
    spawned_agent_id: str = "",
    tokens: Tokens | None = None,
) -> Event:
    result = ToolResult(spawned_agent_id=spawned_agent_id) if spawned_agent_id else None
    return Event(
        uuid=f"t{offset_seconds}",
        ts=BASE_TIME + timedelta(seconds=offset_seconds),
        type=EV_TOOL_USE,
        tool=ToolCall(id=f"id{offset_seconds}", name=name,
                      input=tool_input or {}, result=result),
        agent_id=agent_id,
        depth=1 if agent_id else 0,
        tokens=tokens or Tokens(),
    )


def user_event(offset_seconds: int, text: str = "hi") -> Event:
    return Event(uuid=f"u{offset_seconds}",
                 ts=BASE_TIME + timedelta(seconds=offset_seconds),
                 type=EV_USER, text=text)


def kinds_of(blocks: list[Block]) -> list[str]:
    return [b.kind for b in blocks]


# ----------------------------------------------------------------------
# Stage A — rules
# ----------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,expected", [
    ("Read", READ),
    ("Grep", READ),
    ("Glob", READ),
    ("WebFetch", READ),
    ("Edit", WRITE),
    ("Write", WRITE),
    ("NotebookEdit", WRITE),
    ("TodoWrite", COORDINATION),
    ("AskUserQuestion", COORDINATION),
    ("Agent", SUBAGENT),
    ("Task", SUBAGENT),
])
def test_rules_classify_unambiguous_tools(tool_name: str, expected: str):
    assert classify.rule_kind(tool_event(0, tool_name)) == expected


def test_todowrite_is_coordination_not_a_write():
    """A substring rule on "write" would get this wrong, silently."""
    assert classify.rule_kind(tool_event(0, "TodoWrite")) == COORDINATION


# ----------------------------------------------------------------------
# Only a person is "chatting with user"
# ----------------------------------------------------------------------
# Claude Code injects into the *user* role a great deal the human never typed:
# a skill's body, IDE context, slash-command echoes, local command output. All
# of it is a ``type: "user"`` line with real text, so treating every one as a
# human turn put skill activations on the bar as if the user had said them.

def _user_event(offset: int, text: str, human: bool = True) -> Event:
    return Event(uuid=f"u{offset}",
                 ts=BASE_TIME + timedelta(seconds=offset),
                 type=EV_USER, text=text, is_human_prompt=human)


def test_an_injected_skill_body_is_not_the_user_talking():
    """The reported bug: activating a skill showed up as a human turn."""
    skill = _user_event(0, "Base directory for this skill: /tmp/skills/dataviz\n"
                           "# Data Visualization\nUse this skill when...",
                        human=False)

    assert classify.rule_kind(skill) == COORDINATION


@pytest.mark.parametrize("text", [
    "<command-name>/model</command-name>",
    "<local-command-stdout>Set model to Haiku</local-command-stdout>",
    "<ide_opened_file>The user opened /repo/a.py</ide_opened_file>",
    "<task-notification><task-id>abc</task-id></task-notification>",
    "<system-reminder>remember the thing</system-reminder>",
])
def test_harness_machinery_in_the_user_role_is_coordination(text: str):
    assert classify.rule_kind(_user_event(0, text, human=False)) == COORDINATION


def test_a_real_prompt_is_still_the_user_talking():
    assert classify.rule_kind(_user_event(0, "fix the login bug")) == USER_CHAT


def test_injected_lines_do_not_split_a_run_of_work():
    """As coordination they absorb, so a skill activation mid-read is invisible.

    As ``user_chat`` each one was a full-width marker cutting the bar in two.
    """
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        _user_event(1, "<command-name>/model</command-name>", human=False),
        tool_event(2, "Read", {"file_path": "/b.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ]
    assert blocks[0].message_count == 3


def test_a_real_prompt_still_breaks_a_run_of_work():
    """The human interrupting is a real boundary and must stay visible."""
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        _user_event(1, "wait, do the other file first"),
        tool_event(2, "Read", {"file_path": "/b.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ, USER_CHAT, READ]


def test_user_messages_are_chatting_with_user():
    assert classify.rule_kind(user_event(0)) == USER_CHAT


def test_bash_is_left_to_the_judge():
    assert classify.rule_kind(tool_event(0, "Bash", {"command": "ls"})) is None


def test_unknown_tools_fall_back_to_coordination():
    """A guess would distort the bar; new tools appear constantly."""
    assert classify.rule_kind(tool_event(0, "SomeBrandNewTool")) == COORDINATION


# ----------------------------------------------------------------------
# The shell heuristic (the no-judge fallback)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("command,expected", [
    ("cat foo.py", READ),
    ("grep -r TODO .", READ),
    ("ls -la", READ),
    ("/usr/bin/head file", READ),
    ("pytest -k login", EXECUTE),
    ("npm run build", EXECUTE),
    ("git commit -m x", EXECUTE),
    ("", EXECUTE),
])
def test_shell_heuristic(command: str, expected: str):
    assert classify.heuristic_shell_kind(command) == expected


def test_chained_commands_are_execute_even_when_they_start_with_a_read():
    """A read-only first word says nothing about what follows the pipe."""
    assert classify.heuristic_shell_kind("cat x | python run.py") == EXECUTE
    assert classify.heuristic_shell_kind("ls && make") == EXECUTE


# ----------------------------------------------------------------------
# The merge rule
# ----------------------------------------------------------------------

def _kinded(*pairs: tuple[int, str]) -> list[tuple[Event, str, None]]:
    return [(tool_event(offset, "X"), kind, None) for offset, kind in pairs]


def test_a_run_of_one_kind_becomes_a_single_block():
    blocks = merge_neighbours(_kinded((0, READ), (1, READ), (2, READ)))

    assert kinds_of(blocks) == [READ]
    assert blocks[0].message_count == 3


def test_different_kinds_stay_separate():
    blocks = merge_neighbours(_kinded((0, READ), (1, WRITE), (2, EXECUTE)))

    assert kinds_of(blocks) == [READ, WRITE, EXECUTE]


def test_coordination_between_two_reads_is_absorbed():
    """The sketch's rule: coordination *inside* a run does not split it."""
    blocks = merge_neighbours(
        _kinded((0, READ), (1, COORDINATION), (2, READ)))

    assert kinds_of(blocks) == [READ]
    assert blocks[0].message_count == 3


def test_coordination_between_different_kinds_survives():
    """There it is a real boundary, not glue."""
    blocks = merge_neighbours(
        _kinded((0, READ), (1, COORDINATION), (2, WRITE)))

    assert kinds_of(blocks) == [READ, COORDINATION, WRITE]


def test_leading_coordination_is_its_own_block():
    """Nothing precedes it, so there is no run to absorb it into."""
    blocks = merge_neighbours(_kinded((0, COORDINATION), (1, READ)))

    assert kinds_of(blocks) == [COORDINATION, READ]


def test_trailing_coordination_is_its_own_block():
    blocks = merge_neighbours(_kinded((0, READ), (1, COORDINATION)))

    assert kinds_of(blocks) == [READ, COORDINATION]


# ----------------------------------------------------------------------
# Preamble absorption
# ----------------------------------------------------------------------
# A coordination run that is small next to the work it introduces is the agent
# announcing the call, not a step of its own. Measured over 407 such pairs in
# real sessions: coordination is 5.4% the size of the run it precedes at p25
# and 61.7% at p75, so preambles and substantive analysis separate cleanly.
# See ``blocks.PREAMBLE_MAX_RATIO``.

def _big_write(offset: int, size: int = 4000) -> Event:
    """A Write whose payload dwarfs any sentence introducing it."""
    return tool_event(offset, "Write",
                      {"file_path": "/repo/a.py", "content": "x" * size})


def test_a_short_preamble_joins_the_work_it_introduces():
    session = _session([
        prose_event(0, "Now the bar rewrite:"),
        _big_write(1),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 2


def test_the_reported_pattern_collapses_to_one_block():
    """coordination + X + coordination + X, all four, becomes one X."""
    session = _session([
        prose_event(0, "Now the bar rewrite:"),
        _big_write(1),
        prose_event(2, "And the session page:"),
        _big_write(3),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 4


def test_substantial_coordination_is_not_swallowed_by_the_next_call():
    """Real analysis stays its own block, however small the call after it."""
    session = _session([
        prose_event(0, "The measurement says otherwise. " * 40),
        # A tiny call: ``pytest`` rather than ``ls``, since a read-only command
        # would classify as READ and stop this testing what it says it does.
        tool_event(1, "Bash", {"command": "pytest"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [COORDINATION, EXECUTE]


def test_a_preamble_joins_the_next_kind_not_the_previous_one():
    """The sentence introduces the write; it is not the tail of the read."""
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        prose_event(1, "Now the fix:"),
        _big_write(2),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ, WRITE]
    assert blocks[1].message_count == 2, "the prose should sit with the write"


def test_a_boundary_that_carries_real_content_still_survives():
    """Absorption must not erase every transition — only the announcements."""
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        prose_event(1, "That rules out the parser. " * 30),
        tool_event(2, "Bash", {"command": "pytest"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ, COORDINATION, EXECUTE]


def test_trailing_coordination_has_nothing_to_join():
    session = _session([
        _big_write(0),
        prose_event(1, "Done."),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [WRITE, COORDINATION]


def test_a_preamble_is_never_folded_into_the_users_turn():
    """A human message is not something the agent was preparing for."""
    session = _session([
        prose_event(0, "Ok."),
        Event(uuid="u1", ts=BASE_TIME + timedelta(seconds=1), type=EV_USER,
              text="and now do the other thing, here is a long follow-up " * 5),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [COORDINATION, USER_CHAT]


def test_absorbing_a_preamble_loses_no_event_and_no_tokens():
    session = _session([
        prose_event(0, "Now the bar rewrite:"),   # 25 output tokens
        _big_write(1),
        prose_event(2, "And the session page:"),  # 25 output tokens
        _big_write(3),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert sum(b.message_count for b in blocks) == 4
    assert blocks[0].tokens.output == 50


def test_several_coordination_events_absorb_as_one_group():
    blocks = merge_neighbours(_kinded(
        (0, READ), (1, COORDINATION), (2, COORDINATION), (3, READ)))

    assert kinds_of(blocks) == [READ]
    assert blocks[0].message_count == 4


def test_absorption_chains_across_repeated_interruptions():
    """read, coord, read, coord, read is one stretch of reading."""
    blocks = merge_neighbours(_kinded(
        (0, READ), (1, COORDINATION), (2, READ), (3, COORDINATION), (4, READ)))

    assert kinds_of(blocks) == [READ]
    assert blocks[0].message_count == 5


def test_absorption_does_not_bridge_a_different_kind():
    blocks = merge_neighbours(_kinded(
        (0, READ), (1, COORDINATION), (2, WRITE), (3, COORDINATION), (4, READ)))

    assert kinds_of(blocks) == [READ, COORDINATION, WRITE, COORDINATION, READ]


# ----------------------------------------------------------------------
# Same-file "edit and verify" loops
# ----------------------------------------------------------------------

def test_a_read_of_a_just_written_file_becomes_part_of_the_write():
    blocks = merge_neighbours([
        (tool_event(0, "Write", {"file_path": "/a.py"}), WRITE, None),
        (tool_event(1, "Read", {"file_path": "/a.py"}), READ, None),
    ])

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 2


def test_a_write_read_write_read_loop_on_one_file_is_a_single_write_block():
    """The reported pattern: edit, check it landed, edit again, check again."""
    blocks = merge_neighbours([
        (tool_event(0, "Edit", {"file_path": "/a.py"}), WRITE, None),
        (tool_event(1, "Read", {"file_path": "/a.py"}), READ, None),
        (tool_event(2, "Edit", {"file_path": "/a.py"}), WRITE, None),
        (tool_event(3, "Read", {"file_path": "/a.py"}), READ, None),
    ])

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 4


def test_coordination_inside_a_verification_loop_is_absorbed_too():
    blocks = merge_neighbours([
        (tool_event(0, "Write", {"file_path": "/a.py"}), WRITE, None),
        (tool_event(1, "TodoWrite", {}), COORDINATION, None),
        (tool_event(2, "Read", {"file_path": "/a.py"}), READ, None),
    ])

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 3


def test_a_read_before_any_write_is_exploration_not_verification():
    """Only a write starts the loop — reading a fresh file first is unrelated,
    and must stay a read rather than being folded into the edit after it."""
    blocks = merge_neighbours([
        (tool_event(0, "Read", {"file_path": "/a.py"}), READ, None),
        (tool_event(1, "Edit", {"file_path": "/a.py"}), WRITE, None),
    ])

    assert kinds_of(blocks) == [READ, WRITE]


def test_a_read_of_a_different_file_does_not_join_the_loop():
    blocks = merge_neighbours([
        (tool_event(0, "Write", {"file_path": "/a.py"}), WRITE, None),
        (tool_event(1, "Read", {"file_path": "/b.py"}), READ, None),
    ])

    assert kinds_of(blocks) == [WRITE, READ]


def test_the_loop_ends_at_the_next_different_kind():
    blocks = merge_neighbours([
        (tool_event(0, "Write", {"file_path": "/a.py"}), WRITE, None),
        (tool_event(1, "Read", {"file_path": "/a.py"}), READ, None),
        (tool_event(2, "Bash", {"command": "pytest"}), EXECUTE, None),
    ])

    assert kinds_of(blocks) == [WRITE, EXECUTE]
    assert blocks[0].message_count == 2


def test_an_edit_verify_loop_survives_the_whole_pipeline():
    """End to end via ``build_blocks``, not just ``merge_neighbours`` directly."""
    session = _session([
        tool_event(0, "Edit", {"file_path": "/a.py"}),
        tool_event(1, "Read", {"file_path": "/a.py"}),
        tool_event(2, "Edit", {"file_path": "/a.py"}),
        tool_event(3, "Read", {"file_path": "/a.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 4
    assert "a.py" in blocks[0].label


# ----------------------------------------------------------------------
# Test-development loops: write, run the matching test, write, run again
# ----------------------------------------------------------------------

def test_a_write_execute_write_execute_test_loop_is_a_single_write_block():
    """The same shape as the read loop, with a shell command standing in for
    the verifying read — matched by the command naming the file's stem."""
    blocks = merge_neighbours([
        (tool_event(0, "Edit", {"file_path": "/blocks.py"}), WRITE, None),
        (tool_event(1, "Bash", {"command": "pytest tests/test_blocks.py"}), EXECUTE, None),
        (tool_event(2, "Edit", {"file_path": "/blocks.py"}), WRITE, None),
        (tool_event(3, "Bash", {"command": "pytest tests/test_blocks.py"}), EXECUTE, None),
    ])

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 4


def test_an_unrelated_shell_command_does_not_join_the_test_loop():
    """A command that never names the file is a real transition, not a check
    on it — ``ls`` after an edit ends the loop rather than joining it."""
    blocks = merge_neighbours([
        (tool_event(0, "Edit", {"file_path": "/blocks.py"}), WRITE, None),
        (tool_event(1, "Bash", {"command": "ls -la"}), EXECUTE, None),
    ])

    assert kinds_of(blocks) == [WRITE, EXECUTE]


def test_naming_the_file_is_not_enough_without_a_test_marker():
    """Real transcripts turned up a command that named the edited file's
    module path purely by coincidence while doing something else entirely
    (restarting a server) — naming the file alone must not be sufficient."""
    blocks = merge_neighbours([
        (tool_event(0, "Edit", {"file_path": "/app.js"}), WRITE, None),
        (tool_event(1, "Bash", {"command": "nohup uvicorn app:app &"}), EXECUTE, None),
    ])

    assert kinds_of(blocks) == [WRITE, EXECUTE]


def test_a_short_filename_still_joins_when_the_command_actually_checks_it():
    """The fix above must not cost real short names their match — ``app.js``
    is a common, legitimate entry-point filename, not just noise."""
    blocks = merge_neighbours([
        (tool_event(0, "Edit", {"file_path": "/app.js"}), WRITE, None),
        (tool_event(1, "Bash", {"command": "node --check app.js && echo OK"}), EXECUTE, None),
    ])

    assert kinds_of(blocks) == [WRITE]


def test_coordination_inside_a_test_loop_is_absorbed_too():
    blocks = merge_neighbours([
        (tool_event(0, "Edit", {"file_path": "/blocks.py"}), WRITE, None),
        (tool_event(1, "TodoWrite", {}), COORDINATION, None),
        (tool_event(2, "Bash", {"command": "pytest test_blocks.py"}), EXECUTE, None),
    ])

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 3


def test_a_test_loop_survives_the_whole_pipeline_and_is_labelled():
    """End to end via ``build_blocks``: the heuristic (no judge) reads
    ``pytest ...`` as execute, and the merged block's label says why the
    write/execute alternation collapsed."""
    session = _session([
        tool_event(0, "Edit", {"file_path": "/blocks.py"}),
        tool_event(1, "Bash", {"command": "pytest tests/test_blocks.py"}),
        tool_event(2, "Edit", {"file_path": "/blocks.py"}),
        tool_event(3, "Bash", {"command": "pytest tests/test_blocks.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [WRITE]
    assert blocks[0].message_count == 4
    assert "test development loop" in blocks[0].label


def test_merging_an_empty_timeline_yields_no_blocks():
    assert merge_neighbours([]) == []


def test_a_merged_block_takes_its_least_confident_members_score():
    """A block is only as trustworthy as its weakest classification."""
    events = [(tool_event(0, "Bash"), READ, 0.9),
              (tool_event(1, "Bash"), READ, 0.4)]
    blocks = merge_neighbours(events)

    assert blocks[0].confidence == 0.4


# ----------------------------------------------------------------------
# Agent prose — classified as coordination
# ----------------------------------------------------------------------

def prose_event(offset_seconds: int, text: str = "Let me check.") -> Event:
    return Event(uuid=f"p{offset_seconds}",
                 ts=BASE_TIME + timedelta(seconds=offset_seconds),
                 type=EV_ASSISTANT, text=text,
                 tokens=Tokens(output=25))


def test_agent_prose_is_coordination():
    assert classify.rule_kind(prose_event(0)) == COORDINATION


def test_prose_only_session_produces_a_coordination_block():
    session = _session([prose_event(0), prose_event(1)])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [COORDINATION]


def test_prose_between_different_kinds_stays_visible_as_the_boundary():
    """It separates a read from an execute, so it marks a real transition."""
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        prose_event(1, "Now let me run the tests."),
        tool_event(2, "Bash", {"command": "pytest"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ, COORDINATION, EXECUTE]


def test_prose_between_two_reads_is_absorbed():
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        prose_event(1),
        tool_event(2, "Read", {"file_path": "/b.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ]
    assert blocks[0].message_count == 3


def test_absorbed_prose_keeps_its_tokens():
    """Absorption must not lose output tokens — prose is where they live."""
    session = _session([
        tool_event(0, "Read", {"file_path": "/a.py"}),
        prose_event(1),  # 25 output tokens
        tool_event(2, "Read", {"file_path": "/b.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [READ]
    assert blocks[0].tokens.output == 25


# ----------------------------------------------------------------------
# Whole-session assembly
# ----------------------------------------------------------------------

def _session(events: list[Event]) -> Session:
    return Session(session_id="s1", project="p", events=events)


def test_build_blocks_classifies_and_merges_a_session():
    session = _session([
        user_event(0),
        tool_event(1, "Read", {"file_path": "/a.py"}),
        tool_event(2, "Read", {"file_path": "/b.py"}),
        tool_event(3, "TodoWrite", {}),
        tool_event(4, "Read", {"file_path": "/c.py"}),
        tool_event(5, "Edit", {"file_path": "/a.py"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [USER_CHAT, READ, WRITE]
    assert blocks[1].message_count == 4  # 3 reads + the absorbed TodoWrite


def test_blocks_carry_a_human_label():
    session = _session([tool_event(0, "Read", {"file_path": "/a.py"})])
    blocks = build_blocks(session, use_judge=False)

    assert blocks[0].label.startswith("read")
    assert "/a.py" in blocks[0].label


def test_a_merged_block_is_labelled_with_its_step_count():
    session = _session([tool_event(i, "Read", {"file_path": f"/{i}.py"})
                        for i in range(4)])
    blocks = build_blocks(session, use_judge=False)

    assert "4 steps" in blocks[0].label


# ----------------------------------------------------------------------
# The merge invariant
# ----------------------------------------------------------------------
#
# The whole point of merging is that no two adjacent blocks share a kind. If
# any pair does, a merge was missed and the bar shows a false boundary. These
# assert the property directly rather than spot-checking sequences.

def assert_no_adjacent_duplicates(blocks: list[Block]) -> None:
    pairs = [(blocks[i].kind, blocks[i + 1].kind)
             for i in range(len(blocks) - 1)
             if blocks[i].kind == blocks[i + 1].kind]
    assert not pairs, f"adjacent blocks share a kind: {pairs}"


@pytest.mark.parametrize("sequence", [
    [READ, READ, READ],
    [READ, COORDINATION, READ],
    [READ, COORDINATION, WRITE, COORDINATION, READ],
    [COORDINATION, READ, COORDINATION],
    [READ, WRITE, READ, WRITE, READ],
    [COORDINATION] * 5,
    [READ, COORDINATION, READ, COORDINATION, WRITE, WRITE, COORDINATION],
    [EXECUTE, COORDINATION, COORDINATION, EXECUTE, READ, COORDINATION, READ],
    [USER_CHAT, COORDINATION, USER_CHAT],
    [READ, COORDINATION, EXECUTE, EXECUTE, COORDINATION, COORDINATION, READ],
])
def test_no_two_adjacent_blocks_ever_share_a_kind(sequence):
    blocks = merge_neighbours(
        [(tool_event(i, "X"), kind, None) for i, kind in enumerate(sequence)])

    assert_no_adjacent_duplicates(blocks)


def test_the_invariant_holds_for_every_possible_short_sequence():
    """Exhaustive, not hand-picked.

    Ten chosen sequences only cover what the author thought of. This enumerates
    **every** ordering of the five non-subagent kinds up to length 6 — 19,530
    sequences — and asserts the invariant on all of them. Anything the merge
    logic mishandles is in here by construction.
    """
    from itertools import product

    kinds = [READ, WRITE, EXECUTE, COORDINATION, USER_CHAT]
    checked = 0
    for length in range(1, 7):
        for sequence in product(kinds, repeat=length):
            blocks = merge_neighbours([
                (tool_event(i, "X"), kind, None)
                for i, kind in enumerate(sequence)
            ])
            pairs = [i for i in range(len(blocks) - 1)
                     if blocks[i].kind == blocks[i + 1].kind]
            assert not pairs, (
                f"{sequence} -> {[b.kind for b in blocks]} has adjacent duplicates")
            checked += 1

    assert checked == sum(5 ** n for n in range(1, 7)) == 19530


def test_no_event_is_lost_for_any_short_sequence():
    """The same exhaustive sweep, checking conservation instead."""
    from itertools import product

    kinds = [READ, WRITE, EXECUTE, COORDINATION, USER_CHAT]
    for length in range(1, 6):
        for sequence in product(kinds, repeat=length):
            kinded = [(tool_event(i, "X"), kind, None)
                      for i, kind in enumerate(sequence)]
            blocks = merge_neighbours(kinded)

            uuids = [e.uuid for b in blocks for e in b.events]
            assert len(uuids) == length, f"{sequence} lost or duplicated an event"
            assert len(set(uuids)) == length


def test_event_order_is_preserved_for_any_short_sequence():
    """Blocks read top to bottom in time; merging must not reorder anything."""
    from itertools import product

    kinds = [READ, WRITE, EXECUTE, COORDINATION]
    for length in range(1, 7):
        for sequence in product(kinds, repeat=length):
            blocks = merge_neighbours([
                (tool_event(i, "X"), kind, None)
                for i, kind in enumerate(sequence)
            ])
            timestamps = [e.ts for b in blocks for e in b.events]
            assert timestamps == sorted(timestamps), f"{sequence} reordered events"


def test_merging_never_loses_or_duplicates_an_event():
    """Every event lands in exactly one block."""
    sequence = [READ, COORDINATION, READ, WRITE, COORDINATION, WRITE, EXECUTE]
    kinded = [(tool_event(i, "X"), kind, None) for i, kind in enumerate(sequence)]
    blocks = merge_neighbours(kinded)

    uuids = [event.uuid for block in blocks for event in block.events]
    assert sorted(uuids) == sorted(e.uuid for e, _, _ in kinded)
    assert len(uuids) == len(set(uuids)), "an event appeared in two blocks"


def test_a_whole_session_holds_the_invariant():
    session = _session([
        user_event(0),
        tool_event(1, "Read", {"file_path": "/a"}),
        prose_event(2),
        tool_event(3, "Read", {"file_path": "/b"}),
        tool_event(4, "TodoWrite"),
        tool_event(5, "Read", {"file_path": "/c"}),
        prose_event(6),
        tool_event(7, "Edit", {"file_path": "/a"}),
        tool_event(8, "Edit", {"file_path": "/b"}),
        prose_event(9),
        tool_event(10, "Bash", {"command": "pytest"}),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert_no_adjacent_duplicates(blocks)
    counted = sum(b.message_count for b in blocks)
    assert counted == len(session.events)


# ----------------------------------------------------------------------
# Subagents
# ----------------------------------------------------------------------

def _session_with_subagent() -> Session:
    return _session([
        user_event(0),
        tool_event(1, "Agent", {"description": "Find TODOs"},
                   spawned_agent_id="agent-1"),
        tool_event(2, "Grep", {"pattern": "TODO"}, agent_id="agent-1"),
        tool_event(3, "Read", {"file_path": "/a.py"}, agent_id="agent-1"),
        tool_event(9, "Edit", {"file_path": "/a.py"}),
    ])


def test_subagent_work_becomes_one_container_block():
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    containers = [b for b in blocks if b.kind == SUBAGENT]

    assert len(containers) == 1
    assert containers[0].agent_id == "agent-1"


def test_the_spawn_call_does_not_become_a_second_empty_container():
    """Regression: the ``Agent`` call classifies as SUBAGENT by rule.

    Left on the main thread it produced a second, empty subagent block beside
    the real one — the delegation counted twice, and callers taking the first
    match got the empty one.
    """
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    containers = [b for b in blocks if b.kind == SUBAGENT]

    assert len(containers) == 1
    assert containers[0].inner_blocks, "the surviving container must hold the work"


def test_the_container_starts_when_the_delegation_was_issued():
    """Anchored at the spawn, not at the child's first line."""
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    container = next(b for b in blocks if b.kind == SUBAGENT)

    assert container.t_start == BASE_TIME + timedelta(seconds=1)


def test_a_spawn_with_no_child_transcript_stays_visible():
    """The subagent file can be missing; the delegation still happened."""
    session = _session([
        tool_event(0, "Agent", {"description": "Orphan"},
                   spawned_agent_id="agent-missing"),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [SUBAGENT]


def test_the_container_holds_the_subagents_own_blocks():
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    container = next(b for b in blocks if b.kind == SUBAGENT)

    assert kinds_of(container.inner_blocks) == [READ]  # Grep + Read merged
    assert container.inner_blocks[0].message_count == 2


def test_the_container_is_labelled_with_the_task_it_was_given():
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    container = next(b for b in blocks if b.kind == SUBAGENT)

    assert container.description == "Find TODOs"


def test_subagent_events_are_excluded_from_the_parents_own_blocks():
    """Otherwise the child's tools would also appear on the main timeline."""
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    main = [b for b in blocks if b.kind != SUBAGENT]

    assert all(not e.is_subagent for b in main for e in b.events)


def _session_with_two_subagents() -> Session:
    """Two agents spawned back to back — the ordinary fan-out shape."""
    return _session([
        tool_event(0, "Agent", {"description": "First task"},
                   spawned_agent_id="agent-1"),
        tool_event(1, "Read", {"file_path": "/a"}, agent_id="agent-1"),
        tool_event(2, "Agent", {"description": "Second task"},
                   spawned_agent_id="agent-2"),
        tool_event(3, "Read", {"file_path": "/b"}, agent_id="agent-2"),
    ])


def test_each_subagent_gets_its_own_description():
    """Matching on the first Agent call would label every child identically."""
    blocks = build_blocks(_session_with_two_subagents(), use_judge=False)
    band = next(b for b in blocks if b.kind == SUBAGENT)
    described = {agent.agent_id: agent.description for agent in band.agents}

    assert described == {"agent-1": "First task", "agent-2": "Second task"}


# ----------------------------------------------------------------------
# Grouping adjacent subagents
# ----------------------------------------------------------------------
# Spawning three agents in one message is one act of delegation. Drawn as
# three separate stripes it reads as three separate decisions, so a run of
# adjacent containers becomes a single band and the agents survive inside it.

def test_subagents_spawned_back_to_back_become_one_band():
    blocks = build_blocks(_session_with_two_subagents(), use_judge=False)
    bands = [b for b in blocks if b.kind == SUBAGENT]

    assert len(bands) == 1
    assert len(bands[0].agents) == 2


def test_a_band_keeps_every_agents_steps_for_the_bar_to_paint():
    blocks = build_blocks(_session_with_two_subagents(), use_judge=False)
    band = next(b for b in blocks if b.kind == SUBAGENT)

    # `inner_blocks` stays flat — it is what the bar draws inside the band —
    # while `agents` keeps the same work split per agent.
    assert len(band.inner_blocks) == 2
    assert [len(agent.inner_blocks) for agent in band.agents] == [1, 1]


def test_a_band_of_several_agents_has_no_single_agent_id():
    blocks = build_blocks(_session_with_two_subagents(), use_judge=False)
    band = next(b for b in blocks if b.kind == SUBAGENT)

    # Claiming one of the two would silently mislabel the other.
    assert band.agent_id is None
    assert "2 agents" in band.label


def test_a_lone_subagent_is_still_wrapped_as_a_band_of_one():
    """So the detail page has one shape to render, not two."""
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    band = next(b for b in blocks if b.kind == SUBAGENT)

    assert len(band.agents) == 1
    # A band of one keeps the agent's identity, so nothing is lost by wrapping.
    assert band.agent_id == "agent-1"
    assert band.description == "Find TODOs"


def test_subagents_separated_by_other_work_stay_separate_bands():
    """Two delegations twenty blocks apart are two decisions, not one."""
    session = _session([
        tool_event(0, "Agent", {"description": "First"},
                   spawned_agent_id="agent-1"),
        tool_event(1, "Read", {"file_path": "/a"}, agent_id="agent-1"),
        tool_event(2, "Bash", {"command": "pytest"}),
        tool_event(3, "Agent", {"description": "Second"},
                   spawned_agent_id="agent-2"),
        tool_event(4, "Read", {"file_path": "/b"}, agent_id="agent-2"),
    ])
    bands = [b for b in build_blocks(session, use_judge=False)
             if b.kind == SUBAGENT]

    assert len(bands) == 2
    assert [len(band.agents) for band in bands] == [1, 1]


def test_grouping_does_not_swallow_the_work_between_two_bands():
    session = _session([
        tool_event(0, "Agent", {"description": "First"},
                   spawned_agent_id="agent-1"),
        tool_event(1, "Read", {"file_path": "/a"}, agent_id="agent-1"),
        tool_event(2, "Bash", {"command": "pytest"}),
        tool_event(3, "Agent", {"description": "Second"},
                   spawned_agent_id="agent-2"),
        tool_event(4, "Read", {"file_path": "/b"}, agent_id="agent-2"),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert kinds_of(blocks) == [SUBAGENT, EXECUTE, SUBAGENT]


def test_a_band_covers_every_event_of_the_agents_it_holds():
    """Nothing is dropped by the merge — the band is the sum of its agents."""
    blocks = build_blocks(_session_with_two_subagents(), use_judge=False)
    band = next(b for b in blocks if b.kind == SUBAGENT)
    from_agents = sum(agent.message_count for agent in band.agents)

    assert band.message_count == from_agents == 4  # 2 spawns + 2 child reads


def test_containers_sit_in_timeline_order_among_the_parents_blocks():
    blocks = build_blocks(_session_with_subagent(), use_judge=False)
    starts = [b.t_start for b in blocks]

    assert starts == sorted(starts)


# ----------------------------------------------------------------------
# Y-axis metrics
# ----------------------------------------------------------------------

def test_every_declared_metric_is_computable():
    block = Block(kind=READ, events=[
        tool_event(0, "Read", tokens=Tokens(input=10, output=5)),
        tool_event(30, "Read", tokens=Tokens(input=1, output=1)),
    ])

    for metric in Y_METRICS:
        assert block_metric(block, metric) >= 0


def test_metrics_measure_different_things():
    block = Block(kind=READ, events=[
        tool_event(0, "Read", tokens=Tokens(input=10, output=5)),
        tool_event(30, "Read", tokens=Tokens(input=1, output=1)),
    ])

    assert block_metric(block, "time") == 30.0
    assert block_metric(block, "messages") == 2.0
    assert block_metric(block, "tokens") == 17.0


def test_an_unknown_metric_falls_back_to_tokens():
    block = Block(kind=READ, events=[tool_event(0, "Read",
                                                tokens=Tokens(input=7))])

    assert block_metric(block, "nonsense") == 7.0
