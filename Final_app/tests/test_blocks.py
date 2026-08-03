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


def test_each_subagent_gets_its_own_description():
    """Matching on the first Agent call would label every child identically."""
    session = _session([
        tool_event(0, "Agent", {"description": "First task"},
                   spawned_agent_id="agent-1"),
        tool_event(1, "Read", {"file_path": "/a"}, agent_id="agent-1"),
        tool_event(2, "Agent", {"description": "Second task"},
                   spawned_agent_id="agent-2"),
        tool_event(3, "Read", {"file_path": "/b"}, agent_id="agent-2"),
    ])
    containers = {b.agent_id: b for b in build_blocks(session, use_judge=False)
                  if b.kind == SUBAGENT}

    assert containers["agent-1"].description == "First task"
    assert containers["agent-2"].description == "Second task"


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
