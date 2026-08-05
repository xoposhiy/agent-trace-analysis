"""Cost attribution: every billed working token lands on exactly one block.

The invariant these all circle is a single equation —

    sum(block.attributed_tokens) + session.orphaned_tokens.working
        == session.tokens.working

exactly, for any transcript. If that ever drifts, the bar and the header stop
agreeing and neither can be trusted.

The behaviour underneath it is that a cost lands on whatever *caused* it, which
for a tool call means its result as well as its call: a ``Read`` emits ~40
tokens and drags thousands back into the next prompt, and the second number is
the one that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Final_app.adapters import claude_code
from Final_app.analysis.attribution import (
    CHARS_PER_TOKEN_PROSE,
    attribute,
    calibrate,
    generated_weight,
    result_weight,
    split_exact,
)
from Final_app.analysis.blocks import build_blocks

from .conftest import (
    PROJECT_SLUG,
    assistant_mixed_line,
    assistant_thinking_line,
    assistant_text_line,
    assistant_tool_line,
    tool_result_line,
    user_line,
    write_transcript,
)


def _loaded(claude_home: Path, lines: list[dict]):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)
    session = claude_code.load_session(PROJECT_SLUG, path)
    session.overhead_tokens = attribute(session)
    return session


# ----------------------------------------------------------------------
# The exact division
# ----------------------------------------------------------------------

@pytest.mark.parametrize("amount,weights", [
    (1000, [1, 1, 1]),
    (7, [500, 3, 1]),
    (3, [1, 1, 1, 1, 1]),
    (104_729, [13, 7, 991]),
])
def test_dividing_a_cost_neither_invents_nor_loses_a_token(amount, weights):
    shares = split_exact(amount, weights)

    assert sum(shares) == amount
    assert len(shares) == len(weights)


def test_a_cost_nobody_can_be_weighed_for_is_still_divided_whole():
    assert sum(split_exact(10, [0, 0])) == 10


def test_nothing_to_divide_gives_nobody_anything():
    assert split_exact(0, [5, 5]) == [0, 0]


# ----------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------

def test_a_tool_calls_result_is_weighed_from_the_whole_envelope(claude_home: Path):
    """A Read's file body lives in ``toolUseResult.file.content``.

    It never reaches ``ToolResult.output``, and it is the entire reason a Read
    is expensive. Measured on a real transcript, ``output`` retained 183,415 of
    5,011,535 result characters — 3.7%.
    """
    body = "x" * 40_000
    session = _loaded(claude_home, [
        user_line("u1", "read it", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z", output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"type": "text",
                                   "file": {"filePath": "/repo/big.py",
                                            "numLines": 900, "content": body}}),
    ])
    read = next(e for e in session.events if e.tool and e.tool.name == "Read")

    assert len(read.tool.result.output) < 100, "output should stay small"
    assert read.tool.result.size_chars > 40_000
    assert result_weight(read) > 10 * generated_weight(read), \
        "the file body should dwarf the filename that fetched it"


def test_a_write_is_weighed_on_its_call_not_its_result(claude_home: Path):
    """Write is Read's mirror: the payload is in the call, the result is 'ok'."""
    session = _loaded(claude_home, [
        user_line("u1", "write it", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Write",
                            {"file_path": "/repo/new.py", "content": "y" * 40_000},
                            "2026-08-01T10:00:01.000Z", output_tokens=9000),
        tool_result_line("r1", "t1", "File created", "2026-08-01T10:00:02.000Z"),
    ])
    write = next(e for e in session.events if e.tool and e.tool.name == "Write")

    assert generated_weight(write) > 10 * result_weight(write)


# ----------------------------------------------------------------------
# The invariant
# ----------------------------------------------------------------------

def _mixed_lines() -> list[dict]:
    return [
        user_line("u1", "please fix the parser", "2026-08-01T10:00:00.000Z"),
        assistant_mixed_line(
            "a1", "msg1", "Reading the parser first.",
            [("t1", "Read", {"file_path": "/repo/parse.py"})],
            "2026-08-01T10:00:01.000Z",
            input_tokens=1200, output_tokens=310, cache_creation=800),
        tool_result_line("r1", "t1", "src", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/parse.py",
                                            "numLines": 400,
                                            "content": "z" * 20_000}}),
        assistant_tool_line("a2", "msg2", "t2", "Edit",
                            {"file_path": "/repo/parse.py", "old_string": "a",
                             "new_string": "b"},
                            "2026-08-01T10:00:03.000Z",
                            input_tokens=40, output_tokens=520, cache_creation=6200),
        tool_result_line("r2", "t2", "edited", "2026-08-01T10:00:04.000Z"),
        user_line("u2", "now run the tests", "2026-08-01T10:00:05.000Z"),
        assistant_tool_line("a3", "msg3", "t3", "Bash", {"command": "pytest -q"},
                            "2026-08-01T10:00:06.000Z",
                            input_tokens=20, output_tokens=95, cache_creation=900),
        tool_result_line("r3", "t3", "3 passed", "2026-08-01T10:00:07.000Z"),
        assistant_text_line("a4", "msg4", "All green.", "2026-08-01T10:00:08.000Z",
                            input_tokens=15, output_tokens=60, cache_creation=140),
    ]


def test_every_billed_token_lands_on_a_block(claude_home: Path):
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    in_blocks = sum(b.attributed_tokens for b in blocks)

    assert session.overhead_tokens == 0, "nothing should be left unplaced"
    assert in_blocks + session.orphaned_tokens.working == session.tokens.working


def test_the_total_holds_when_cache_reads_are_enormous(claude_home: Path):
    """``cache_read`` is excluded by design, so it must not shift the total."""
    lines = [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=10, output_tokens=50,
                            cache_read=9_000_000, cache_creation=300),
        tool_result_line("r1", "t1", "src", "2026-08-01T10:00:02.000Z"),
        assistant_text_line("a2", "msg2", "Done.", "2026-08-01T10:00:03.000Z",
                            output_tokens=20, cache_read=9_500_000),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    assert sum(b.attributed_tokens for b in blocks) == session.tokens.working
    assert session.tokens.cache_read == 18_500_000


def test_a_transcript_with_no_assistant_turn_attributes_nothing(claude_home: Path):
    session = _loaded(claude_home, [
        user_line("u1", "hello?", "2026-08-01T10:00:00.000Z"),
    ])

    assert session.tokens.working == 0
    assert session.overhead_tokens == 0


# ----------------------------------------------------------------------
# Where the cost lands
# ----------------------------------------------------------------------

def test_a_read_is_charged_for_the_file_it_pulled_into_context(claude_home: Path):
    """The whole point: a Read's cost is its result, not its filename."""
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    read = next(b for b in blocks if b.kind == "read")

    # The Read emitted ~40 tokens but dragged 20k characters into the next
    # prompt, which that call paid 6,200 cache_creation for.
    assert read.attributed_tokens > 1000, \
        f"a 20k-char file read came out at {read.attributed_tokens} tokens"


def test_a_user_prompt_carries_its_own_cost(claude_home: Path):
    """User lines carry no ``usage`` at all — 0 of 268 in a real transcript.

    Their cost only exists as part of the next call's prompt, so attribution is
    the only way a "chatting with user" block gets a number.
    """
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    prompts = [b for b in blocks if b.kind == "user_chat"]

    assert prompts, "expected the user's messages to become blocks"
    assert all(b.attributed_tokens > 0 for b in prompts), \
        [(b.label, b.attributed_tokens) for b in prompts]


def test_the_opening_prompt_is_not_charged_for_the_system_prompt(claude_home: Path):
    """The first call's prompt is mostly system prompt and tool definitions.

    Charging that to whichever user message preceded it made a 200-token
    opening prompt the single largest block on a real session's bar.
    """
    lines = [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
        # A first call billed far more than the two-character prompt explains.
        assistant_text_line("a1", "msg1", "Hello.", "2026-08-01T10:00:01.000Z",
                            input_tokens=22_000, output_tokens=40),
        assistant_tool_line("a2", "msg2", "t1", "Bash", {"command": "ls"},
                            "2026-08-01T10:00:02.000Z",
                            output_tokens=5000, cache_creation=5000),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    prompt = next(b for b in blocks if b.kind == "user_chat")
    biggest = max(blocks, key=lambda b: b.attributed_tokens)

    assert prompt is not biggest, \
        f'"hi" became the largest block at {prompt.attributed_tokens} tokens'
    assert sum(b.attributed_tokens for b in blocks) == session.tokens.working


def test_subagent_costs_stay_inside_the_subagent(claude_home: Path):
    """Threads attribute independently — a parent call must not pay for a child."""
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    assert sum(b.attributed_tokens for b in blocks) == session.tokens.working


# ----------------------------------------------------------------------
# Every tool call is registered in the block that holds it
# ----------------------------------------------------------------------
# The reported symptom that started this: a block covering a tool call showed
# no tokens for it. These pin the chain end to end — the call is in a block,
# the call has a cost, and the block's number is the sum of its members with
# nothing dropped on the way.

def _tool_heavy_lines() -> list[dict]:
    """One of every kind of call, each with a result of a different size."""
    return [
        user_line("u1", "do the work", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=900, output_tokens=45),
        tool_result_line("r1", "t1", "short", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/a.py",
                                            "numLines": 300,
                                            "content": "a" * 12_000}}),
        assistant_tool_line("a2", "msg2", "t2", "Grep", {"pattern": "def load"},
                            "2026-08-01T10:00:03.000Z",
                            output_tokens=30, cache_creation=4100),
        tool_result_line("r2", "t2", "3 matches", "2026-08-01T10:00:04.000Z"),
        assistant_mixed_line(
            "a3", "msg3", "Now editing both files.",
            [("t3", "Edit", {"file_path": "/repo/a.py", "old_string": "p",
                             "new_string": "q"}),
             ("t4", "Edit", {"file_path": "/repo/b.py", "old_string": "r",
                             "new_string": "s"})],
            "2026-08-01T10:00:05.000Z",
            output_tokens=880, cache_creation=200),
        tool_result_line("r3", "t3", "edited a", "2026-08-01T10:00:06.000Z"),
        tool_result_line("r4", "t4", "edited b", "2026-08-01T10:00:07.000Z"),
        assistant_tool_line("a4", "msg4", "t5", "Bash", {"command": "pytest -q"},
                            "2026-08-01T10:00:08.000Z",
                            output_tokens=70, cache_creation=1500),
        tool_result_line("r5", "t5", "12 passed", "2026-08-01T10:00:09.000Z"),
        assistant_text_line("a5", "msg5", "Done.", "2026-08-01T10:00:10.000Z",
                            output_tokens=25, cache_creation=600),
    ]


def test_every_tool_call_ends_up_in_exactly_one_block(claude_home: Path):
    session = _loaded(claude_home, _tool_heavy_lines())
    blocks = build_blocks(session, use_judge=False)

    called = {e.tool.id for e in session.events if e.tool is not None}
    in_blocks = [e.tool.id for b in blocks for e in b.events if e.tool is not None]

    assert len(called) == 5
    assert sorted(in_blocks) == sorted(called)
    assert len(in_blocks) == len(set(in_blocks)), "a call was counted twice"


def test_every_tool_call_carries_a_token_cost(claude_home: Path):
    """No tool call may sit in a block contributing nothing."""
    session = _loaded(claude_home, _tool_heavy_lines())

    calls = [e for e in session.events if e.tool is not None]

    assert len(calls) == 5
    assert all(e.attributed_tokens > 0 for e in calls), \
        [(e.tool.name, e.attributed_tokens) for e in calls]


def test_a_blocks_tokens_are_exactly_its_members_tokens(claude_home: Path):
    """The block number must be the sum of what is inside it — nothing else."""
    session = _loaded(claude_home, _tool_heavy_lines())
    blocks = build_blocks(session, use_judge=False)

    for block in blocks:
        assert block.attributed_tokens == sum(
            e.attributed_tokens for e in block.events), block.label


def test_a_block_holding_tool_calls_costs_at_least_what_they_cost(claude_home: Path):
    session = _loaded(claude_home, _tool_heavy_lines())
    blocks = build_blocks(session, use_judge=False)

    checked = 0
    for block in blocks:
        calls = [e for e in block.events if e.tool is not None]
        if not calls:
            continue
        checked += 1
        assert block.attributed_tokens >= sum(e.attributed_tokens for e in calls)

    assert checked >= 3, "expected several blocks to hold tool calls"


def test_a_single_step_tool_block_reports_that_calls_own_cost(claude_home: Path):
    """A one-call block is the strict case: its number IS that call's number."""
    session = _loaded(claude_home, _tool_heavy_lines())
    blocks = build_blocks(session, use_judge=False)

    singles = [b for b in blocks
               if b.message_count == 1 and b.events[0].tool is not None]

    assert singles, "expected at least one single-call block"
    for block in singles:
        assert block.attributed_tokens == block.events[0].attributed_tokens
        assert block.attributed_tokens > 0, block.label


def test_two_calls_in_one_message_are_registered_separately(claude_home: Path):
    """Parallel tool use: one message, two Edits, two independent costs."""
    session = _loaded(claude_home, _tool_heavy_lines())

    edits = [e for e in session.events
             if e.tool is not None and e.tool.name == "Edit"]

    assert len(edits) == 2
    assert all(e.attributed_tokens > 0 for e in edits)
    assert edits[0].message_id == edits[1].message_id


# ----------------------------------------------------------------------
# Per-session calibration
# ----------------------------------------------------------------------
# Characters per token is a property of the content, not of the tool: a session
# of minified JSON tokenises nothing like a session of English prose. The
# defaults are what one sample of transcripts happened to show, so each session
# re-derives its own from its own billing and only falls back when it cannot.

def _prose_lines(count: int, chars_per_token: float) -> list[dict]:
    """``count`` prose-only messages billed at a chosen characters-per-token."""
    lines = [user_line("u1", "go", "2026-08-01T10:00:00.000Z")]
    for index in range(count):
        text = "word " * 200                       # 1000 characters
        lines.append(assistant_text_line(
            f"a{index}", f"msg{index}", text,
            f"2026-08-01T10:{index + 1:02d}:00.000Z",
            output_tokens=round(len(text) / chars_per_token)))
    return lines


def test_a_session_derives_its_own_characters_per_token(claude_home: Path):
    """A session that tokenises densely must not be sized by someone else's."""
    session = _loaded(claude_home, _prose_lines(12, chars_per_token=8.0))
    calibration = calibrate(session)

    assert calibration.prose_samples == 12
    assert calibration.prose == pytest.approx(8.0, rel=0.05)
    assert calibration.prose != CHARS_PER_TOKEN_PROSE


def test_too_few_samples_falls_back_to_the_default(claude_home: Path):
    """Two odd messages must not redefine a session's sizing."""
    session = _loaded(claude_home, _prose_lines(2, chars_per_token=8.0))
    calibration = calibrate(session)

    assert calibration.prose_samples == 2
    assert calibration.prose == CHARS_PER_TOKEN_PROSE


def test_messages_with_thinking_are_excluded_from_calibration(claude_home: Path):
    """Thinking is billed but stored empty, so its message cannot be measured."""
    lines = _prose_lines(12, chars_per_token=8.0)
    # Same message id as the first prose message, so it is that message's own
    # thinking rather than a separate turn.
    lines.insert(1, assistant_thinking_line("t0", "msg0", "2026-08-01T10:00:30.000Z"))
    session = _loaded(claude_home, lines)
    calibration = calibrate(session)

    assert "msg0" in session.thinking_message_ids
    assert calibration.prose_samples == 11, "the thinking message should be skipped"


def test_calibration_changes_the_weights_it_feeds(claude_home: Path):
    session = _loaded(claude_home, _prose_lines(12, chars_per_token=8.0))
    prose = next(e for e in session.events if e.type == "assistant")

    dense = calibrate(session)

    assert generated_weight(prose, dense) < generated_weight(prose)


# ----------------------------------------------------------------------
# No real content reports zero
# ----------------------------------------------------------------------

def test_a_starved_prompt_still_gets_a_token(claude_home: Path):
    """Under-supply must scale everyone down, never zero the smallest.

    A session can be billed less ``input + cache_creation`` than its content
    accounts for, because cache reads covered the rest. Everything is then
    scaled proportionally and short pieces round away — real user messages were
    showing "0 tokens".
    """
    lines = [
        user_line("u1", "start", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z", output_tokens=400),
        tool_result_line("r1", "t1", "x", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/a.py",
                                            "numLines": 9000,
                                            "content": "q" * 400_000}}),
        # A short prompt against a huge backlog, and almost nothing billed.
        user_line("u2", "and now?", "2026-08-01T10:00:03.000Z"),
        assistant_text_line("a2", "msg2", "Done.", "2026-08-01T10:00:04.000Z",
                            input_tokens=2, output_tokens=5, cache_creation=3),
    ]
    session = _loaded(claude_home, lines)
    prompt = next(e for e in session.events if e.text == "and now?")

    assert prompt.attributed_tokens > 0, "a real prompt must never report zero"


def test_the_floor_does_not_change_the_session_total(claude_home: Path):
    """Tokens for a starved Event are taken from the largest, not invented."""
    lines = [
        user_line("u1", "start", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z", output_tokens=400),
        tool_result_line("r1", "t1", "x", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/a.py",
                                            "numLines": 9000,
                                            "content": "q" * 400_000}}),
        user_line("u2", "and now?", "2026-08-01T10:00:03.000Z"),
        assistant_text_line("a2", "msg2", "Done.", "2026-08-01T10:00:04.000Z",
                            input_tokens=2, output_tokens=5, cache_creation=3),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    assert sum(b.attributed_tokens for b in blocks) == session.tokens.working


def test_a_session_that_was_never_billed_stays_at_zero(claude_home: Path):
    """The one honest zero: no assistant reply, so nothing was ever charged.

    Inventing a token here would make the blocks disagree with a header that
    correctly reads 0.
    """
    session = _loaded(claude_home, [
        user_line("u1", "hello, anyone there?", "2026-08-01T10:00:00.000Z"),
    ])
    blocks = build_blocks(session, use_judge=False)

    assert session.tokens.working == 0
    assert sum(b.attributed_tokens for b in blocks) == 0


def test_the_floor_never_pushes_a_donor_to_zero(claude_home: Path):
    """Taking from the largest must not simply move the problem."""
    session = _loaded(claude_home, _mixed_lines())
    paying = [e for e in session.events if e.attributed_tokens > 0]

    assert all(e.attributed_tokens >= 1 for e in paying)


def test_no_block_of_real_work_is_ever_zero(claude_home: Path):
    session = _loaded(claude_home, _tool_heavy_lines())
    blocks = build_blocks(session, use_judge=False)

    work = [b for b in blocks if b.kind in ("read", "write", "execute")]

    assert work
    assert all(b.attributed_tokens > 0 for b in work), \
        [(b.kind, b.label, b.attributed_tokens) for b in work]
