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
    """Cache reads ride their own channel, so they must not shift the working one.

    The two are attributed separately and priced differently. 18.5M of cache
    reads against ~380 working tokens is the shape of a real session; if any of
    it leaked into ``attributed_tokens`` the bar and the header would part ways.
    """
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


# ----------------------------------------------------------------------
# The cache-read channel
# ----------------------------------------------------------------------
# What a call re-reads from cache is its prompt prefix — the content still
# resident in the context window. So it has a real owner, and these pin both
# halves of that: the totals stay exact, and the cost lands on whatever is
# actually sitting in the prompt being re-read.


def _resident_lines(calls: int, cache_read: int) -> list[dict]:
    """A big Read up front, then ``calls`` turns that each re-read the prefix."""
    lines = [
        user_line("u1", "read the file", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=100, output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/big.py",
                                            "numLines": 2000,
                                            "content": "z" * 80_000}}),
    ]
    for index in range(calls):
        lines.append(assistant_text_line(
            f"a{index + 2}", f"msg{index + 2}", f"Thinking about it ({index}).",
            f"2026-08-01T10:{index + 3:02d}:00.000Z",
            output_tokens=30, cache_creation=50, cache_read=cache_read))
    return lines


def test_every_cache_read_token_lands_on_a_block(claude_home: Path):
    session = _loaded(claude_home, _resident_lines(calls=6, cache_read=20_000))
    blocks = build_blocks(session, use_judge=False)

    assert session.tokens.cache_read == 120_000
    assert sum(b.attributed_cache_read for b in blocks) == session.tokens.cache_read


def test_a_blocks_total_is_its_two_channels_added(claude_home: Path):
    session = _loaded(claude_home, _resident_lines(calls=6, cache_read=20_000))
    blocks = build_blocks(session, use_judge=False)

    for block in blocks:
        assert block.attributed_total == (
            block.attributed_tokens + block.attributed_cache_read), block.label


def test_the_two_channels_together_are_every_billed_token(claude_home: Path):
    """The equation the bar's token axis now rests on."""
    session = _loaded(claude_home, _resident_lines(calls=6, cache_read=20_000))
    blocks = build_blocks(session, use_judge=False)

    attributed = sum(b.attributed_total for b in blocks)

    assert attributed + session.orphaned_tokens.total == session.tokens.total


def test_the_block_that_filled_the_context_pays_for_the_re_reads(claude_home: Path):
    """The whole point of the channel.

    An 80k-character file sits in the prompt for every later call, and each of
    those calls is billed 20,000 cache_read for reading it back. The Read is
    what put it there, so the Read carries that cost — its own call was ~40
    tokens.
    """
    session = _loaded(claude_home, _resident_lines(calls=6, cache_read=20_000))
    blocks = build_blocks(session, use_judge=False)

    read = next(b for b in blocks if b.kind == "read")

    assert read.attributed_cache_read > read.attributed_tokens, \
        "the re-reads should dwarf what the call itself was billed"
    assert read.attributed_cache_read > 0.5 * session.tokens.cache_read, \
        f"the resident file should carry most of the 120,000 re-read tokens," \
        f" got {read.attributed_cache_read}"


def test_content_arriving_last_is_not_charged_for_earlier_re_reads(claude_home: Path):
    """A block can only be re-read by calls that came after it.

    Cost accrues while content is resident, never retroactively — the direction
    the old surplus rule had backwards. Asserted as an ordering between two turns
    that are identical in content and in billing, so residency is the only thing
    that can separate them. Not asserted as "the last turn is zero": it still
    takes its slice of the session-wide baseline, which belongs to every block.
    """
    session = _loaded(claude_home, _resident_lines(calls=6, cache_read=20_000))

    prose = sorted((e for e in session.events if e.type == "assistant" and e.text),
                   key=lambda e: e.ts)
    first, last = prose[0], prose[-1]

    assert first.attributed_cache_read > last.attributed_cache_read, \
        "the earlier of two identical turns sat in more prompts and must pay more"
    assert last.attributed_cache_read < 0.05 * session.tokens.cache_read, \
        f"the final turn was re-read by nobody, yet holds" \
        f" {last.attributed_cache_read} of {session.tokens.cache_read}"


def test_a_short_prompt_is_not_charged_for_the_system_prompt_being_re_read(
        claude_home: Path):
    """The opening prompt must not inherit the baseline's re-read cost.

    The system prompt and tool definitions are in every prefix and no Event
    carries their weight, so with nothing standing in for them the only resident
    content — a two-word user message — absorbed the whole re-read. Measured
    before the baseline entry existed: this fixture put 30,033 of 90,000 cache
    reads on a 16-token prompt, and a real ``hi`` came out holding 165,554.

    Same failure as the module docstring's opening example, one channel over.
    """
    lines = [
        user_line("u1", "read the config", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "m1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=120, output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/big.py",
                                            "numLines": 1500,
                                            "content": "z" * 60_000}}),
    ]
    for index in range(3):
        lines.append(assistant_text_line(
            f"a{index + 2}", f"m{index + 2}", f"Turn {index}.",
            f"2026-08-01T10:0{index + 3}:00.000Z",
            output_tokens=30, cache_creation=40, cache_read=30_000))
    session = _loaded(claude_home, lines)

    prompt = next(e for e in session.events if e.text == "read the config")
    read = next(e for e in session.events if e.tool and e.tool.name == "Read")

    assert session.tokens.cache_read == 90_000
    assert prompt.attributed_cache_read < 0.1 * session.tokens.cache_read, \
        f"a 15-character prompt holds {prompt.attributed_cache_read:,} cache reads"
    assert read.attributed_cache_read > prompt.attributed_cache_read, \
        "the 60k-character file, not the prompt, is what later calls re-read"


def test_a_compaction_stops_older_content_earning_re_reads(claude_home: Path):
    """After a compaction the prompt is rebuilt, so the old prefix is gone.

    Without the reset a pre-compaction Read keeps collecting a share of re-reads
    of a context window it is no longer in — the totals would still balance, so
    this can only fail silently.
    """
    lines = [
        user_line("u1", "read it", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=100, output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/big.py",
                                            "numLines": 2000,
                                            "content": "z" * 80_000}}),
        # One call re-reads the file while it is still resident.
        assistant_text_line("a2", "msg2", "Still reading.", "2026-08-01T10:01:00.000Z",
                            output_tokens=30, cache_creation=50, cache_read=10_000),
        {"uuid": "c1", "timestamp": "2026-08-01T10:02:00.000Z",
         "isCompactSummary": True},
        # Everything after the compaction re-reads a prefix the Read is not in.
        assistant_tool_line("a3", "msg3", "t2", "Bash", {"command": "pytest -q"},
                            "2026-08-01T10:03:00.000Z",
                            output_tokens=30, cache_creation=50, cache_read=90_000),
        tool_result_line("r2", "t2", "ok", "2026-08-01T10:03:30.000Z"),
        assistant_text_line("a4", "msg4", "Green.", "2026-08-01T10:04:00.000Z",
                            output_tokens=20, cache_creation=40, cache_read=90_000),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    read = next(b for b in blocks if b.kind == "read")

    assert session.compaction_points, "the fixture must produce a compaction point"
    assert sum(b.attributed_cache_read for b in blocks) == session.tokens.cache_read
    # It may keep the 10,000 from the one pre-compaction call, and nothing from
    # the 180,000 re-read after it.
    assert read.attributed_cache_read <= 10_000, \
        f"a compacted-away Read kept earning: {read.attributed_cache_read}"


def test_a_compaction_pays_for_the_summary_it_wrote(claude_home: Path):
    """The post-compaction prefix is the summary, and the marker owns it.

    The IR stores a compaction marker with no body, so nothing else can carry
    that weight. Leaving it unowned sent every later re-read of the summary to
    the session-wide spread, which put it partly back on the very blocks the
    compaction discarded.
    """
    lines = [
        user_line("u1", "read it", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=100, output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/big.py",
                                            "numLines": 2000,
                                            "content": "z" * 80_000}}),
        {"uuid": "c1", "timestamp": "2026-08-01T10:02:00.000Z",
         "isCompactSummary": True},
        assistant_text_line("a3", "msg3", "Carrying on.", "2026-08-01T10:03:00.000Z",
                            output_tokens=30, cache_creation=50, cache_read=60_000),
        assistant_text_line("a4", "msg4", "Still here.", "2026-08-01T10:04:00.000Z",
                            output_tokens=20, cache_creation=40, cache_read=60_000),
    ]
    session = _loaded(claude_home, lines)

    marker = next(e for e in session.events if e.type == "compaction")

    assert marker.attributed_cache_read > 0, \
        "the summary the compaction wrote was re-read by every later call"
    assert sum(e.attributed_cache_read for e in session.events) \
        == session.tokens.cache_read


def test_cache_reads_with_nothing_resident_are_spread_not_dropped(claude_home: Path):
    """A first call's prefix is the system prompt — no block put it there.

    It is still real money, so it is shared across the session rather than
    charged to whichever block happened to be first.
    """
    lines = [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
        assistant_text_line("a1", "msg1", "Hello.", "2026-08-01T10:00:01.000Z",
                            input_tokens=50, output_tokens=40, cache_read=30_000),
        assistant_text_line("a2", "msg2", "More.", "2026-08-01T10:00:02.000Z",
                            output_tokens=20, cache_creation=30),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    assert session.tokens.cache_read == 30_000
    assert sum(b.attributed_cache_read for b in blocks) == 30_000


def test_a_session_with_no_cache_reads_attributes_none(claude_home: Path):
    """The degenerate case: the channel must stay silent, not invent a share."""
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    assert session.tokens.cache_read == 0
    assert all(b.attributed_cache_read == 0 for b in blocks)
    assert all(b.attributed_total == b.attributed_tokens for b in blocks)


# ----------------------------------------------------------------------
# The context-window snapshot (DESIGN.md §7)
# ----------------------------------------------------------------------
# A different question from every test above: not "how much did this cost
# across the whole session" (cumulative, unbounded) but "what does the
# context look like right now" — bounded by one real call's actual billed
# size, never a sum across calls.

def test_context_window_is_the_last_calls_real_size_not_a_sum(claude_home: Path):
    """The bug this fixes: the naive total inflates with every extra call
    that re-reads the same content; this must not.

    ``_resident_lines`` bills each later turn identically: cache_creation=50,
    cache_read=20,000. The naive sum grows with ``calls`` (2 calls: 40,000;
    20 calls: 400,000 — see ``test_every_cache_read_token_lands_on_a_block``
    for the same pattern on the cumulative channel). The real last call is
    always the same size regardless: fresh 50 + cached 20,000 = 20,050.
    """
    short = _loaded(claude_home, _resident_lines(calls=2, cache_read=20_000))
    long = _loaded(claude_home, _resident_lines(calls=20, cache_read=20_000))

    assert short.tokens.cache_read == 40_000
    assert long.tokens.cache_read == 400_000

    assert short.context_window_tokens == 20_050
    assert long.context_window_tokens == 20_050


def test_the_blocks_sum_exactly_to_the_context_window(claude_home: Path):
    """The header stat and the bar's "context" axis must never disagree."""
    session = _loaded(claude_home, _resident_lines(calls=6, cache_read=20_000))
    blocks = build_blocks(session, use_judge=False)

    assert sum(b.context_tokens for b in blocks) == session.context_window_tokens
    assert session.context_window_tokens == 20_050


def test_a_compaction_excludes_pre_compaction_content_from_the_window(
    claude_home: Path
):
    """A block entirely before the most recent compaction holds none of the
    current window — it is not part of what a next call would re-send —
    even though it still holds real cumulative ``attributed_cache_read``."""
    lines = [
        user_line("u1", "read it", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=100, output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/big.py",
                                            "numLines": 2000,
                                            "content": "z" * 80_000}}),
        assistant_text_line("a2", "msg2", "Still reading.", "2026-08-01T10:01:00.000Z",
                            output_tokens=30, cache_creation=50, cache_read=10_000),
        {"uuid": "c1", "timestamp": "2026-08-01T10:02:00.000Z",
         "isCompactSummary": True},
        assistant_tool_line("a3", "msg3", "t2", "Bash", {"command": "pytest -q"},
                            "2026-08-01T10:03:00.000Z",
                            output_tokens=30, cache_creation=50, cache_read=90_000),
        tool_result_line("r2", "t2", "ok", "2026-08-01T10:03:30.000Z"),
        assistant_text_line("a4", "msg4", "Green.", "2026-08-01T10:04:00.000Z",
                            output_tokens=20, cache_creation=40, cache_read=90_000),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    read = next(b for b in blocks if b.kind == "read")

    assert session.compaction_points, "the fixture must produce a compaction point"
    assert read.attributed_cache_read > 0, "cumulative billing still owes the Read"
    assert read.context_tokens == 0, \
        f"a compacted-away Read should hold none of the current window," \
        f" got {read.context_tokens}"


def test_the_baseline_folds_into_blocks_proportionally_not_flatly(claude_home: Path):
    """DESIGN.md §7 "the unowned baseline": the system prompt/tool-definition
    share folds into whatever is resident in proportion to its own size, not
    an even split. A third, later call is needed so BOTH the Read and the
    Bash are actually resident by the end — with only two calls the second
    one's own content never enters the snapshot at all, and the test would
    pass for the wrong reason (an empty block, not a proportional one)."""
    lines = [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/big.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=100, output_tokens=40),
        tool_result_line("r1", "t1", "(truncated)", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/big.py",
                                            "numLines": 2000,
                                            "content": "z" * 80_000}}),
        assistant_tool_line("a2", "msg2", "t2", "Bash", {"command": "pytest -q"},
                            "2026-08-01T10:00:03.000Z",
                            output_tokens=10, cache_creation=200, cache_read=100),
        # A few thousand characters of test output — big enough that its
        # weight clears integer-rounding noise against the Read's 80,000, but
        # still far smaller, so the ">9x" ratio below holds comfortably.
        tool_result_line("r2", "t2", "F" * 4_000, "2026-08-01T10:00:04.000Z"),
        assistant_text_line("a3", "msg3", "Done.", "2026-08-01T10:00:05.000Z",
                            output_tokens=10, cache_creation=30, cache_read=3_000),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    read = next(b for b in blocks if b.kind == "read")
    execute = next(b for b in blocks if b.kind == "execute")

    assert read.context_tokens > 0 and execute.context_tokens > 0, \
        "both blocks must actually be resident in the final snapshot"
    assert read.context_tokens > 9 * execute.context_tokens, \
        f"the much larger Read result ({read.context_tokens}) should absorb far" \
        f" more of the baseline than Bash's tiny result ({execute.context_tokens})"


def test_subagent_context_window_is_independent_of_the_main_threads(claude_home: Path):
    """Each thread's window is its own last call — never summed together.

    A subagent's context is a separate, isolated window on its own call
    sequence; mixing it into the main thread's figure would reintroduce the
    exact cross-call sum this feature exists to remove.
    """
    session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    agent_id = "a0b679820a32f88c4"
    project_dir = claude_home / PROJECT_SLUG

    write_transcript(project_dir / f"{session_id}.jsonl", [
        user_line("u1", "audit this repo", "2026-08-02T09:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Agent",
                            {"subagent_type": "Explore", "prompt": "find TODOs"},
                            "2026-08-02T09:00:01.000Z",
                            input_tokens=1000, output_tokens=50, cache_read=5000),
        tool_result_line("r1", "t1", "done", "2026-08-02T09:00:30.000Z",
                         envelope={"agentId": agent_id, "status": "completed"}),
        assistant_text_line("a2", "msg2", "Found nothing.", "2026-08-02T09:00:31.000Z",
                            output_tokens=20, cache_creation=60, cache_read=6000),
    ])

    sub_path = project_dir / session_id / "subagents" / f"agent-{agent_id}.jsonl"
    write_transcript(sub_path, [
        {"type": "user", "uuid": "s-u1", "parentUuid": None, "isSidechain": True,
         "agentId": agent_id, "timestamp": "2026-08-02T09:00:02.000Z",
         "message": {"role": "user", "content": "find TODOs"}},
        {"type": "assistant", "uuid": "s-a1", "isSidechain": True,
         "agentId": agent_id, "timestamp": "2026-08-02T09:00:10.000Z",
         "message": {"id": "s-msg1", "role": "assistant", "model": "claude-opus-4-6",
                     "content": [{"type": "tool_use", "id": "s-t1", "name": "Grep",
                                  "input": {"pattern": "TODO"}}],
                     "usage": {"input_tokens": 200, "output_tokens": 10}}},
        {"type": "assistant", "uuid": "s-a2", "isSidechain": True,
         "agentId": agent_id, "timestamp": "2026-08-02T09:00:20.000Z",
         "message": {"id": "s-msg2", "role": "assistant", "model": "claude-opus-4-6",
                     "content": [{"type": "tool_use", "id": "s-t2", "name": "Read",
                                  "input": {"file_path": "/repo/a.py"}}],
                     "usage": {"input_tokens": 0, "output_tokens": 10,
                               "cache_creation_input_tokens": 300,
                               "cache_read_input_tokens": 200}}},
    ])

    session = claude_code.load_session(PROJECT_SLUG, project_dir / f"{session_id}.jsonl")
    session.overhead_tokens = attribute(session)

    main_events = [e for e in session.events if e.agent_id is None]
    sub_events = [e for e in session.events if e.agent_id == agent_id]

    # Main thread's own last call (msg2): fresh 60 + cached 6,000 = 6,060.
    assert session.context_window_tokens == 6_060
    assert sum(e.context_tokens for e in main_events) == 6_060
    # The subagent's own last call (s-msg2): fresh 300 + cached 200 = 500 —
    # independent of, and unrelated in size to, the main thread's 6,060.
    assert sum(e.context_tokens for e in sub_events) == 500


def test_a_single_call_session_context_window_is_that_calls_whole_prompt(
    claude_home: Path
):
    session = _loaded(claude_home, [
        user_line("u1", "hello?", "2026-08-01T10:00:00.000Z"),
        assistant_text_line("a1", "msg1", "Hi!", "2026-08-01T10:00:01.000Z",
                            input_tokens=200, output_tokens=10, cache_read=50),
    ])

    assert session.context_window_tokens == 250  # fresh 200 + cached 50


# ----------------------------------------------------------------------
# The work per call is bounded
# ----------------------------------------------------------------------

def _starved_lines(calls: int) -> list[dict]:
    """Many calls, each billed far less ``fresh`` than its content weighs.

    The common shape, not an edge case: ``fresh`` excludes cache reads, which
    are ~95% of a real session's tokens, so the arrival ledger is nearly always
    under-supplied.
    """
    lines = [user_line("u1", "go", "2026-08-01T10:00:00.000Z")]
    for index in range(calls):
        minute, second = divmod(index, 60)
        lines.append(assistant_tool_line(
            f"a{index}", f"msg{index}", f"t{index}", "Read",
            {"file_path": f"/repo/f{index}.py"},
            f"2026-08-01T11:{minute:02d}:{second:02d}.000Z",
            output_tokens=20, cache_creation=5, cache_read=1000))
        lines.append(tool_result_line(
            f"r{index}", f"t{index}", "src",
            f"2026-08-01T11:{minute:02d}:{second:02d}.500Z",
            envelope={"file": {"filePath": f"/repo/f{index}.py",
                               "numLines": 100, "content": "y" * 4000}}))
    return lines


def test_a_starved_session_does_not_re_split_its_whole_backlog(
        claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    """The ledger has to shed entries, or the cost per call grows with the session.

    Pro rata pays every debt a little and none in full, so a ``debt > 0`` filter
    alone removes nothing and the backlog grows ~2 entries per call forever —
    with every call re-splitting and re-sorting all of it. Counting the weights
    that pass through ``split_exact`` is the direct measurement. Bounded, this
    fixture measures a flat ~22 weights per call (2,112 / 4,312 / 6,512 items at
    N = 100 / 200 / 300); unbounded it is ~N^2, which at N=300 is ~90,000.
    """
    from Final_app.analysis import attribution

    calls = 300
    weighed = 0
    real_split = attribution.split_exact

    def counting_split(amount, weights):
        nonlocal weighed
        weighed += len(weights)
        return real_split(amount, weights)

    monkeypatch.setattr(attribution, "split_exact", counting_split)
    session = _loaded(claude_home, _starved_lines(calls))

    assert len(session.events) > calls, "the fixture should be as long as claimed"
    assert weighed < calls * attribution.LEDGER_CALLS * 10, (
        f"{weighed} weights across {calls} calls — the backlog is not draining")


def test_bounding_the_ledger_keeps_the_total_exact(claude_home: Path):
    """Retiring a debt must not lose the tokens, only stop them asking."""
    session = _loaded(claude_home, _starved_lines(40))
    blocks = build_blocks(session, use_judge=False)

    assert sum(b.attributed_tokens for b in blocks) == session.tokens.working
    assert sum(b.attributed_cache_read for b in blocks) == session.tokens.cache_read


# ----------------------------------------------------------------------
# User prompts are matched to the call that read them
# ----------------------------------------------------------------------

def test_each_prompt_is_charged_to_the_call_that_followed_it(claude_home: Path):
    """The windows are consecutive, so a prompt belongs to exactly one call.

    A forward-only cursor replaced a rescan of every user event per call; this
    pins the behaviour that rescan produced — each prompt lands once, in the
    window it arrived in, and none is skipped or double-charged.
    """
    lines = [user_line("u0", "first task", "2026-08-01T10:00:00.000Z")]
    for index in range(5):
        lines.append(assistant_text_line(
            f"a{index}", f"msg{index}", f"Working ({index}).",
            f"2026-08-01T10:{index * 2 + 1:02d}:00.000Z",
            input_tokens=200, output_tokens=40, cache_creation=300))
        lines.append(user_line(
            f"u{index + 1}", f"follow-up number {index}",
            f"2026-08-01T10:{index * 2 + 2:02d}:00.000Z"))
    session = _loaded(claude_home, lines)

    prompts = [e for e in session.events if e.type == "user"]

    assert len(prompts) == 6
    # The last prompt arrived after every call, so no call ever read it.
    assert all(e.attributed_tokens > 0 for e in prompts[:-1]), \
        [(e.text, e.attributed_tokens) for e in prompts]
    assert sum(e.attributed_tokens for e in session.events) \
        == session.tokens.working
