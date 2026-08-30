"""Token accounting: every billed token reaches exactly one Event, once.

These pin the fix made on 2026-08-05. Before it, the adapter attached a
message's whole ``usage`` to that message's *prose* Event and gave tool calls
``Tokens()``. Two things followed, both measured on a real 259-message
transcript (``51db4d3e``):

  * every tool call reported 0 tokens, so the bar's "tokens" Y-axis could not
    size a read/write/execute block unless prose happened to merge into it;
  * a message with no prose had nowhere to put its usage and it was silently
    dropped — 476,169 of 853,096 working tokens (55.8%).

The invariant asserted throughout: the sum over Events (plus
``Session.orphaned_tokens``) equals the sum over the transcript's ``usage``
objects, exactly. Not approximately — the largest-remainder split in
``_split_by_weight`` exists so that "exactly" holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Final_app.adapters import claude_code
from Final_app.adapters.claude_code import _split_by_weight
from Final_app.analysis.blocks import build_blocks
from Final_app.ir.models import EV_TOOL_USE

from .conftest import (
    PROJECT_SLUG,
    assistant_mixed_line,
    assistant_text_line,
    assistant_thinking_line,
    assistant_tool_line,
    tool_result_line,
    user_line,
    write_transcript,
)


# ----------------------------------------------------------------------
# The exact split
# ----------------------------------------------------------------------

@pytest.mark.parametrize("amount,weights", [
    (100, [1, 1, 1]),          # does not divide evenly
    (7, [5, 3, 1]),            # small amount, uneven weights
    (1, [1, 1, 1, 1]),         # fewer tokens than recipients
    (999_983, [17, 4, 129]),   # a prime, so every share has a remainder
])
def test_splitting_output_invents_and_loses_nothing(amount: int, weights: list[int]):
    shares = _split_by_weight(amount, weights)

    assert sum(shares) == amount
    assert len(shares) == len(weights)
    assert all(share >= 0 for share in shares)


def test_a_heavier_content_block_gets_at_least_as_many_tokens():
    assert _split_by_weight(100, [90, 10]) == [90, 10]


def test_splitting_with_no_weight_information_is_still_exact():
    """All-zero weights happen when every content block is empty."""
    assert sum(_split_by_weight(10, [0, 0, 0])) == 10


def test_splitting_across_nobody_yields_nothing():
    assert _split_by_weight(500, []) == []


# ----------------------------------------------------------------------
# Attribution across a message's Events
# ----------------------------------------------------------------------

def test_a_tool_only_message_keeps_its_tokens(claude_home: Path):
    """The 55.8% leak. Before the fix this session reported 0 working tokens."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "read the file", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=120, output_tokens=340,
                            cache_read=9000, cache_creation=55),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.tokens.input == 120
    assert session.tokens.output == 340
    assert session.tokens.cache_creation == 55
    assert session.tokens.cache_read == 9000
    assert session.tokens.working == 120 + 340 + 55


def test_a_single_tool_call_reports_its_own_tokens(claude_home: Path):
    """What the bar shows on a one-step read/write/execute block."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "read the file", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=120, output_tokens=340),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    call = next(e for e in session.events if e.type == EV_TOOL_USE)

    assert call.tokens.output == 340
    assert call.tokens.working == 460


def test_output_is_shared_between_prose_and_the_calls_it_precedes(claude_home: Path):
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_mixed_line(
            "a1", "msg1", "Let me look at both files.",
            [("t1", "Read", {"file_path": "/repo/a.py"}),
             ("t2", "Read", {"file_path": "/repo/b.py"})],
            "2026-08-01T10:00:01.000Z",
            input_tokens=200, output_tokens=900, cache_creation=30),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    from_message = [e for e in session.events if e.message_id == "msg1"]

    assert len(from_message) == 3
    assert sum(e.tokens.output for e in from_message) == 900
    assert all(e.tokens.output > 0 for e in from_message), \
        "no content block of a billed message should report zero output"


def test_prompt_side_tokens_are_charged_once_not_per_content_block(claude_home: Path):
    """Splitting ``input`` would let a block claim context it did not cause."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_mixed_line(
            "a1", "msg1", "Looking.",
            [("t1", "Read", {"file_path": "/repo/a.py"}),
             ("t2", "Read", {"file_path": "/repo/b.py"})],
            "2026-08-01T10:00:01.000Z",
            input_tokens=200, output_tokens=90,
            cache_read=40_000, cache_creation=30),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    from_message = [e for e in session.events if e.message_id == "msg1"]
    charged = [e for e in from_message if e.tokens.input]

    assert len(charged) == 1
    assert charged[0].tokens.input == 200
    assert charged[0].tokens.cache_creation == 30
    assert charged[0].tokens.cache_read == 40_000
    assert session.tokens.input == 200


def test_a_thinking_only_message_keeps_its_tokens_as_orphaned(claude_home: Path):
    """Thinking never reaches the IR, so its usage has no Event to land on.

    Rare but real: 1 message of 106 in session 5f12bde3. It is held on the
    Session so the total still matches the transcript.
    """
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_thinking_line("a1", "msg1", "2026-08-01T10:00:01.000Z"),
        assistant_text_line("a2", "msg2", "Done.", "2026-08-01T10:00:02.000Z",
                            output_tokens=10),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    # assistant_thinking_line bills 1 in / 1 out.
    assert session.orphaned_tokens.working == 2
    assert session.tokens.working == 12


def test_streamed_fragments_are_billed_once_for_the_whole_message(claude_home: Path):
    """Streaming usage is cumulative; counting each fragment would multiply it."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
        assistant_text_line("a1", "msg1", "Hello ", "2026-08-01T10:00:01.000Z",
                            input_tokens=10, output_tokens=5),
        assistant_text_line("a2", "msg1", "there", "2026-08-01T10:00:02.000Z",
                            input_tokens=10, output_tokens=9),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert session.tokens.output == 9
    assert session.tokens.input == 10


# ----------------------------------------------------------------------
# The end-to-end invariant
# ----------------------------------------------------------------------

def _billed_totals(lines: list[dict]) -> dict[str, int]:
    """Ground truth: the transcript's own ``usage``, one entry per message id."""
    by_message: dict[str, dict] = {}
    for line in lines:
        if line.get("type") != "assistant":
            continue
        message = line["message"]
        usage = message.get("usage") or {}
        if any(usage.values()):
            by_message[message["id"]] = usage
    return {
        "input": sum(u.get("input_tokens", 0) for u in by_message.values()),
        "output": sum(u.get("output_tokens", 0) for u in by_message.values()),
        "cache_read": sum(u.get("cache_read_input_tokens", 0)
                          for u in by_message.values()),
        "cache_creation": sum(u.get("cache_creation_input_tokens", 0)
                              for u in by_message.values()),
    }


def test_a_mixed_session_totals_exactly_what_the_transcript_billed(claude_home: Path):
    lines = [
        user_line("u1", "fix it", "2026-08-01T10:00:00.000Z"),
        assistant_thinking_line("a0", "msg0", "2026-08-01T10:00:01.000Z"),
        assistant_mixed_line(
            "a1", "msg1", "Reading first.",
            [("t1", "Read", {"file_path": "/repo/a.py"})],
            "2026-08-01T10:00:02.000Z",
            input_tokens=13, output_tokens=777, cache_read=4321, cache_creation=99),
        tool_result_line("r1", "t1", "contents", "2026-08-01T10:00:03.000Z"),
        assistant_tool_line("a2", "msg2", "t2", "Edit",
                            {"file_path": "/repo/a.py", "old_string": "x"},
                            "2026-08-01T10:00:04.000Z",
                            input_tokens=7, output_tokens=1013, cache_read=5000),
        tool_result_line("r2", "t2", "ok", "2026-08-01T10:00:05.000Z"),
        assistant_text_line("a3", "msg3", "Fixed.", "2026-08-01T10:00:06.000Z",
                            input_tokens=3, output_tokens=41),
    ]
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)
    session = claude_code.load_session(PROJECT_SLUG, path)
    expected = _billed_totals(lines)

    assert session.tokens.input == expected["input"]
    assert session.tokens.output == expected["output"]
    assert session.tokens.cache_read == expected["cache_read"]
    assert session.tokens.cache_creation == expected["cache_creation"]


def test_blocks_account_for_every_token_the_session_has(claude_home: Path):
    """The bar's heights must add up to the header's total, or one of them lies.

    ``user_chat`` blocks are human input and genuinely cost nothing, so the
    identity is over the assistant side: block tokens + orphaned == session.
    """
    lines = [
        user_line("u1", "fix it", "2026-08-01T10:00:00.000Z"),
        assistant_thinking_line("a0", "msg0", "2026-08-01T10:00:01.000Z"),
        assistant_mixed_line(
            "a1", "msg1", "Reading.",
            [("t1", "Read", {"file_path": "/repo/a.py"}),
             ("t2", "Read", {"file_path": "/repo/b.py"})],
            "2026-08-01T10:00:02.000Z",
            input_tokens=13, output_tokens=777, cache_creation=99),
        tool_result_line("r1", "t1", "contents", "2026-08-01T10:00:03.000Z"),
        assistant_tool_line("a2", "msg2", "t3", "Bash", {"command": "pytest"},
                            "2026-08-01T10:00:04.000Z",
                            input_tokens=7, output_tokens=1013),
    ]
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)
    session = claude_code.load_session(PROJECT_SLUG, path)
    blocks = build_blocks(session, use_judge=False)

    in_blocks = sum(b.tokens.working for b in blocks)

    assert in_blocks + session.orphaned_tokens.working == session.tokens.working


def test_no_work_block_reports_zero_tokens_when_the_message_was_billed(
    claude_home: Path,
):
    """The reported symptom: single-step read/write/execute blocks showing 0."""
    lines = [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z", output_tokens=50),
        tool_result_line("r1", "t1", "x", "2026-08-01T10:00:02.000Z"),
        assistant_tool_line("a2", "msg2", "t2", "Write",
                            {"file_path": "/repo/b.py", "content": "y"},
                            "2026-08-01T10:00:03.000Z", output_tokens=60),
        tool_result_line("r2", "t2", "x", "2026-08-01T10:00:04.000Z"),
        assistant_tool_line("a3", "msg3", "t3", "Bash", {"command": "pytest"},
                            "2026-08-01T10:00:05.000Z", output_tokens=70),
    ]
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)
    session = claude_code.load_session(PROJECT_SLUG, path)
    blocks = build_blocks(session, use_judge=False)

    work = [b for b in blocks if b.kind in ("read", "write", "execute")]

    assert len(work) == 3
    assert all(b.tokens.working > 0 for b in work), \
        [(b.kind, b.tokens.working) for b in work]
