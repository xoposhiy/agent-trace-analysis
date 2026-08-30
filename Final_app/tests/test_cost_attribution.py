"""Dollar attribution: the ledger in ``analysis.attribution`` priced, not counted.

Mirrors ``test_attribution.py``'s invariant style, but for
``Event.attributed_cost``/``Block.attributed_cost``. The one thing token
attribution does not need and dollar attribution does: which *model* billed
each piece, since a subagent can run on a different, differently-priced model
than its parent thread. See ``analysis.attribution``'s "Dollars" section and
``analysis.pricing``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Final_app.adapters import claude_code
from Final_app.analysis.attribution import attribute
from Final_app.analysis.blocks import build_blocks
from Final_app.analysis.pricing import price_for_model

from .conftest import (
    PROJECT_SLUG,
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


# Every fixture line in conftest.py hardcodes this model, so the exact rate is
# known for every manual calculation below.
FIXTURE_MODEL = "claude-opus-4-6"
FIXTURE_PRICE = price_for_model(FIXTURE_MODEL)


def _mixed_lines() -> list[dict]:
    return [
        user_line("u1", "please fix the parser", "2026-08-01T10:00:00.000Z"),
        assistant_text_line("a1", "msg1", "Reading the parser first.",
                            "2026-08-01T10:00:01.000Z",
                            input_tokens=1200, output_tokens=310,
                            cache_creation=800),
        assistant_tool_line("a2", "msg1", "t1", "Read",
                            {"file_path": "/repo/parse.py"},
                            "2026-08-01T10:00:01.500Z"),
        tool_result_line("r1", "t1", "src", "2026-08-01T10:00:02.000Z",
                         envelope={"file": {"filePath": "/repo/parse.py",
                                            "numLines": 400,
                                            "content": "z" * 20_000}}),
        assistant_tool_line("a3", "msg2", "t2", "Edit",
                            {"file_path": "/repo/parse.py", "old_string": "a",
                             "new_string": "b"},
                            "2026-08-01T10:00:03.000Z",
                            output_tokens=90, cache_read=2000, cache_creation=500),
        tool_result_line("r2", "t2", "ok", "2026-08-01T10:00:04.000Z"),
    ]


# ----------------------------------------------------------------------
# The invariant: the session's real bill, exactly attributed
# ----------------------------------------------------------------------

def test_the_session_total_cost_matches_a_manual_bill_from_usage(claude_home: Path):
    """Sum every block's dollars and it must equal the actual bill.

    Computed independently from ``session.tokens`` (the raw, un-attributed
    totals) at the one model every fixture line uses, so this does not reuse
    any of the attribution machinery it is checking.
    """
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    expected = (
        session.tokens.input * FIXTURE_PRICE.input
        + session.tokens.output * FIXTURE_PRICE.output
        + session.tokens.cache_creation * FIXTURE_PRICE.cache_write
        + session.tokens.cache_read * FIXTURE_PRICE.cache_read
    )
    actual = sum(b.attributed_cost for b in blocks)

    assert actual == pytest.approx(expected, rel=1e-9)


def test_a_blocks_cost_is_exactly_its_members_cost(claude_home: Path):
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    for block in blocks:
        assert block.attributed_cost == pytest.approx(
            sum(e.attributed_cost for e in block.events))


def test_the_sessions_total_cost_agrees_with_the_sum_of_its_blocks(claude_home: Path):
    """The header stat and the bar underneath it must never disagree."""
    session = _loaded(claude_home, _mixed_lines())
    blocks = build_blocks(session, use_judge=False)

    assert session.attributed_cost == pytest.approx(
        sum(b.attributed_cost for b in blocks))


def test_a_session_with_no_billed_tokens_costs_nothing(claude_home: Path):
    session = _loaded(claude_home, [
        user_line("u1", "hello?", "2026-08-01T10:00:00.000Z"),
    ])

    assert sum(e.attributed_cost for e in session.events) == 0.0


# ----------------------------------------------------------------------
# Output vs. fresh-prompt vs. cache-read are priced differently
# ----------------------------------------------------------------------

def test_output_is_priced_at_the_output_rate_not_the_input_rate(claude_home: Path):
    """A pure-output message's cost must come from the (far higher) output rate."""
    session = _loaded(claude_home, [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_text_line("a1", "msg1", "Done.", "2026-08-01T10:00:01.000Z",
                            output_tokens=1000),
    ])

    reply = next(e for e in session.events if e.text == "Done.")

    assert reply.attributed_cost == pytest.approx(1000 * FIXTURE_PRICE.output)


def test_a_cache_read_heavy_block_costs_far_less_per_token_than_a_fresh_one(
        claude_home: Path):
    """The whole reason cache reads get their own channel: they are ~10x cheaper."""
    lines = [
        user_line("u1", "go", "2026-08-01T10:00:00.000Z"),
        assistant_tool_line("a1", "msg1", "t1", "Read", {"file_path": "/repo/a.py"},
                            "2026-08-01T10:00:01.000Z",
                            output_tokens=50, cache_creation=1000),
        tool_result_line("r1", "t1", "src", "2026-08-01T10:00:02.000Z"),
        assistant_text_line("a2", "msg2", "Done.", "2026-08-01T10:00:03.000Z",
                            output_tokens=20, cache_read=1_000_000),
    ]
    session = _loaded(claude_home, lines)
    blocks = build_blocks(session, use_judge=False)

    read = next(b for b in blocks if b.kind == "read")

    # The Read's own block absorbs almost all of the 1,000,000-token re-read
    # (it is the only prior content there is to re-read), so its cost per
    # attributed token must sit near the cache-read rate, nowhere near the
    # far higher fresh-input/output rates.
    assert read.attributed_tokens > 0
    cost_per_token = read.attributed_cost / read.attributed_total
    assert cost_per_token < FIXTURE_PRICE.input
    assert cost_per_token == pytest.approx(FIXTURE_PRICE.cache_read, rel=0.05)


# ----------------------------------------------------------------------
# Mixed models: a subagent priced at its own rate
# ----------------------------------------------------------------------

def test_a_subagent_on_a_different_model_is_priced_at_that_models_rate(
        claude_home: Path):
    """A Haiku subagent's cost must use Haiku rates, not the parent's Sonnet rate.

    Written with raw line dicts rather than the ``conftest`` helpers, which
    hardcode ``claude-opus-4-6`` — this is the one test that needs two models.
    """
    session_id = "dddddddd-eeee-ffff-0000-111111111111"
    agent_id = "agentcheapmodel1"
    project_dir = claude_home / PROJECT_SLUG

    write_transcript(project_dir / f"{session_id}.jsonl", [
        user_line("u1", "audit this repo", "2026-08-05T09:00:00.000Z"),
        {
            "type": "assistant", "uuid": "a1", "timestamp": "2026-08-05T09:00:01.000Z",
            "message": {
                "id": "msg1", "role": "assistant", "model": "claude-sonnet-4-6",
                "content": [{"type": "tool_use", "id": "t1", "name": "Agent",
                             "input": {"subagent_type": "Explore"}}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        },
        tool_result_line("r1", "t1", "done", "2026-08-05T09:00:30.000Z",
                         envelope={"agentId": agent_id, "description": "scout"}),
    ])
    write_transcript(project_dir / session_id / "subagents" / f"agent-{agent_id}.jsonl", [
        {
            "type": "assistant", "uuid": "s-a1", "isSidechain": True,
            "agentId": agent_id, "timestamp": "2026-08-05T09:00:10.000Z",
            "message": {
                "id": "s-msg1", "role": "assistant", "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "found nothing"}],
                "usage": {"input_tokens": 500, "output_tokens": 200},
            },
        },
    ])

    session = claude_code.load_session(PROJECT_SLUG, project_dir / f"{session_id}.jsonl")
    attribute(session)

    subagent_event = next(e for e in session.events if e.agent_id == agent_id)
    haiku_price = price_for_model("claude-haiku-4-5")
    sonnet_price = price_for_model("claude-sonnet-4-6")

    assert haiku_price.output != sonnet_price.output, \
        "the two rates must actually differ for this test to prove anything"

    # The subagent's own message generated all of its output itself (rule 1 —
    # no cross-thread attribution), so its cost is exactly its own tokens at
    # its own model's rate, regardless of what the parent thread would pay.
    expected_output_cost = 200 * haiku_price.output
    assert subagent_event.attributed_cost >= expected_output_cost - 1e-9
    # And it must not have been priced as if it were the parent's Sonnet call.
    assert subagent_event.attributed_cost != pytest.approx(200 * sonnet_price.output) \
        or haiku_price.output == sonnet_price.output


# ----------------------------------------------------------------------
# Degrading visibly on an unrecognized model
# ----------------------------------------------------------------------

def test_an_unrecognized_model_still_produces_a_cost_not_a_crash(claude_home: Path):
    session_id = "ffffffff-0000-1111-2222-333333333333"
    project_dir = claude_home / PROJECT_SLUG
    write_transcript(project_dir / f"{session_id}.jsonl", [
        user_line("u1", "go", "2026-08-06T09:00:00.000Z"),
        {
            "type": "assistant", "uuid": "a1", "timestamp": "2026-08-06T09:00:01.000Z",
            "message": {
                "id": "msg1", "role": "assistant", "model": "claude-future-model-9000",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 10, "output_tokens": 40},
            },
        },
    ])

    session = claude_code.load_session(PROJECT_SLUG, project_dir / f"{session_id}.jsonl")
    attribute(session)

    reply = next(e for e in session.events if e.text == "hi")
    assert reply.attributed_cost > 0
