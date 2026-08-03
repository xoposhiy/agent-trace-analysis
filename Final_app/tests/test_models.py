"""IR tests: token arithmetic, block aggregation, session rollups.

The token tests exist because getting this wrong is silent and large — an
early build displayed 13.2M tokens for a session whose real figure was 726k.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from Final_app.ir.models import (
    EV_TOOL_USE,
    EV_USER,
    READ,
    WRITE,
    Block,
    Event,
    Problem,
    Session,
    Tokens,
)

BASE_TIME = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)


def _event(offset_seconds: int, event_type: str = EV_TOOL_USE, **kwargs) -> Event:
    return Event(
        uuid=f"u{offset_seconds}",
        ts=BASE_TIME + timedelta(seconds=offset_seconds),
        type=event_type,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------

def test_total_counts_every_token_including_cache_reads():
    tokens = Tokens(input=10, output=20, cache_read=1000, cache_creation=100)

    assert tokens.total == 1130


def test_working_excludes_cache_reads():
    """Cache reads are the same context re-read per message; summing inflates.

    Ratio here mirrors a real session: 95% of the raw total was cache reads.
    """
    tokens = Tokens(input=10, output=20, cache_read=1000, cache_creation=100)

    assert tokens.working == 130


def test_tokens_add_componentwise():
    total = (Tokens(input=1, output=2, cache_read=3, cache_creation=4)
             + Tokens(input=10, output=20, cache_read=30, cache_creation=40))

    assert (total.input, total.output, total.cache_read, total.cache_creation) \
        == (11, 22, 33, 44)


def test_tokens_serialise_both_figures():
    payload = Tokens(input=10, output=20, cache_read=1000, cache_creation=100).as_dict()

    assert payload["total"] == 1130
    assert payload["working"] == 130


# ----------------------------------------------------------------------
# Block
# ----------------------------------------------------------------------

def test_block_spans_its_events():
    block = Block(kind=READ, events=[_event(0), _event(30), _event(10)])

    assert block.t_start == BASE_TIME
    assert block.t_end == BASE_TIME + timedelta(seconds=30)
    assert block.duration_s == 30.0
    assert block.message_count == 3


def test_block_sums_the_tokens_of_its_events():
    block = Block(kind=WRITE, events=[
        _event(0, tokens=Tokens(input=5, output=10)),
        _event(1, tokens=Tokens(input=7, output=1)),
    ])

    assert block.tokens.input == 12
    assert block.tokens.output == 11


def test_empty_block_has_no_span_rather_than_crashing():
    block = Block(kind=READ)

    assert block.t_start is None
    assert block.duration_s == 0.0


def test_block_serialises_nested_subagent_blocks():
    inner = Block(kind=READ, events=[_event(1)])
    outer = Block(kind="subagent", events=[_event(0)], inner_blocks=[inner],
                  agent_id="abc", description="Find TODOs")
    payload = outer.as_dict()

    assert payload["agent_id"] == "abc"
    assert len(payload["inner_blocks"]) == 1
    assert payload["inner_blocks"][0]["kind"] == READ


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------

def _session(**kwargs) -> Session:
    defaults = dict(session_id="s1", project="-Users-tester-repo")
    defaults.update(kwargs)
    return Session(**defaults)


def test_session_counts_only_tool_use_events_as_tool_calls():
    session = _session(events=[
        _event(0, EV_USER),
        _event(1, EV_TOOL_USE),
        _event(2, EV_TOOL_USE),
    ])

    assert len(session.tool_calls) == 2
    assert len(session.events) == 3


def test_session_duration_spans_first_to_last():
    session = _session(
        first_ts=BASE_TIME,
        last_ts=BASE_TIME + timedelta(minutes=5),
    )

    assert session.duration_s == 300.0


def test_max_severity_is_none_when_there_are_no_problems():
    assert _session().max_severity == "none"


def test_max_severity_reports_the_worst_problem():
    session = _session(problems=[
        Problem(id="a", title="A", severity="low"),
        Problem(id="b", title="B", severity="high"),
        Problem(id="c", title="C", severity="medium"),
    ])

    assert session.max_severity == "high"


def test_summary_dict_omits_events_and_blocks():
    """The list page loads every session; full event arrays would be megabytes."""
    session = _session(events=[_event(0) for _ in range(50)])
    payload = session.summary_dict()

    assert "events" not in payload
    assert "blocks" not in payload
    assert payload["message_count"] == 50
