"""Tests for the missed plan-mode opportunity detector (CLAUDE.md §6).

Three layers, matching the module's own split:
  * ``price_split`` — the adapter from ``Session``/``Event`` into the
    unmodified ``chunk_split_model`` math, checked against hand-computed
    dollar figures (not ``> 0``).
  * ``find_front_loaded_reading`` — the pattern match over ``Block``s, built
    directly like ``test_blocks.py``'s fixtures rather than through a real
    transcript.
  * ``detect`` — end to end, through a real transcript, ``load_session``,
    ``attribute`` and ``build_blocks(use_judge=False)``, so no VPN is needed.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from Final_app.adapters import claude_code
from Final_app.analysis import classify, plan_mode
from Final_app.analysis.attribution import attribute
from Final_app.analysis.blocks import build_blocks
from Final_app.analysis.plan_mode import (
    build_justification_prompt,
    detect,
    find_front_loaded_reading,
    justify,
    price_split,
)
from Final_app.analysis.problems import detect_problems
from Final_app.ir.models import (
    EV_TOOL_USE,
    EXECUTE,
    READ,
    WRITE,
    Block,
    Event,
    Session,
    Tokens,
    ToolCall,
)

from .conftest import (
    PROJECT_SLUG,
    assistant_tool_line,
    permission_mode_line,
    tool_result_line,
    user_line,
    write_transcript,
)

BASE_TIME = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_justify_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Redirect the on-disk justification cache so tests never touch or
    depend on the real ``~/.cache/tracelens`` (CLAUDE.md §6), and never
    inherit another test's cached answer for the same content hash."""
    monkeypatch.setattr(plan_mode, "_JUSTIFY_CACHE_FILE", tmp_path / "plan_mode_justifications.json")
    monkeypatch.setattr(plan_mode, "_justify_cache", None)


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, on_call):
    """Stand a fake ``openai`` module up so no request ever leaves the process.

    Mirrors ``test_config.py``'s own helper of the same shape. ``on_call``
    runs in place of ``chat.completions.create``; raise to simulate a
    failure, return a fake response object to simulate success.
    """
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return on_call(**kwargs)

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = _Chat()

    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


def _fake_response(text: str):
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def _call(index: int, cache_read: int = 0, cost: float = 0.0) -> Event:
    """``cost`` stands in for what ``analysis.attribution`` would have set on
    a real Event; these tests build ``Session``/``Event`` directly rather
    than through a real transcript, so nothing else ever populates it."""
    return Event(
        uuid=f"e{index}", ts=BASE_TIME + timedelta(seconds=index),
        type=EV_TOOL_USE, message_id=f"m{index}",
        tokens=Tokens(cache_read=cache_read),
        attributed_cost=cost,
    )


# ----------------------------------------------------------------------
# _rates_for_session — real per-model rates, in chunk_split_model's shape
# ----------------------------------------------------------------------
# The bug this exists to fix: chunk_split_model's own DEFAULT_RATES is one
# flat, model-agnostic guess (4.75/23.75, a blended Opus-4.x estimate), so
# every session's `as_is_cost` disagreed with `Session.attributed_cost`
# ("Retrospective cost") by the ratio between that guess and whichever model
# actually ran the session.

@pytest.mark.parametrize("model,expected_input,expected_output", [
    ("claude-opus-4-6", 5.00, 25.00),
    ("claude-haiku-4-5", 1.00, 5.00),
])
def test_rates_for_session_matches_the_real_per_model_price(
    model: str, expected_input: float, expected_output: float
):
    from Final_app.analysis.plan_mode import _rates_for_session
    from Final_app.analysis.pricing import price_for_model

    session = Session(session_id="s", project="p", model=model)
    rates = _rates_for_session(session)
    price = price_for_model(model)

    assert rates.input_per_mtok == pytest.approx(expected_input)
    assert rates.output_per_mtok == pytest.approx(expected_output)
    # Reconstructing $/token from the Rates shape must reproduce exactly what
    # `pricing.price_for_model` says this model bills — the whole point.
    assert rates.input_per_mtok / 1e6 == pytest.approx(price.input)
    assert rates.output_per_mtok / 1e6 == pytest.approx(price.output)
    assert rates.cache_write_mult * rates.input_per_mtok / 1e6 == pytest.approx(
        price.cache_write)
    assert rates.cache_read_mult * rates.input_per_mtok / 1e6 == pytest.approx(
        price.cache_read)


def test_rates_for_session_falls_back_to_the_default_price_for_an_unknown_model():
    from Final_app.analysis.plan_mode import _rates_for_session
    from Final_app.analysis.pricing import DEFAULT_PRICE

    session = Session(session_id="s", project="p", model="")
    rates = _rates_for_session(session)

    assert rates.input_per_mtok == pytest.approx(DEFAULT_PRICE.input * 1_000_000)
    assert rates.output_per_mtok == pytest.approx(DEFAULT_PRICE.output * 1_000_000)


# ----------------------------------------------------------------------
# price_split — the adapter into chunk_split_model
# ----------------------------------------------------------------------

def test_price_split_returns_none_with_no_cache_read():
    session = Session(session_id="s", project="p",
                      events=[_call(i) for i in range(5)])

    assert price_split(session, 0.5) is None


def test_price_split_returns_none_with_too_few_calls():
    session = Session(session_id="s", project="p", events=[_call(0, 1000)])

    assert price_split(session, 0.5) is None


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.95, 1.0])
def test_price_split_returns_none_for_a_degenerate_split_point(fraction):
    session = Session(session_id="s", project="p",
                      events=[_call(i, 1000) for i in range(10)])

    assert price_split(session, fraction) is None


def test_price_split_as_is_cost_is_the_sessions_real_attributed_cost():
    """``as_is_cost`` must be ``session.attributed_cost`` itself, not a
    second, independently-computed estimate — the whole point of the fix
    this pins: the same number the session header calls "Retrospective
    cost" everywhere in the app, never a different figure that can disagree
    with it. Each of the 20 calls carries $0.50 of real attributed cost
    (set directly, since this fixture builds ``Event``s by hand rather than
    running them through ``analysis.attribution``), summing to $10.00.
    """
    session = Session(
        session_id="s", project="p", model="claude-opus-4-6",
        events=[_call(i, 1_000_000, cost=0.5) for i in range(20)],
    )

    saving = price_split(session, 0.25)

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(session.attributed_cost)
    assert saving["as_is_cost"] == pytest.approx(10.0)


def test_price_split_matches_a_hand_computed_saving():
    """20 calls, 20M cache-read tokens total, split at 0.25, on a session
    whose model is ``claude-opus-4-6`` ($5.00/$25.00 per MTok). ``as_is_cost``
    is the fixture's own $10.00 of real attributed cost (see the test above);
    ``split_cost`` is still the ``chunk_split_model`` ESTIMATE, since there is
    no exact figure for a split that never happened.

    Hand-computed against ``chunk_split_model``'s own formulas at this
    session's real per-model rate (``plan_mode._rates_for_session``), not
    ``chunk_split_model``'s flat model-agnostic default of 4.75/23.75. The
    cache read/write multipliers (1.25x / 0.10x input) are unchanged, since
    ``pricing.py`` uses the same ones by default: peak = 2*20e6/20 = 2e6;
    heights = [0.5e6, 1.5e6]; calls per chunk = [5, 15]; cache-read after =
    (20/(2*2e6)) * (0.5e6^2 + 1.5e6^2) = 12.5e6; summary tokens =
    0.1 * 0.5e6 = 5e4; overhead over 15 calls = (25*5e4 + 1.25*5*5e4 +
    0.10*5*5e4*14) / 1e6 = 0.19125; split = (12.5e6 * 5 * 0.10) / 1e6
    + 0.19125 = 8.1625; saving = 10.0 - 8.1625 = 1.8375 (18.375%).
    """
    session = Session(
        session_id="s", project="p", model="claude-opus-4-6",
        events=[_call(i, 1_000_000, cost=0.5) for i in range(20)],
    )

    saving = price_split(session, 0.25)

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(10.0)
    assert saving["cache_read_after"] == pytest.approx(12_500_000)
    assert saving["split_cost"] == pytest.approx(8.1625)
    assert saving["dollar_saving"] == pytest.approx(1.8375)
    assert saving["percent_saving"] == pytest.approx(18.375)
    assert saving["token_saving"] == pytest.approx(7_500_000)


def test_price_split_falls_back_to_the_default_rate_with_no_model_recorded(
):
    """A session with no ``model`` field must still price ``split_cost``,
    not crash — the same "estimate flagged, never silently $0 or a raised
    error" rule ``pricing.price_for_model`` already applies elsewhere.
    ``as_is_cost`` is unaffected by the model at all now, since it comes
    straight from ``session.attributed_cost``.
    """
    session = Session(
        session_id="s", project="p",
        events=[_call(i, 1_000_000, cost=1.0) for i in range(20)],
    )

    saving = price_split(session, 0.25)

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(20.0)
    assert saving["split_cost"] > 0


def test_price_split_adds_subagent_cost_to_both_sides_unchanged():
    """A subagent's own dollars land in ``as_is_cost`` "for free" now — it is
    ``session.attributed_cost``, which already sums every event, subagent
    included — and are added to ``split_cost`` on top of the ramp estimate
    (``_subagent_cost``), since splitting the main thread doesn't touch what
    a subagent cost. Both sides grow by exactly the same amount either way,
    so ``dollar_saving`` (their difference) is exactly unchanged — only
    ``percent_saving`` moves, since it is now a share of the real total bill.
    """
    main_events = [_call(i, 1_000_000, cost=0.5) for i in range(20)]
    subagent_event = Event(
        uuid="sub-e0", ts=BASE_TIME, type=EV_TOOL_USE, message_id="sub-m0",
        agent_id="sub1", attributed_cost=3.0,
    )
    without_subagent = Session(session_id="s", project="p", model="claude-opus-4-6",
                               events=main_events)
    with_subagent = Session(session_id="s", project="p", model="claude-opus-4-6",
                            events=main_events + [subagent_event])

    baseline = price_split(without_subagent, 0.25)
    plus_subagent = price_split(with_subagent, 0.25)

    assert plus_subagent["as_is_cost"] == pytest.approx(baseline["as_is_cost"] + 3.0)
    assert plus_subagent["split_cost"] == pytest.approx(baseline["split_cost"] + 3.0)
    assert plus_subagent["dollar_saving"] == pytest.approx(baseline["dollar_saving"])
    assert plus_subagent["percent_saving"] < baseline["percent_saving"]


# ----------------------------------------------------------------------
# find_front_loaded_reading — the Block-level pattern match
# ----------------------------------------------------------------------

def _events(n: int, offset: int = 0) -> list[Event]:
    return [Event(uuid=f"b{offset}-{i}",
                  ts=BASE_TIME + timedelta(seconds=offset + i),
                  type=EV_TOOL_USE)
            for i in range(n)]


def test_finds_a_qualifying_front_loaded_read_block():
    read_block = Block(kind=READ, events=_events(5, 0))
    write_block = Block(kind=WRITE, events=_events(5, 5))
    session = Session(session_id="s", project="p",
                      blocks=[read_block, write_block])

    assert find_front_loaded_reading(session) is read_block


def test_a_session_that_is_all_reading_is_not_a_missed_opportunity():
    session = Session(session_id="s", project="p",
                      blocks=[Block(kind=READ, events=_events(5, 0))])

    assert find_front_loaded_reading(session) is None


def test_a_read_run_that_starts_too_late_is_not_front_loaded():
    session = Session(session_id="s", project="p", blocks=[
        Block(kind=EXECUTE, events=_events(20, 0)),
        Block(kind=READ, events=_events(5, 20)),
        Block(kind=WRITE, events=_events(5, 25)),
    ])

    assert find_front_loaded_reading(session) is None


def test_a_read_run_shorter_than_the_minimum_is_skipped():
    session = Session(session_id="s", project="p", blocks=[
        Block(kind=READ, events=_events(2, 0)),
        Block(kind=WRITE, events=_events(5, 2)),
    ])

    assert find_front_loaded_reading(session) is None


def test_an_empty_block_list_returns_none():
    assert find_front_loaded_reading(Session(session_id="s", project="p")) is None


# ----------------------------------------------------------------------
# detect — end to end, through a real transcript
# ----------------------------------------------------------------------

def _front_loaded_transcript(claude_home: Path, used_plan_mode: bool = False) -> Path:
    """A session that reads six files (with real cache-read usage) before ten
    edits — the shape Pattern A is built to catch. Ten edits, not one: the
    ramp model's carried-summary overhead is only worth paying when enough of
    the session remains afterward to amortise it over.

    Every call also carries a small, realistic ``input``/``output``
    (and, for edits, ``cache_creation``) figure — a real transcript never
    bills zero on every channel but ``cache_read``, and ``price_split`` now
    needs ``session.attributed_cost`` to be a real, complete number (not the
    near-total loss ``analysis.attribution`` suffers when every call's
    ``fresh`` channel is zero — see ``attribute()``'s own "single case that
    leaves either unplaced" note).
    """
    lines = [user_line("u1", "Fix the bug", "2026-08-01T10:00:00.000Z")]
    if used_plan_mode:
        lines.append(permission_mode_line("plan", "2026-08-01T10:00:00.500Z"))
    for i in range(6):
        when = f"2026-08-01T10:00:{i + 1:02d}.000Z"
        lines.append(assistant_tool_line(
            f"a{i}", f"msg{i}", f"t{i}", "Read", {"file_path": f"/repo/f{i}.py"},
            when, input_tokens=50, output_tokens=40, cache_read=2_000_000,
        ))
        lines.append(tool_result_line(
            f"r{i}", f"t{i}", "file contents", when,
            envelope={"file": {"filePath": f"/repo/f{i}.py", "numLines": 10}},
        ))
    for i in range(10):
        when = f"2026-08-01T10:00:{i + 7:02d}.000Z"
        lines.append(assistant_tool_line(
            f"ae{i}", f"msg-edit{i}", f"te{i}", "Edit", {"file_path": "/repo/f0.py"},
            when, input_tokens=30, output_tokens=25, cache_creation=40,
        ))
        lines.append(tool_result_line(
            f"re{i}", f"te{i}", "ok", when,
            envelope={"filePath": "/repo/f0.py", "oldString": "a", "newString": "b"},
        ))
    return write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)


def test_detect_flags_a_front_loaded_reading_session(claude_home: Path):
    path = _front_loaded_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    session.blocks = build_blocks(session, use_judge=False)

    problem = detect(session)

    assert problem is not None
    assert problem.id == "plan-mode"
    assert problem.severity == "medium"
    read_block = next(b for b in session.blocks if b.kind == READ)
    assert problem.data["split_after_event_uuid"] == read_block.events[-1].uuid
    assert problem.data["dollar_saving"] > 0


def test_detect_returns_none_when_plan_mode_was_already_used(claude_home: Path):
    path = _front_loaded_transcript(claude_home, used_plan_mode=True)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    session.blocks = build_blocks(session, use_judge=False)

    assert session.used_plan_mode is True
    assert detect(session) is None


def test_detect_returns_none_for_an_ordinary_session(claude_home: Path, simple_session: Path):
    """Only three tool calls, none of them a qualifying front-loaded read run."""
    session = claude_code.load_session(PROJECT_SLUG, simple_session)
    attribute(session)
    session.blocks = build_blocks(session, use_judge=False)

    assert detect(session) is None
    assert detect_problems(session) == []


# ----------------------------------------------------------------------
# justify — the LLM-optional one-sentence justification
# ----------------------------------------------------------------------

def _read_block_with_targets(*file_paths: str) -> Block:
    events = [
        Event(uuid=f"r{i}", ts=BASE_TIME + timedelta(seconds=i), type=EV_TOOL_USE,
              tool=ToolCall(id=f"t{i}", name="Read", input={"file_path": path}))
        for i, path in enumerate(file_paths)
    ]
    return Block(kind=READ, events=events)


def test_read_block_targets_lists_file_paths_in_order_deduped():
    block = _read_block_with_targets("/repo/a.py", "/repo/b.py", "/repo/a.py")

    assert plan_mode._read_block_targets(block) == ["/repo/a.py", "/repo/b.py"]


def test_read_block_targets_is_empty_for_a_block_with_no_tool_calls():
    block = Block(kind=READ, events=[Event(uuid="e", ts=BASE_TIME, type=EV_TOOL_USE)])

    assert plan_mode._read_block_targets(block) == []


def test_next_work_target_finds_the_first_edit_after_the_read_block():
    read_block = _read_block_with_targets("/repo/a.py")
    write_event = Event(uuid="w0", ts=BASE_TIME + timedelta(seconds=10), type=EV_TOOL_USE,
                        tool=ToolCall(id="tw", name="Edit", input={"file_path": "/repo/a.py"}))
    write_block = Block(kind=WRITE, events=[write_event])
    session = Session(session_id="s", project="p", blocks=[read_block, write_block])

    assert plan_mode._next_work_target(session, read_block) == "/repo/a.py"


def test_next_work_target_is_empty_when_nothing_follows():
    read_block = _read_block_with_targets("/repo/a.py")
    session = Session(session_id="s", project="p", blocks=[read_block])

    assert plan_mode._next_work_target(session, read_block) == ""


def test_justification_prompt_lists_targets_and_the_next_change():
    prompt = build_justification_prompt(
        "fix the login bug", ["/repo/a.py", "/repo/b.py"], "/repo/a.py")

    assert "fix the login bug" in prompt
    assert "/repo/a.py" in prompt
    assert "/repo/b.py" in prompt
    assert "ONE" in prompt


def test_justification_prompt_degrades_gracefully_with_no_targets():
    prompt = build_justification_prompt("do something", [], "")

    assert "no file paths or search terms" in prompt
    assert "not recorded" in prompt


def test_justify_returns_none_when_the_llm_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: False)
    session = Session(session_id="s", project="p", user_prompts=["fix it"])
    read_block = _read_block_with_targets("/repo/a.py")

    assert justify(session, read_block) is None


def test_justify_returns_none_with_no_task_to_ground_the_question_in(
    monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    session = Session(session_id="s", project="p")  # no prompts, no title
    read_block = _read_block_with_targets("/repo/a.py")

    assert justify(session, read_block) is None


def test_justify_returns_the_judges_sentence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    calls = install_fake_openai(monkeypatch, lambda **kwargs: _fake_response(
        "It skimmed six files to map the module before editing any of them."))
    session = Session(session_id="s", project="p", user_prompts=["fix the login bug"])
    read_block = _read_block_with_targets("/repo/a.py", "/repo/b.py")

    result = justify(session, read_block)

    assert result == "It skimmed six files to map the module before editing any of them."
    assert len(calls) == 1


def test_justify_caches_so_a_repeat_call_never_reaches_the_llm_again(
    monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    calls = install_fake_openai(monkeypatch, lambda **kwargs: _fake_response("a reason"))
    session = Session(session_id="s", project="p", user_prompts=["fix the login bug"])
    read_block = _read_block_with_targets("/repo/a.py")

    justify(session, read_block)
    justify(session, read_block)

    assert len(calls) == 1


def test_justify_returns_none_when_the_call_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: True)

    def raiser(**kwargs):
        raise RuntimeError("connection refused")

    install_fake_openai(monkeypatch, raiser)
    session = Session(session_id="s", project="p", user_prompts=["fix the login bug"])
    read_block = _read_block_with_targets("/repo/a.py")

    assert justify(session, read_block) is None


def test_a_failed_call_is_retried_rather_than_cached(monkeypatch: pytest.MonkeyPatch):
    """A timeout used to be cached as the empty string, permanently losing the
    justification even after the failure was transient. A failure must not be
    cached, so the next call gets a real second attempt."""
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    attempts = {"n": 0}

    def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("timed out")
        return _fake_response("a reason, found on the second try")

    install_fake_openai(monkeypatch, flaky)
    session = Session(session_id="s", project="p", user_prompts=["fix the login bug"])
    read_block = _read_block_with_targets("/repo/a.py")

    assert justify(session, read_block) is None
    assert justify(session, read_block) == "a reason, found on the second try"
    assert attempts["n"] == 2, "the second call must have reached the LLM, not a stale cache entry"


def test_detect_includes_a_justification_when_the_llm_is_available(
    claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    install_fake_openai(monkeypatch, lambda **kwargs: _fake_response(
        "Six files were read to understand the logging format before any change."))

    path = _front_loaded_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    session.blocks = build_blocks(session, use_judge=False)

    problem = detect(session)

    assert problem is not None
    assert problem.data["justification"] == (
        "Six files were read to understand the logging format before any change.")
