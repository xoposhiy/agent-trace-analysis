"""Tests for the independent-tasks-in-one-session detector (CLAUDE.md §6).

Mirrors ``test_plan_mode.py``'s own structure:
  * pure segmentation logic (``compress_timeline``/``detect_interleaving``/
    ``_top_level_bands``) against hand-built assignment lists.
  * ``build_task_forest_prompt`` — pure string assertions.
  * ``judge_task_forest`` — unavailable/success/cached/failure, via a fake
    ``openai`` module, so no VPN is ever needed.
  * ``price_multi_split`` — the adapter into the unmodified
    ``chunk_split_model`` math, checked against a hand-computed dollar figure.
  * ``detect`` — end to end, through a real synthetic transcript, with the
    judge itself faked.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from Final_app.adapters import claude_code
from Final_app.analysis import classify, task_forest
from Final_app.analysis.attribution import attribute
from Final_app.analysis.task_forest import (
    build_task_forest_prompt,
    compress_timeline,
    detect,
    detect_interleaving,
    is_subtask,
    judge_task_forest,
    price_multi_split,
    top_level,
)
from Final_app.ir.models import EV_TOOL_USE, Event, Session, Tokens

from .conftest import (
    PROJECT_SLUG,
    assistant_tool_line,
    tool_result_line,
    user_line,
    write_transcript,
)

BASE_TIME = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_forest_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Redirect the on-disk judge cache so tests never touch or depend on
    the real ``~/.cache/tracelens``, and never inherit another test's
    cached answer for the same content hash."""
    monkeypatch.setattr(task_forest, "_CACHE_FILE", tmp_path / "task_forest.json")
    monkeypatch.setattr(task_forest, "_cache", None)


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, on_call):
    """Stand a fake ``openai`` module up so no request ever leaves the process.

    Mirrors ``test_plan_mode.py``'s own helper of the same shape.
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


# ----------------------------------------------------------------------
# Pure segmentation logic
# ----------------------------------------------------------------------

def test_top_level_strips_a_child_suffix():
    assert top_level("T1.2") == "T1"
    assert top_level("T1") == "T1"


def test_is_subtask_detects_a_dotted_id():
    assert is_subtask("T1.2") is True
    assert is_subtask("T1") is False


def test_join_natural_reads_as_a_sentence():
    assert task_forest._join_natural([]) == ""
    assert task_forest._join_natural(["T1"]) == "T1"
    assert task_forest._join_natural(["T1", "T2"]) == "T1 and T2"
    assert task_forest._join_natural(["T1", "T2", "T3"]) == "T1, T2, and T3"


def test_compress_timeline_collapses_consecutive_runs():
    assert compress_timeline(["T1", "T1", "T2", "T1"]) == [
        ("T1", 1, 2), ("T2", 3, 3), ("T1", 4, 4),
    ]


def test_detect_interleaving_counts_switches_and_returns():
    result = detect_interleaving(["T1", "T1", "T2", "T1"])

    assert result["runs"] == [("T1", 1, 2), ("T2", 3, 3), ("T1", 4, 4)]
    assert result["num_switches"] == 2
    assert result["recurring"] == ["T1"]


def test_detect_interleaving_folds_a_child_dip_into_its_parent():
    """T1 -> T1.1 -> T1 is one continuous stretch of T1, not a return."""
    result = detect_interleaving(["T1", "T1.1", "T1"])

    assert result["recurring"] == []


def test_top_level_bands_merges_a_child_dip_into_its_parent():
    timeline = compress_timeline(["T1", "T1.1", "T1", "T2"])

    assert task_forest._top_level_bands(timeline) == [("T1", 1, 3), ("T2", 4, 4)]


def test_top_level_bands_keeps_a_later_return_as_its_own_band():
    timeline = compress_timeline(["T1", "T2", "T1"])

    assert task_forest._top_level_bands(timeline) == [
        ("T1", 1, 1), ("T2", 2, 2), ("T1", 3, 3),
    ]


# ----------------------------------------------------------------------
# _cluster_independent_spans — what actually gets priced a split
# ----------------------------------------------------------------------
# A switch is only worth pricing when it is a clean handoff to a task the
# user never returns to. A stretch of back-and-forth between RECURRING
# tasks has no such handoff anywhere inside it, so it must price as one
# span, however many bands the judge reported within it.

def _bands_of(assignments: list[str]) -> list[tuple]:
    return task_forest._top_level_bands(compress_timeline(assignments))


def test_all_independent_tasks_each_get_their_own_span():
    bands = _bands_of(["T1", "T2", "T3"])

    assert task_forest._cluster_independent_spans(bands, recurring=set()) == [
        (0, 0), (1, 1), (2, 2),
    ]


def test_full_interleaving_prices_as_one_span():
    """T1, T2, T1, T2 — both recurring — must not split at any of the three
    internal switches: there is no clean handoff inside a tangle nobody
    would actually want as four separate sessions."""
    bands = _bands_of(["T1", "T2", "T1", "T2"])

    assert task_forest._cluster_independent_spans(bands, recurring={"T1", "T2"}) == [
        (0, 3),
    ]


def test_an_independent_task_after_interleaving_gets_its_own_span():
    """T1, T2, T1 (T1 recurring, T2 a one-off sandwiched inside it), then a
    genuinely independent T3: the interleaved stretch prices as one span,
    and T3 — never returned to — starts a real, new one."""
    bands = _bands_of(["T1", "T2", "T1", "T3"])

    assert task_forest._cluster_independent_spans(bands, recurring={"T1"}) == [
        (0, 2), (3, 3),
    ]


def test_an_independent_task_before_interleaving_gets_its_own_span():
    bands = _bands_of(["T1", "T2", "T3", "T2"])

    assert task_forest._cluster_independent_spans(bands, recurring={"T2"}) == [
        (0, 0), (1, 3),
    ]


def test_two_separate_interleavings_stay_separate_spans():
    """T1<->T2 tangle, then a clean handoff, then a separate T3<->T4 tangle —
    two spans, not one: nothing links the first tangle's reach to the
    second's."""
    bands = _bands_of(["T1", "T2", "T1", "T3", "T4", "T3"])

    assert task_forest._cluster_independent_spans(
        bands, recurring={"T1", "T3"}
    ) == [(0, 2), (3, 5)]


def test_no_bands_gives_no_spans():
    assert task_forest._cluster_independent_spans([], recurring=set()) == []


# ----------------------------------------------------------------------
# build_task_forest_prompt
# ----------------------------------------------------------------------

def test_prompt_numbers_every_message_and_asks_for_hierarchical_ids():
    prompt = build_task_forest_prompt(["fix the login bug", "now optimize the query"])

    assert "1. fix the login bug" in prompt
    assert "2. now optimize the query" in prompt
    assert "T1.1" in prompt
    assert "MINIFIED JSON" in prompt


def test_prompt_states_a_boundary_falls_only_between_numbered_messages():
    prompt = build_task_forest_prompt(["fix the login bug", "now optimize the query"])

    assert "ATOMIC MESSAGE" in prompt
    assert "never split one, never merge two" in prompt


def test_prompt_states_the_exact_expected_assignment_count():
    """The judge model (haiku) drops messages from its own ``assignments``

    array once a session gets long enough — verified against real sessions,
    2026-08-23: a 48-message session came back with only 39-41 assignments
    under the old prompt, silently failing ``detect``'s length guard on most
    real sessions. Stating the exact count twice and asking the model to
    self-check fixed it in live testing.
    """
    prompt = build_task_forest_prompt(["a", "b", "c"])

    assert "EXACTLY 3 messages" in prompt
    assert "EXACTLY 3 entries" in prompt


# ----------------------------------------------------------------------
# judge_task_forest
# ----------------------------------------------------------------------

def test_judge_returns_none_when_the_llm_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: False)

    assert judge_task_forest(["a", "b"]) is None


def test_judge_returns_the_parsed_forest(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    forest_json = json.dumps({"tasks": [{"id": "T1", "label": "Fix login"}],
                              "assignments": ["T1", "T1"], "summary": "Fixed login"})
    calls = install_fake_openai(monkeypatch, lambda **kwargs: _fake_response(forest_json))

    result = judge_task_forest(["fix login", "fix login again"])

    assert result == {"tasks": [{"id": "T1", "label": "Fix login"}],
                      "assignments": ["T1", "T1"], "summary": "Fixed login"}
    assert len(calls) == 1


def test_judge_caches_so_a_repeat_call_never_reaches_the_llm_again(
    monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    forest_json = json.dumps({"tasks": [], "assignments": ["T1"], "summary": ""})
    calls = install_fake_openai(monkeypatch, lambda **kwargs: _fake_response(forest_json))

    judge_task_forest(["hi"])
    judge_task_forest(["hi"])

    assert len(calls) == 1


def test_judge_cache_key_changes_when_the_prompt_wording_changes(
    monkeypatch: pytest.MonkeyPatch
):
    """A prompt-template edit must invalidate every prior verdict.

    ``_forest_cache_key`` hashes the FULL rendered prompt, not just the raw
    ``prompt_texts`` — otherwise a session judged under an old prompt version
    would keep serving that stale answer forever after the prompt (and what
    "correct" means) changes. This pins that behaviour directly, rather than
    relying on the real prompt text staying edited.
    """
    key_before = task_forest._forest_cache_key(["hi"])

    monkeypatch.setattr(task_forest, "build_task_forest_prompt", lambda texts: "a different prompt")
    key_after = task_forest._forest_cache_key(["hi"])

    assert key_before != key_after


def test_judge_returns_none_when_the_response_is_not_valid_json(
    monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    install_fake_openai(monkeypatch, lambda **kwargs: _fake_response("not json"))

    assert judge_task_forest(["hi"]) is None


def test_judge_returns_none_when_the_call_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: True)

    def raiser(**kwargs):
        raise RuntimeError("connection refused")

    install_fake_openai(monkeypatch, raiser)

    assert judge_task_forest(["hi"]) is None


def test_a_failed_call_is_retried_rather_than_cached(monkeypatch: pytest.MonkeyPatch):
    """A timeout used to be cached as ``None`` — indistinguishable from, and
    permanently masking, a genuine "no split-worthy pattern" verdict. Against
    a reasoning model whose calls can legitimately run 40s+, that baked real
    timeouts in as false negatives forever. A failure must not be cached, so
    the next scan gets a real second attempt."""
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    attempts = {"n": 0}

    def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("timed out")
        return _fake_response(json.dumps(
            {"tasks": [{"id": "T1", "label": "x"}], "assignments": ["T1"], "summary": "x"}
        ))

    install_fake_openai(monkeypatch, flaky)

    assert judge_task_forest(["hi"]) is None
    assert judge_task_forest(["hi"]) is not None
    assert attempts["n"] == 2, "the second call must have reached the LLM, not a stale cache entry"


# ----------------------------------------------------------------------
# price_multi_split — the adapter into chunk_split_model
# ----------------------------------------------------------------------

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


def test_price_multi_split_returns_none_with_no_cache_read():
    session = Session(session_id="s", project="p", events=[_call(i) for i in range(30)])

    assert price_multi_split(session, [1 / 3, 2 / 3]) is None


def test_price_multi_split_returns_none_when_no_fraction_survives_the_guard():
    session = Session(session_id="s", project="p",
                      events=[_call(i, 1_000_000) for i in range(30)])

    assert price_multi_split(session, [0.0, 1.0]) is None


def test_price_multi_split_as_is_cost_is_the_sessions_real_attributed_cost():
    """``as_is_cost`` must be ``session.attributed_cost`` itself, not a
    second, independently-computed estimate — mirrors
    ``test_plan_mode``'s own pin of this."""
    session = Session(
        session_id="s", project="p", model="claude-opus-4-6",
        events=[_call(i, 1_000_000, cost=0.5) for i in range(30)],
    )

    saving = price_multi_split(session, [1 / 3, 2 / 3])

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(session.attributed_cost)
    assert saving["as_is_cost"] == pytest.approx(15.0)


def test_price_multi_split_matches_a_hand_computed_saving():
    """30 calls, 30M cache-read tokens, cut into 3 EQUAL thirds, on a session
    whose model is ``claude-opus-4-6`` ($5.00/$25.00 per MTok). ``as_is_cost``
    is the fixture's own $15.00 of real attributed cost (each of the 30
    calls carries $0.50, set directly since this fixture builds ``Event``s by
    hand); ``split_cost`` is still the ``chunk_split_model`` ESTIMATE, since
    there is no exact figure for a split that never happened.

    Hand-computed at THIS session's real per-model rate
    (``plan_mode._rates_for_session``), not ``chunk_split_model``'s flat
    model-agnostic default of 4.75/23.75. Splitting a uniform ramp into k
    equal chunks reduces the cache-read sum by exactly 1/k (each of the k
    equal-height chunks contributes 1/k^2 of the original triangle's area,
    times k chunks = 1/k), so cache-read-after = 30e6/3 = 10e6 exactly.
    Summary tokens per new chunk = 0.1 * (peak/3) = 200000/3, carried into a
    10-call chunk each time (two new chunks): overhead per chunk =
    (25*200000/3 + 1.25*5*200000/3 + 0.10*5*200000/3*9) / 1e6 ~= 2.383333, so
    total overhead ~= 4.766667. split = (10e6*5*0.10)/1e6 + overhead ~=
    5.0+4.766667 = 9.766667; saving = 15.0 - 9.766667 ~= 5.233333 (34.89%).
    """
    session = Session(session_id="s", project="p", model="claude-opus-4-6",
                      events=[_call(i, 1_000_000, cost=0.5) for i in range(30)])

    saving = price_multi_split(session, [1 / 3, 2 / 3])

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(15.0)
    assert saving["cache_read_after"] == pytest.approx(10_000_000)
    assert saving["split_cost"] == pytest.approx(9.766667, rel=1e-4)
    assert saving["dollar_saving"] == pytest.approx(5.233333, rel=1e-4)
    assert saving["percent_saving"] == pytest.approx(34.888889, rel=1e-4)
    assert saving["token_saving"] == pytest.approx(20_000_000)
    assert saving["num_chunks"] == 3


def test_price_multi_split_falls_back_to_the_default_rate_with_no_model_recorded():
    """A session with no ``model`` field must still price ``split_cost``,
    not crash. ``as_is_cost`` is unaffected by the model at all now, since
    it comes straight from ``session.attributed_cost``."""
    session = Session(session_id="s", project="p",
                      events=[_call(i, 1_000_000, cost=1.0) for i in range(30)])

    saving = price_multi_split(session, [1 / 3, 2 / 3])

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(30.0)


def test_price_multi_split_adds_subagent_cost_to_both_sides_unchanged():
    """Mirrors ``test_plan_mode``'s own pin of this: a subagent's dollars
    land in ``as_is_cost`` "for free" (it is ``session.attributed_cost``,
    which already sums every event) and are added to ``split_cost`` on top
    of the ramp estimate, leaving ``dollar_saving`` unchanged and only
    moving ``percent_saving``."""
    main_events = [_call(i, 1_000_000, cost=0.5) for i in range(30)]
    subagent_event = Event(
        uuid="sub-e0", ts=BASE_TIME, type=EV_TOOL_USE, message_id="sub-m0",
        agent_id="sub1", attributed_cost=4.0,
    )
    without_subagent = Session(session_id="s", project="p", model="claude-opus-4-6",
                               events=main_events)
    with_subagent = Session(session_id="s", project="p", model="claude-opus-4-6",
                            events=main_events + [subagent_event])

    baseline = price_multi_split(without_subagent, [1 / 3, 2 / 3])
    plus_subagent = price_multi_split(with_subagent, [1 / 3, 2 / 3])

    assert plus_subagent["as_is_cost"] == pytest.approx(baseline["as_is_cost"] + 4.0)
    assert plus_subagent["split_cost"] == pytest.approx(baseline["split_cost"] + 4.0)
    assert plus_subagent["dollar_saving"] == pytest.approx(baseline["dollar_saving"])
    assert plus_subagent["percent_saving"] < baseline["percent_saving"]


# ----------------------------------------------------------------------
# detect — end to end, through a real transcript, judge faked
# ----------------------------------------------------------------------

def _iso_at(offset_seconds: int) -> str:
    ts = BASE_TIME + timedelta(seconds=offset_seconds)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _two_task_transcript(claude_home: Path, prompts: list[str] | None = None) -> Path:
    """Two unrelated tasks, two prompts each, five cache-read-heavy edits
    after every prompt — enough calls either side of the switch to price a
    real saving. The judge's own segmentation is faked in each test; the
    prompt text here only has to be distinct enough to look like two goals.

    Each edit also carries a small, realistic ``input``/``output`` figure —
    a real transcript never bills zero on every channel but ``cache_read``,
    and ``price_multi_split`` now needs ``session.attributed_cost`` to be a
    real, complete number rather than the near-total loss
    ``analysis.attribution`` suffers when every call's ``fresh`` channel is
    zero. See ``test_plan_mode``'s ``_front_loaded_transcript`` for the same
    fix and the fuller explanation.
    """
    lines = []
    prompts = prompts if prompts is not None else [
        "Fix the login bug",
        "Also add a test for the login fix",
        "Now optimize the slow database query",
        "Add an index to speed it up further",
    ]
    second = 0
    for prompt in prompts:
        lines.append(user_line(f"u{second}", prompt, _iso_at(second)))
        second += 10
        for i in range(5):
            when = _iso_at(second)
            lines.append(assistant_tool_line(
                f"a{second}", f"msg{second}", f"t{second}", "Edit",
                {"file_path": "/repo/f.py"}, when,
                input_tokens=30, output_tokens=25, cache_read=500_000,
            ))
            lines.append(tool_result_line(f"r{second}", f"t{second}", "ok", when))
            second += 1
    return write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)


def _install_forest_judge(monkeypatch: pytest.MonkeyPatch, assignments, tasks):
    monkeypatch.setattr(classify, "llm_available", lambda: True)
    forest_json = json.dumps({"tasks": tasks, "assignments": assignments, "summary": "x"})
    return install_fake_openai(monkeypatch, lambda **kwargs: _fake_response(forest_json))


def test_detect_flags_two_independent_tasks(claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    _install_forest_judge(
        monkeypatch, ["T1", "T1", "T2", "T2"],
        [{"id": "T1", "label": "Fix login bug"}, {"id": "T2", "label": "Optimize query"}],
    )

    problem = detect(session)

    assert problem is not None
    assert problem.id == "task-switch"
    assert problem.severity == "medium"
    assert [t["id"] for t in problem.data["tasks"]] == ["T1", "T2"]
    assert len(problem.data["runs"]) == 2
    assert problem.data["runs"][0]["label"] == "Fix login bug"
    assert problem.data["runs"][1]["end_ts"] is None
    assert problem.data["dollar_saving"] > 0


def test_detect_prices_interleaving_as_one_span_not_every_switch(
    claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """T3, T1, T2, T1: T3 is a genuinely independent task done once up front;
    T1 and T2 then interleave for the rest of the session (T1 recurring).
    The task lane (``runs``) still shows all four bands, but the PRICED
    split only cuts once — after T3 — not three times at every switch
    within the T1/T2 tangle.
    """
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    _install_forest_judge(
        monkeypatch, ["T3", "T1", "T2", "T1"],
        [{"id": "T3", "label": "Fix login bug"},
         {"id": "T1", "label": "Optimize query"},
         {"id": "T2", "label": "A quick detour"}],
    )

    problem = detect(session)

    assert problem is not None
    assert len(problem.data["runs"]) == 4, "the task lane stays band-by-band"
    assert problem.data["num_chunks"] == 2, \
        "T3's own span, plus the interleaved T1/T2/T1 stretch priced as one"
    assert problem.severity == "high", \
        "severity is driven by whether ANY task recurs, independent of clustering"
    # What each priced chunk actually contains: T3 alone, then the merged
    # T1/T2 stretch — readable as "T1 and T2", not silently just "T1".
    assert problem.data["chunks"] == [
        {"task_ids": ["T3"], "label": "T3"},
        {"task_ids": ["T1", "T2"], "label": "T1 and T2"},
    ]
    # The detail sentence must name the actual chunk count, never claim "at
    # each switch" — this session has 3 switches (T3->T1, T1->T2, T2->T1)
    # but only 2 priced chunks, and saying "at each switch" would be wrong.
    assert "2 separate sessions" in problem.detail
    assert "at each switch" not in problem.detail
    assert "back-and-forth" in problem.detail, \
        "recurring tasks are involved, so the detail should say they stay together"


def test_detail_omits_the_interleaving_note_with_no_recurring_task(
    claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """T1, T1, T2, T2 — no task recurs, so every switch is a genuine,
    independent handoff and the "back-and-forth" note would be misleading."""
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    _install_forest_judge(
        monkeypatch, ["T1", "T1", "T2", "T2"],
        [{"id": "T1", "label": "Fix login bug"}, {"id": "T2", "label": "Optimize query"}],
    )

    problem = detect(session)

    assert problem is not None
    assert problem.data["num_chunks"] == 2
    assert "2 separate sessions" in problem.detail
    assert "back-and-forth" not in problem.detail
    assert problem.data["chunks"] == [
        {"task_ids": ["T1"], "label": "T1"},
        {"task_ids": ["T2"], "label": "T2"},
    ]


def test_detect_strips_injected_machinery_before_it_reaches_the_judge(
    claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """A trailing ``<system-reminder>`` on an otherwise-human line still

    counts as ``is_human_prompt`` (the human part is real), but it must never
    reach the judge as if the user had typed it.
    """
    path = _two_task_transcript(claude_home, prompts=[
        "Fix the login bug",
        "Also add a test for the login fix\n"
        "<system-reminder>remember the thing</system-reminder>",
        "Now optimize the slow database query",
        "Add an index to speed it up further",
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    calls = _install_forest_judge(
        monkeypatch, ["T1", "T1", "T2", "T2"],
        [{"id": "T1", "label": "Fix login bug"}, {"id": "T2", "label": "Optimize query"}],
    )

    detect(session)

    sent_prompt = calls[0]["messages"][0]["content"]
    assert "<system-reminder>" not in sent_prompt
    assert "remember the thing" not in sent_prompt
    assert "Also add a test for the login fix" in sent_prompt


def test_detect_returns_none_for_a_single_task(claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    _install_forest_judge(
        monkeypatch, ["T1", "T1", "T1", "T1"], [{"id": "T1", "label": "One thing"}],
    )

    assert detect(session) is None


def test_detect_returns_none_when_everything_interleaves(
    claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """T1, T2, T1, T2 — both tasks recurring, nothing independent anywhere —
    leaves ``_cluster_independent_spans`` with a single span covering the
    whole session and zero fractions to price. There is no clean handoff to
    propose splitting at, so this must report nothing rather than fall back
    to pricing every switch (the old behaviour this replaces)."""
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    _install_forest_judge(
        monkeypatch, ["T1", "T2", "T1", "T2"],
        [{"id": "T1", "label": "Login"}, {"id": "T2", "label": "Query"}],
    )

    assert detect(session) is None


def test_detect_returns_none_with_too_few_prompts(claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    """The judge is never even called below the prompt-count gate."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)
    calls = _install_forest_judge(monkeypatch, ["T1"], [{"id": "T1", "label": "x"}])

    assert detect(session) is None
    assert len(calls) == 0


def test_detect_returns_none_when_the_llm_is_unavailable(claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: False)
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    attribute(session)

    assert detect(session) is None
