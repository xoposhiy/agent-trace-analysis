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


# ----------------------------------------------------------------------
# price_multi_split — the adapter into chunk_split_model
# ----------------------------------------------------------------------

def _call(index: int, cache_read: int = 0) -> Event:
    return Event(
        uuid=f"e{index}", ts=BASE_TIME + timedelta(seconds=index),
        type=EV_TOOL_USE, message_id=f"m{index}",
        tokens=Tokens(cache_read=cache_read),
    )


def test_price_multi_split_returns_none_with_no_cache_read():
    session = Session(session_id="s", project="p", events=[_call(i) for i in range(30)])

    assert price_multi_split(session, [1 / 3, 2 / 3]) is None


def test_price_multi_split_returns_none_when_no_fraction_survives_the_guard():
    session = Session(session_id="s", project="p",
                      events=[_call(i, 1_000_000) for i in range(30)])

    assert price_multi_split(session, [0.0, 1.0]) is None


def test_price_multi_split_matches_a_hand_computed_saving():
    """30 calls, 30M cache-read tokens, cut into 3 EQUAL thirds.

    Splitting a uniform ramp into k equal chunks reduces the cache-read sum
    by exactly 1/k (each of the k equal-height chunks contributes 1/k^2 of
    the original triangle's area, times k chunks = 1/k), so cache-read-after
    = 30e6/3 = 10e6 exactly. Summary tokens per new chunk = 0.1 * (peak/3) =
    200000/3, carried into a 10-call chunk each time (two new chunks):
    overhead per chunk = (23.75*200000/3 + 1.25*4.75*200000/3 +
    0.10*4.75*200000/3*9) / 1e6 ~= 2.264167, so total overhead ~= 4.528333.
    as-is = (30e6*4.75*0.10)/1e6 = 14.25; split = (10e6*4.75*0.10)/1e6 +
    overhead ~= 4.75+4.528333 = 9.278333; saving ~= 4.971667 (34.89%).
    """
    session = Session(session_id="s", project="p",
                      events=[_call(i, 1_000_000) for i in range(30)])

    saving = price_multi_split(session, [1 / 3, 2 / 3])

    assert saving is not None
    assert saving["as_is_cost"] == pytest.approx(14.25)
    assert saving["cache_read_after"] == pytest.approx(10_000_000)
    assert saving["split_cost"] == pytest.approx(9.278333, rel=1e-4)
    assert saving["dollar_saving"] == pytest.approx(4.971667, rel=1e-4)
    assert saving["percent_saving"] == pytest.approx(34.888889, rel=1e-4)
    assert saving["token_saving"] == pytest.approx(20_000_000)
    assert saving["num_chunks"] == 3


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
                {"file_path": "/repo/f.py"}, when, cache_read=500_000,
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
    _install_forest_judge(
        monkeypatch, ["T1", "T1", "T1", "T1"], [{"id": "T1", "label": "One thing"}],
    )

    assert detect(session) is None


def test_detect_marks_a_recurring_task_switch_as_high_severity(
    claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)
    _install_forest_judge(
        monkeypatch, ["T1", "T2", "T1", "T2"],
        [{"id": "T1", "label": "Login"}, {"id": "T2", "label": "Query"}],
    )

    problem = detect(session)

    assert problem is not None
    assert problem.severity == "high"
    # Both T1 and T2 alternate twice each — both count as recurring.
    assert problem.data["recurring"] == ["T1", "T2"]


def test_detect_returns_none_with_too_few_prompts(claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    """The judge is never even called below the prompt-count gate."""
    path = write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", [
        user_line("u1", "hi", "2026-08-01T10:00:00.000Z"),
    ])
    session = claude_code.load_session(PROJECT_SLUG, path)
    calls = _install_forest_judge(monkeypatch, ["T1"], [{"id": "T1", "label": "x"}])

    assert detect(session) is None
    assert len(calls) == 0


def test_detect_returns_none_when_the_llm_is_unavailable(claude_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(classify, "llm_available", lambda: False)
    path = _two_task_transcript(claude_home)
    session = claude_code.load_session(PROJECT_SLUG, path)

    assert detect(session) is None
