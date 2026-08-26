"""API tests: endpoint shapes, filtering, and cache behaviour.

Nothing here reaches the network. Block classification is the only remaining
LLM caller and the detail endpoint serves it from cache, so these tests assert
on routing, attribution and caching only.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from Final_app.adapters import claude_code
from Final_app.analysis import classify, task_forest
from Final_app.api import app as api

from .conftest import (
    PROJECT_SLUG,
    assistant_text_line,
    assistant_tool_line,
    tool_result_line,
    user_line,
    write_transcript,
)


@pytest.fixture(autouse=True)
def clean_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Empty the module-level parse cache and cut every route to the network.

    ``_cache`` persists for the process lifetime, so without this one test's
    sessions show up in the next.

    The detail endpoint builds blocks, and classification calls the judge for
    ambiguous Bash commands. ``classify.llm_available()`` is true whenever
    ``OPENAI_API_KEY`` is set, so on a developer machine these tests would
    reach the LiteLLM proxy and write to the real ``~/.cache/tracelens``,
    making them pass or fail on whether the VPN is up (CLAUDE.md §6).
    """
    monkeypatch.setattr(classify, "llm_available", lambda: False)
    monkeypatch.setattr(classify, "_CACHE_FILE", tmp_path / "tool_kinds.json")
    monkeypatch.setattr(classify, "_cache", None)

    api._cache.clear()
    yield
    api._cache.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


# ----------------------------------------------------------------------
# Health and projects
# ----------------------------------------------------------------------

def test_health_reports_the_projects_root(client: TestClient, claude_home: Path):
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert body["projects_root"] == str(claude_home)


def test_health_explains_why_the_llm_is_off(client: TestClient, claude_home: Path,
                                            monkeypatch: pytest.MonkeyPatch):
    """A disabled LLM must say why, not fail silently."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    body = client.get("/api/health").json()

    assert body["llm"]["enabled"] is False
    assert body["llm"]["reasons"], "expected a stated reason"


def test_llm_check_reports_a_broken_llm_without_failing_the_request(
    client: TestClient, claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """A wrong configuration must produce an error *message*, not an HTTP error.

    The banner is the only thing that tells a user their VPN is down; a 500
    here would leave them with a silent fallback to the shell heuristic.
    """
    monkeypatch.setattr(api, "probe_llm", lambda force=False: {
        "ok": False, "configured": True,
        "reason": "the LLM host could not be reached",
        "hint": "Check OPENAI_BASE_URL and your VPN.",
        "latency_ms": 8000, "model": "some-model", "cached": False,
    })
    response = client.get("/api/llm-check")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["reason"] and body["hint"], "a failure must say why and what to do"


def test_llm_check_passes_force_through(client: TestClient, claude_home: Path,
                                        monkeypatch: pytest.MonkeyPatch):
    """The Retry button has to bypass the one-minute cache to be useful."""
    seen: list[bool] = []
    monkeypatch.setattr(api, "probe_llm", lambda force=False: (
        seen.append(force) or {"ok": True, "configured": True, "reason": "",
                               "hint": "", "latency_ms": 1, "model": "m",
                               "cached": False}))

    client.get("/api/llm-check")
    client.get("/api/llm-check?force=true")

    assert seen == [False, True]


def test_projects_lists_slugs_with_counts(client: TestClient, claude_home: Path,
                                          simple_session: Path):
    projects = client.get("/api/projects").json()["projects"]

    assert len(projects) == 1
    assert projects[0]["slug"] == PROJECT_SLUG
    assert projects[0]["count"] == 1


# ----------------------------------------------------------------------
# Session list
# ----------------------------------------------------------------------

def test_sessions_returns_the_list_shape(client: TestClient, claude_home: Path,
                                         simple_session: Path):
    body = client.get("/api/sessions").json()

    assert body["total"] == 1
    session = body["sessions"][0]
    assert session["title"] == "Fix the login bug"
    assert session["tool_call_count"] == 3
    assert session["project_label"]


def test_sessions_are_newest_first(client: TestClient, claude_home: Path):
    for name, hour in (("old", "09"), ("new", "11")):
        write_transcript(claude_home / PROJECT_SLUG / f"{name}.jsonl", [
            user_line("u1", name, f"2026-08-01T{hour}:00:00.000Z"),
        ])
    body = client.get("/api/sessions").json()

    assert [s["session_id"] for s in body["sessions"]] == ["new", "old"]


def test_project_filter_narrows_the_list(client: TestClient, claude_home: Path,
                                         simple_session: Path):
    other = "-Users-tester-Desktop-other-repo"
    write_transcript(claude_home / other / "x.jsonl", [
        user_line("u1", "elsewhere", "2026-08-01T10:00:00.000Z"),
    ])

    assert client.get("/api/sessions").json()["total"] == 2
    assert client.get(f"/api/sessions?project={other}").json()["total"] == 1


def test_unknown_project_yields_an_empty_list_not_an_error(
    client: TestClient, claude_home: Path, simple_session: Path
):
    response = client.get("/api/sessions?project=-does-not-exist")

    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_severity_filter_narrows_the_returned_rows(client: TestClient,
                                                   claude_home: Path,
                                                   simple_session: Path):
    """Every listed row reports ``none`` here, since the list never runs a
    detector — only ``session_detail()`` does, once a session's own page is
    opened (see ``test_session_detail_reports_a_detected_problem`` below).

    Asserted on the rows, not on ``total``: since paging parses only the page
    it returns, ``total`` counts candidate transcripts on disk and cannot
    account for a filter that needs a session parsed to evaluate.
    """
    assert len(client.get("/api/sessions?severity=any").json()["sessions"]) == 1
    assert len(client.get("/api/sessions?severity=high").json()["sessions"]) == 0


def test_the_list_never_advertises_an_llm_summary(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """Summaries were removed; the free ``ai-title`` is the only label."""
    row = client.get("/api/sessions").json()["sessions"][0]

    assert "summary" not in row
    assert "pending_summaries" not in client.get("/api/sessions").json()
    assert row["title"] == "Fix the login bug"


# ----------------------------------------------------------------------
# Paging
# ----------------------------------------------------------------------
# The list opens only the transcripts it returns. A user with years of history
# was waiting many seconds for a page of twenty rows, because ordering came
# from ``last_ts`` and ``last_ts`` needs a parse. Ordering now comes from file
# mtime, which ``stat`` gives for free.

def _many_sessions(claude_home: Path, count: int) -> None:
    """``count`` transcripts, oldest first, one minute apart."""
    for index in range(count):
        path = write_transcript(claude_home / PROJECT_SLUG / f"s{index:03d}.jsonl", [
            user_line("u1", f"session {index}",
                      f"2026-08-01T10:{index:02d}:00.000Z"),
        ])
        # mtime is the sort key, so it has to match the message order.
        os.utime(path, (1_800_000_000 + index * 60, 1_800_000_000 + index * 60))


def test_the_list_returns_one_page_by_default(client: TestClient,
                                              claude_home: Path):
    _many_sessions(claude_home, 25)
    body = client.get("/api/sessions").json()

    assert len(body["sessions"]) == api.PAGE_SIZE == 20
    assert body["total"] == 25
    assert body["has_more"] is True


def test_the_first_page_is_the_newest_sessions(client: TestClient,
                                               claude_home: Path):
    _many_sessions(claude_home, 25)
    body = client.get("/api/sessions").json()

    assert body["sessions"][0]["session_id"] == "s024"
    assert body["sessions"][-1]["session_id"] == "s005"


def test_load_more_continues_where_the_page_ended(client: TestClient,
                                                  claude_home: Path):
    _many_sessions(claude_home, 25)
    first = client.get("/api/sessions").json()
    second = client.get("/api/sessions?offset=20").json()

    assert len(second["sessions"]) == 5
    assert second["has_more"] is False

    ids = [s["session_id"] for s in first["sessions"] + second["sessions"]]
    assert len(ids) == len(set(ids)) == 25, "paging must not repeat or skip"


def test_a_short_last_page_reports_no_more(client: TestClient, claude_home: Path,
                                           simple_session: Path):
    body = client.get("/api/sessions").json()

    assert body["total"] == 1
    assert body["has_more"] is False


def test_paging_past_the_end_is_empty_not_an_error(client: TestClient,
                                                   claude_home: Path,
                                                   simple_session: Path):
    response = client.get("/api/sessions?offset=500")

    assert response.status_code == 200
    assert response.json()["sessions"] == []
    assert response.json()["has_more"] is False


def test_the_page_size_is_capped(client: TestClient, claude_home: Path):
    """An unbounded ``limit`` would reintroduce the parse-everything cost."""
    assert client.get("/api/sessions?limit=5000").status_code == 422
    assert client.get("/api/sessions?offset=-1").status_code == 422


def test_the_project_filter_applies_before_paging(client: TestClient,
                                                  claude_home: Path):
    _many_sessions(claude_home, 25)
    other = "-Users-tester-Desktop-other-repo"
    write_transcript(claude_home / other / "x.jsonl", [
        user_line("u1", "elsewhere", "2026-08-01T10:00:00.000Z"),
    ])
    body = client.get(f"/api/sessions?project={other}").json()

    assert body["total"] == 1, "total must count only the filtered project"
    assert body["has_more"] is False
    assert [s["session_id"] for s in body["sessions"]] == ["x"]


def test_listing_a_page_parses_only_that_page(client: TestClient,
                                              claude_home: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    """The whole point of the change — assert it, do not assume it."""
    _many_sessions(claude_home, 25)

    parsed: list[str] = []
    real = claude_code.load_session

    def counting(project_slug, path, *args, **kwargs):
        parsed.append(path.stem)
        return real(project_slug, path, *args, **kwargs)

    monkeypatch.setattr(api.claude_code, "load_session", counting)
    client.get("/api/sessions")

    assert len(parsed) == 20, f"parsed {len(parsed)} transcripts for a 20-row page"


def test_the_detail_route_parses_only_its_own_session(
    client: TestClient, claude_home: Path, monkeypatch: pytest.MonkeyPatch
):
    _many_sessions(claude_home, 25)

    parsed: list[str] = []
    real = claude_code.load_session

    def counting(project_slug, path, *args, **kwargs):
        parsed.append(path.stem)
        return real(project_slug, path, *args, **kwargs)

    monkeypatch.setattr(api.claude_code, "load_session", counting)
    client.get("/api/sessions/s007")

    assert parsed == ["s007"]


def _front_loaded_reading_session(claude_home: Path) -> Path:
    """Six real-cache-read reads followed by ten edits — a missed plan-mode
    opportunity (see ``analysis/plan_mode.py`` and its own tests for why ten
    edits, not one). Each call also carries a small, realistic
    ``input``/``output`` figure — ``price_split``'s ``as_is_cost`` is now
    ``session.attributed_cost`` itself, which needs a non-degenerate
    ``fresh`` channel to attribute anything at all (see
    ``test_plan_mode._front_loaded_transcript`` for the fuller explanation).
    """
    lines = [user_line("u1", "Fix the bug", "2026-08-01T10:00:00.000Z")]
    for i in range(6):
        when = f"2026-08-01T10:00:{i + 1:02d}.000Z"
        lines.append(assistant_tool_line(
            f"a{i}", f"msg{i}", f"t{i}", "Read", {"file_path": f"/repo/f{i}.py"},
            when, input_tokens=50, output_tokens=40, cache_read=2_000_000,
        ))
        lines.append(tool_result_line(f"r{i}", f"t{i}", "contents", when))
    for i in range(10):
        when = f"2026-08-01T10:00:{i + 7:02d}.000Z"
        lines.append(assistant_tool_line(
            f"ae{i}", f"msg-edit{i}", f"te{i}", "Edit", {"file_path": "/repo/f0.py"}, when,
            input_tokens=30, output_tokens=25, cache_creation=40,
        ))
        lines.append(tool_result_line(f"re{i}", f"te{i}", "ok", when))
    return write_transcript(claude_home / PROJECT_SLUG / "s.jsonl", lines)


def test_session_detail_reports_a_detected_problem(client: TestClient, claude_home: Path):
    _front_loaded_reading_session(claude_home)

    data = client.get("/api/sessions/s").json()

    assert [p["id"] for p in data["problems"]] == ["plan-mode"]
    assert data["max_severity"] == "medium"


def test_session_detail_reports_no_problems_for_a_clean_session(
    client: TestClient, claude_home: Path, simple_session: Path
):
    data = client.get(f"/api/sessions/{simple_session.stem}").json()

    assert data["problems"] == []
    assert data["max_severity"] == "none"


def test_problems_endpoint_finds_a_problem_without_the_detail_page_being_opened_first(
    client: TestClient, claude_home: Path
):
    """Unlike ``/api/sessions``, this endpoint runs detection itself."""
    _front_loaded_reading_session(claude_home)

    data = client.get("/api/problems").json()

    assert len(data["problems"]) == 1
    row = data["problems"][0]
    assert row["session_id"] == "s"
    assert row["problem"]["id"] == "plan-mode"
    # The session's own summary fields ride along, so a row is self-contained
    # (no second request needed to know whose session this is or how big it was).
    assert row["title"] == ""  # this fixture never sets an ai-title line
    assert row["tokens"]["total"] > 0
    assert row["tool_call_count"] == 16
    assert row["subagent_count"] == 0


def test_problems_endpoint_scans_sessions_concurrently(
    client: TestClient, claude_home: Path
):
    """Each session's detectors are dominated by a judge-LLM network call, so
    scanning a page serially made a reasoning model's per-call latency additive
    across the whole page. A ``threading.Barrier`` only releases once every
    session's scan has started at the same time — if the pool ran sessions one
    after another instead, this deadlocks and ``client.get`` raises."""
    session_count = 3
    for i in range(session_count):
        write_transcript(claude_home / PROJECT_SLUG / f"concurrent{i}.jsonl", [
            user_line(f"u{i}", "hello", "2026-08-01T10:00:00.000Z"),
            assistant_text_line(f"a{i}", f"msg{i}", "hi", "2026-08-01T10:00:01.000Z"),
        ])

    barrier = threading.Barrier(session_count, timeout=2)

    def all_sessions_scanning_at_once(session):
        barrier.wait()
        return []

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(api, "detect_problems", all_sessions_scanning_at_once)
        response = client.get("/api/problems")

    assert response.status_code == 200


def test_problems_endpoint_is_empty_for_a_clean_session(
    client: TestClient, claude_home: Path, simple_session: Path
):
    data = client.get("/api/problems").json()

    assert data["problems"] == []
    assert data["has_more"] is False
    assert data["total_sessions"] == 1


def _two_task_session(claude_home: Path) -> Path:
    """Four prompts, two unrelated goals — a candidate for the task-switch
    detector (see ``test_task_forest.py`` for the exact pricing shape and
    why every edit also carries a small, realistic ``input``/``output``
    figure)."""
    lines = []
    prompts = ["Fix the login bug", "Add a test for it",
              "Now optimize the slow query", "Add an index too"]
    second = 0
    for prompt in prompts:
        lines.append(user_line(f"u{second}", prompt, f"2026-08-01T10:00:{second:02d}.000Z"))
        second += 10
        for _ in range(5):
            when = f"2026-08-01T10:00:{second:02d}.000Z"
            lines.append(assistant_tool_line(
                f"a{second}", f"msg{second}", f"t{second}", "Edit",
                {"file_path": "/repo/f.py"}, when,
                input_tokens=30, output_tokens=25, cache_read=500_000,
            ))
            second += 1
    return write_transcript(claude_home / PROJECT_SLUG / "two-task.jsonl", lines)


def _install_fake_forest_judge(monkeypatch: pytest.MonkeyPatch):
    """Force the task-forest judge on and answer with a fixed two-task forest,
    so this one test never reaches the real LLM (``clean_api`` forces it off
    for every other test in this file)."""
    monkeypatch.setattr(classify, "llm_available", lambda: True)

    class _Completions:
        def create(self, **kwargs):
            content = json.dumps({
                "tasks": [{"id": "T1", "label": "Fix login bug"},
                         {"id": "T2", "label": "Optimize query"}],
                "assignments": ["T1", "T1", "T2", "T2"],
                "summary": "Fixed login and optimized queries",
            })
            message = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)


def test_task_switch_problem_round_trips_through_the_api(
    client: TestClient, claude_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(task_forest, "_CACHE_FILE", tmp_path / "task_forest.json")
    monkeypatch.setattr(task_forest, "_cache", None)
    _install_fake_forest_judge(monkeypatch)
    _two_task_session(claude_home)

    data = client.get("/api/sessions/two-task").json()

    problem = next(p for p in data["problems"] if p["id"] == "task-switch")
    assert [t["id"] for t in problem["data"]["tasks"]] == ["T1", "T2"]
    assert data["max_severity"] in ("medium", "high")


def test_pages_version_their_asset_urls(client: TestClient, claude_home: Path):
    """A cached ``app.js`` outlived the code it called and broke the page.

    ``Cache-Control: no-cache`` alone did not shift it — there is no build step,
    so the URL never changes and the browser was free to keep its copy. The
    version in the query string is what makes reuse impossible.
    """
    html = client.get("/").text

    assert 'src="/static/app.js?v=' in html
    assert 'href="/static/style.css?v=' in html
    assert 'src="/static/app.js"' not in html, "an unversioned URL slipped through"


def test_the_asset_version_changes_when_a_file_changes(
    client: TestClient, claude_home: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
):
    before = api._asset_version()

    web = tmp_path / "web"
    web.mkdir()
    for name in ("index.html", "app.js"):
        (web / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(api, "WEB_DIR", web)

    assert api._asset_version() != before


def test_pages_are_never_cached(client: TestClient, claude_home: Path):
    """The HTML carries the version string, so a stale page undoes the fix."""
    for route in ("/", "/session/anything"):
        assert client.get(route).headers["cache-control"] == "no-store", route


def test_static_files_are_always_revalidated(client: TestClient,
                                             claude_home: Path):
    """A cached ``app.js`` outlived the code it called and broke the page.

    There is no build step and no content hash in these filenames, so an edited
    file keeps its URL; without ``no-cache`` a browser may reuse the old one
    indefinitely.
    """
    for asset in ("/static/app.js", "/static/bar.js", "/static/style.css"):
        response = client.get(asset)
        assert response.status_code == 200, asset
        assert response.headers["cache-control"] == "no-cache", asset


def test_projects_counts_without_parsing(client: TestClient, claude_home: Path,
                                         monkeypatch: pytest.MonkeyPatch):
    _many_sessions(claude_home, 25)

    def forbidden(*args, **kwargs):
        raise AssertionError("/api/projects must not parse transcripts")

    monkeypatch.setattr(api.claude_code, "load_session", forbidden)
    projects = client.get("/api/projects").json()["projects"]

    assert projects[0]["count"] == 25


# ----------------------------------------------------------------------
# Session detail
# ----------------------------------------------------------------------

def test_session_detail_returns_the_full_shape(client: TestClient, claude_home: Path,
                                               simple_session: Path):
    body = client.get(f"/api/sessions/{simple_session.stem}").json()

    assert body["session_id"] == simple_session.stem
    assert body["title"] == "Fix the login bug"
    assert "blocks" in body and "compaction_points" in body


def test_unknown_session_is_a_404(client: TestClient, claude_home: Path):
    assert client.get("/api/sessions/nope").status_code == 404


def test_detail_serialises_every_field_the_bar_reads(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """bar.js indexes these directly; a rename here breaks rendering silently."""
    body = client.get(f"/api/sessions/{simple_session.stem}").json()

    assert body["blocks"], "expected the detail endpoint to build blocks"
    for field in ("kind", "label", "confidence", "message_count",
                  "duration_s", "tokens", "attributed_tokens", "inner_blocks",
                  "agent_id", "description"):
        assert field in body["blocks"][0], f"blocks[].{field} missing"
    assert "working" in body["blocks"][0]["tokens"]


def test_the_api_divides_the_context_window_across_the_blocks(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """The bar sizes by ``attributed_tokens``; it must add up to the header.

    Wired in ``_load_sessions``, so this also pins that a cached session comes
    back attributed rather than only a freshly parsed one.
    """
    body = client.get(f"/api/sessions/{simple_session.stem}").json()
    in_blocks = sum(b["attributed_tokens"] for b in body["blocks"])

    assert in_blocks == body["tokens"]["working"]


def test_the_session_header_agrees_with_the_bar_on_retrospective_cost(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """session.js's header stat reads ``attributed_cost`` off the same body."""
    body = client.get(f"/api/sessions/{simple_session.stem}").json()

    assert "attributed_cost" in body
    assert body["attributed_cost"] > 0
    assert body["attributed_cost"] == pytest.approx(
        sum(b["attributed_cost"] for b in body["blocks"]))


def test_attribution_survives_the_session_cache(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """A second request is served from ``_cache`` — it must not come back at 0."""
    first = client.get(f"/api/sessions/{simple_session.stem}").json()
    second = client.get(f"/api/sessions/{simple_session.stem}").json()

    assert [b["attributed_tokens"] for b in first["blocks"]] \
        == [b["attributed_tokens"] for b in second["blocks"]]
    assert sum(b["attributed_tokens"] for b in second["blocks"]) > 0


def test_every_block_kind_has_a_label(client: TestClient, claude_home: Path,
                                      simple_session: Path):
    """The legend renders from this map; a missing kind would be unlabelled."""
    from Final_app.ir.models import BLOCK_KINDS

    labels = client.get(f"/api/sessions/{simple_session.stem}").json()["kind_labels"]

    assert set(labels) == set(BLOCK_KINDS)


def test_detail_blocks_cover_the_sessions_events(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """No event may be dropped by classification or merging."""
    body = client.get(f"/api/sessions/{simple_session.stem}").json()
    counted = sum(b["message_count"] for b in body["blocks"])

    assert counted == body["message_count"]


def test_detail_reports_subagents(client: TestClient, claude_home: Path,
                                  session_with_subagent: Path):
    body = client.get(f"/api/sessions/{session_with_subagent.stem}").json()

    assert body["subagent_count"] == 1
    assert body["subagent_ids"] == ["a0b679820a32f88c4"]


# ----------------------------------------------------------------------
# Discovery and caching on refresh
# ----------------------------------------------------------------------

def test_a_new_session_appears_without_a_restart(client: TestClient, claude_home: Path,
                                                 simple_session: Path):
    """Refreshing the page must pick up sessions started since it loaded."""
    assert client.get("/api/sessions").json()["total"] == 1

    write_transcript(claude_home / PROJECT_SLUG / "brand-new.jsonl", [
        user_line("u1", "started later", "2026-08-01T12:00:00.000Z"),
    ])

    assert client.get("/api/sessions").json()["total"] == 2


def test_a_new_project_directory_is_discovered(client: TestClient, claude_home: Path,
                                               simple_session: Path):
    write_transcript(claude_home / "-Users-tester-Desktop-fresh" / "s.jsonl", [
        user_line("u1", "new project", "2026-08-01T12:00:00.000Z"),
    ])

    assert len(client.get("/api/projects").json()["projects"]) == 2


def test_an_appended_session_is_reparsed(client: TestClient, claude_home: Path):
    """A live session must not go stale behind the cache."""
    path = write_transcript(claude_home / PROJECT_SLUG / "live.jsonl", [
        user_line("u1", "first", "2026-08-01T10:00:00.000Z"),
    ])
    assert client.get("/api/sessions").json()["sessions"][0]["message_count"] == 1

    with path.open("a", encoding="utf-8") as handle:
        import json
        handle.write(json.dumps(
            user_line("u2", "second", "2026-08-01T10:05:00.000Z")) + "\n")

    assert client.get("/api/sessions").json()["sessions"][0]["message_count"] == 2


def test_an_untouched_session_is_served_from_cache(
    client: TestClient, claude_home: Path, simple_session: Path,
    monkeypatch: pytest.MonkeyPatch
):
    client.get("/api/sessions")  # warm

    def explode(*args, **kwargs):
        raise AssertionError("unchanged transcript should not be re-parsed")

    monkeypatch.setattr(claude_code, "load_session", explode)
    assert client.get("/api/sessions").json()["total"] == 1


def test_a_deleted_session_is_evicted_from_the_cache(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """Otherwise a long-running server holds every session it has ever seen."""
    client.get("/api/sessions")
    assert len(api._cache) == 1

    simple_session.unlink()

    assert client.get("/api/sessions").json()["total"] == 0
    assert len(api._cache) == 0


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/session/anything", "/static/app.js",
                                  "/session/anything/block/0",
                                  "/session/anything/problem/plan-mode",
                                  "/session/anything/agent/abc",
                                  "/session/anything/agent/abc/block/0"])
def test_pages_are_served(client: TestClient, claude_home: Path, path: str):
    assert client.get(path).status_code == 200


# ----------------------------------------------------------------------
# One block's steps
# ----------------------------------------------------------------------
# The page behind a click on the bar. It has to answer "what was read, run or
# written here" — the detail `Block.as_dict()` deliberately leaves out.

def _blocks_of(client: TestClient, session_id: str) -> list[dict]:
    return client.get(f"/api/sessions/{session_id}").json()["blocks"]


def test_a_block_reports_the_steps_behind_it(
    client: TestClient, claude_home: Path, simple_session: Path
):
    session_id = simple_session.stem
    blocks = _blocks_of(client, session_id)
    execute = next(index for index, block in enumerate(blocks)
                   if block["kind"] == "execute")

    body = client.get(f"/api/sessions/{session_id}/blocks/{execute}").json()

    assert body["index"] == execute
    assert body["block_count"] == len(blocks)
    # The fixture's execute block is one Bash call: `pytest -k login`.
    commands = [argument["value"]
                for step in body["steps"] if step["tool"]
                for argument in step["tool"]["arguments"]
                if argument["name"] == "command"]
    assert commands == ["pytest -k login"]


def test_a_block_page_reports_its_context_window_share(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """A regression test for a real bug: ``_block_payload`` is a manually
    built dict, not ``Block.as_dict()``, and it copied ``attributed_total``
    but not the newer ``context_tokens`` — silently serving ``None`` on this
    endpoint even though the field existed on the ``Block`` all along."""
    session_id = simple_session.stem
    blocks = _blocks_of(client, session_id)
    index = next(i for i, block in enumerate(blocks) if block["kind"] == "read")

    body = client.get(f"/api/sessions/{session_id}/blocks/{index}").json()

    assert body["context_tokens"] is not None
    assert body["context_tokens"] == blocks[index]["context_tokens"]


def test_a_read_block_names_the_file_it_read(
    client: TestClient, claude_home: Path, simple_session: Path
):
    session_id = simple_session.stem
    blocks = _blocks_of(client, session_id)
    read = next(index for index, block in enumerate(blocks)
                if block["kind"] == "read")

    body = client.get(f"/api/sessions/{session_id}/blocks/{read}").json()

    assert body["summary"]["files"] == ["/repo/login.py"]
    assert body["summary"]["tool_calls"] == 1


def test_a_block_out_of_range_is_a_404_saying_how_many_there_are(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """A link kept from yesterday must not silently show a different block."""
    session_id = simple_session.stem
    count = len(_blocks_of(client, session_id))

    response = client.get(f"/api/sessions/{session_id}/blocks/{count}")

    assert response.status_code == 404
    assert str(count) in response.json()["detail"]


def test_a_negative_block_index_is_a_404_not_the_last_block(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """Python would happily read -1 as "from the end"; the URL must not."""
    response = client.get(f"/api/sessions/{simple_session.stem}/blocks/-1")

    assert response.status_code == 404


def test_a_block_of_an_unknown_session_is_a_404(
    client: TestClient, claude_home: Path
):
    assert client.get("/api/sessions/nope/blocks/0").status_code == 404


def test_a_subagent_band_lists_the_agents_it_covers(
    client: TestClient, claude_home: Path, session_with_parallel_subagents: Path
):
    session_id = session_with_parallel_subagents.stem
    blocks = _blocks_of(client, session_id)
    band = next(index for index, block in enumerate(blocks)
                if block["kind"] == "subagent")

    body = client.get(f"/api/sessions/{session_id}/blocks/{band}").json()

    assert [agent["description"] for agent in body["agents"]] == [
        "Map the adapters", "Map the web layer"]
    # Each agent summarises its own work, not the band's pooled list. The spawn
    # that launched it leads its events, so `Agent` appears alongside the
    # child's own tools — that is what anchors the agent at its delegation.
    assert [agent["summary"]["tools"] for agent in body["agents"]] == [
        {"Agent": 1, "Grep": 1}, {"Agent": 1, "Read": 1}]


def test_a_band_lists_only_the_calls_that_delegated(
    client: TestClient, claude_home: Path, session_with_parallel_subagents: Path
):
    """Not the children's own steps — those belong to each agent's own page.

    The band's events include every child event, because that is what makes
    its size on the bar honest. Listing them here repeated the whole subtree
    one level up.
    """
    session_id = session_with_parallel_subagents.stem
    blocks = _blocks_of(client, session_id)
    band = next(index for index, block in enumerate(blocks)
                if block["kind"] == "subagent")

    body = client.get(f"/api/sessions/{session_id}/blocks/{band}").json()

    assert [step["tool"]["name"] for step in body["steps"]] == ["Agent", "Agent"]
    # The summary above the list still covers the whole band, children included.
    assert body["summary"]["tools"] == {"Agent": 2, "Grep": 1, "Read": 1}


def test_an_ordinary_block_still_lists_every_step(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """The spawn-only filter must apply to delegation bands and nothing else."""
    session_id = simple_session.stem
    blocks = _blocks_of(client, session_id)
    read = next(index for index, block in enumerate(blocks)
                if block["kind"] == "read")

    body = client.get(f"/api/sessions/{session_id}/blocks/{read}").json()

    assert len(body["steps"]) == body["summary"]["steps"]


def test_a_band_lists_its_agents_without_shipping_their_steps(
    client: TestClient, claude_home: Path, session_with_parallel_subagents: Path
):
    """Each agent links to its own page; inlining every subtree here would
    make one request carry the whole session's delegated work."""
    session_id = session_with_parallel_subagents.stem
    blocks = _blocks_of(client, session_id)
    band = next(index for index, block in enumerate(blocks)
                if block["kind"] == "subagent")

    body = client.get(f"/api/sessions/{session_id}/blocks/{band}").json()

    assert all("steps" not in agent for agent in body["agents"])
    assert [agent["block_count"] for agent in body["agents"]] == [1, 1]


# ----------------------------------------------------------------------
# One subagent's own bar
# ----------------------------------------------------------------------
# A subagent is a session in miniature: its own blocks, its own bar, its own
# click-through. These endpoints are what make that page possible.

def _first_agent_id(client: TestClient, session_id: str) -> str:
    blocks = _blocks_of(client, session_id)
    band = next(block for block in blocks if block["kind"] == "subagent")
    return band["agents"][0]["agent_id"]


def test_a_subagent_serves_its_own_blocks_for_a_bar_of_its_own(
    client: TestClient, claude_home: Path, session_with_subagent: Path
):
    session_id = session_with_subagent.stem
    agent_id = _first_agent_id(client, session_id)

    body = client.get(f"/api/sessions/{session_id}/agents/{agent_id}").json()

    assert body["agent_id"] == agent_id
    assert body["description"] == "Find all TODOs"
    # The task prompt the parent handed the child leads the agent's bar as
    # `coordination`, never `user_chat` — no human is in a subagent's loop.
    # Then Grep and Read merge into one `read`.
    assert [block["kind"] for block in body["blocks"]] == ["coordination", "read"]
    assert body["blocks"][1]["message_count"] == 2


def test_a_subagent_detail_reports_its_own_retrospective_cost(
    client: TestClient, claude_home: Path, session_with_subagent: Path
):
    """agent.js's header stat reads this field directly off the subagent."""
    session_id = session_with_subagent.stem
    agent_id = _first_agent_id(client, session_id)

    body = client.get(f"/api/sessions/{session_id}/agents/{agent_id}").json()

    assert "attributed_cost" in body
    assert body["attributed_cost"] > 0
    assert body["attributed_cost"] == pytest.approx(
        sum(b["attributed_cost"] for b in body["blocks"]))


def test_a_subagent_detail_reports_its_own_context_window(
    client: TestClient, claude_home: Path, session_with_subagent: Path
):
    """agent.js's "Context window" stat reads this field directly off the
    subagent — a regression test for a real bug: the endpoint's manually
    built dict omitted ``context_tokens`` entirely (present on ``Block`` but
    never copied into the response), so the field came back ``None``."""
    session_id = session_with_subagent.stem
    agent_id = _first_agent_id(client, session_id)

    body = client.get(f"/api/sessions/{session_id}/agents/{agent_id}").json()

    assert body["context_tokens"] is not None
    assert body["context_tokens"] > 0
    assert body["context_tokens"] == sum(b["context_tokens"] for b in body["blocks"])


def test_each_parallel_subagent_serves_only_its_own_work(
    client: TestClient, claude_home: Path, session_with_parallel_subagents: Path
):
    """Two agents under one band must not see each other's blocks."""
    session_id = session_with_parallel_subagents.stem
    blocks = _blocks_of(client, session_id)
    band = next(block for block in blocks if block["kind"] == "subagent")
    tools_by_agent = {}

    for agent in band["agents"]:
        body = client.get(
            f"/api/sessions/{session_id}/agents/{agent['agent_id']}").json()
        tools_by_agent[body["description"]] = body["summary"]["tools"]

    assert tools_by_agent == {
        "Map the adapters": {"Agent": 1, "Grep": 1},
        "Map the web layer": {"Agent": 1, "Read": 1},
    }


def test_an_unknown_subagent_is_a_404(
    client: TestClient, claude_home: Path, session_with_subagent: Path
):
    response = client.get(
        f"/api/sessions/{session_with_subagent.stem}/agents/nope")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_a_block_of_a_subagents_bar_reports_that_agent(
    client: TestClient, claude_home: Path, session_with_subagent: Path
):
    session_id = session_with_subagent.stem
    agent_id = _first_agent_id(client, session_id)

    # Block 1 is the agent's work; block 0 is the task prompt it was handed.
    body = client.get(
        f"/api/sessions/{session_id}/agents/{agent_id}/blocks/1").json()

    assert body["agent_id"] == agent_id
    assert body["agent_description"] == "Find all TODOs"
    # Counted within the agent's bar, not the session's.
    assert body["block_count"] == 2
    assert body["summary"]["tools"] == {"Grep": 1, "Read": 1}


def test_a_subagent_block_out_of_range_is_a_404(
    client: TestClient, claude_home: Path, session_with_subagent: Path
):
    session_id = session_with_subagent.stem
    agent_id = _first_agent_id(client, session_id)

    response = client.get(
        f"/api/sessions/{session_id}/agents/{agent_id}/blocks/9")

    assert response.status_code == 404
    assert agent_id in response.json()["detail"]


def test_a_session_block_reports_no_agent_of_its_own(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """So the page can tell "block of a session" from "block of a subagent"."""
    body = client.get(f"/api/sessions/{simple_session.stem}/blocks/0").json()

    assert body["agent_id"] is None


def test_a_non_subagent_block_lists_no_agents(
    client: TestClient, claude_home: Path, simple_session: Path
):
    body = client.get(f"/api/sessions/{simple_session.stem}/blocks/0").json()

    assert body["agents"] == []


def test_a_blocks_steps_carry_the_api_call_they_were_billed_to(
    client: TestClient, claude_home: Path, simple_session: Path
):
    """The cost unit is the message, not the step — the page has to show it.

    Only assistant-side steps carry one: a user prompt is a line the human
    typed, not an API response, so it has no ``message.id`` to report.
    """
    session_id = simple_session.stem
    blocks = _blocks_of(client, session_id)
    read = next(index for index, block in enumerate(blocks)
                if block["kind"] == "read")

    body = client.get(f"/api/sessions/{session_id}/blocks/{read}").json()

    assert all(step["message_id"] for step in body["steps"] if step["tool"])
    assert body["summary"]["api_calls"] >= 1
