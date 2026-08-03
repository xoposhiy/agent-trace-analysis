"""API tests: endpoint shapes, filtering, and cache behaviour.

Background summarisation is disabled throughout — these tests assert on
routing and caching, never on the LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from Final_app.adapters import claude_code
from Final_app.api import app as api

from .conftest import PROJECT_SLUG, user_line, write_transcript


@pytest.fixture(autouse=True)
def clean_api(monkeypatch: pytest.MonkeyPatch):
    """Empty the module-level parse cache and silence the judge.

    ``_cache`` persists for the process lifetime, so without this one test's
    sessions show up in the next.
    """
    api._cache.clear()
    monkeypatch.setattr(api.judge, "llm_available", lambda: False)
    monkeypatch.setattr(api.judge, "cached_summary", lambda prompts: "")
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


def test_severity_filter_accepts_any(client: TestClient, claude_home: Path,
                                     simple_session: Path):
    """Every session reports ``none`` until problem detection lands."""
    assert client.get("/api/sessions?severity=any").json()["total"] == 1
    assert client.get("/api/sessions?severity=high").json()["total"] == 0


def test_pending_summaries_is_zero_when_the_llm_is_off(
    client: TestClient, claude_home: Path, simple_session: Path
):
    assert client.get("/api/sessions").json()["pending_summaries"] == 0


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
                  "duration_s", "tokens", "inner_blocks",
                  "agent_id", "description"):
        assert field in body["blocks"][0], f"blocks[].{field} missing"
    assert "working" in body["blocks"][0]["tokens"]


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

@pytest.mark.parametrize("path", ["/", "/session/anything", "/static/app.js"])
def test_pages_are_served(client: TestClient, claude_home: Path, path: str):
    assert client.get(path).status_code == 200
