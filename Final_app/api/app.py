"""FastAPI backend for the TraceLens dashboard.

Run it with:

    uvicorn Final_app.api.app:app --reload --port 8765

Endpoints
    GET  /                        the dashboard (two tabs)
    GET  /session/{id}            one session's page
    GET  /api/health              LLM configuration diagnostics
    GET  /api/projects            project slugs + session counts, for the filter
    GET  /api/sessions            session list; missing summaries fill in async
    GET  /api/sessions/{id}       one full session
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from Final_app.adapters import claude_code
from Final_app.analysis.blocks import KIND_LABELS, build_blocks
from Final_app.config import llm_diagnostics
from Final_app.ir.models import Session
from Final_app.judge import summary as judge

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="TraceLens", docs_url="/api/docs")

# ----------------------------------------------------------------------
# Session cache
# ----------------------------------------------------------------------
# Parsing every transcript on each request is wasteful and makes the list page
# feel slow. Cache by (path, mtime, size) so an appended-to session reparses
# automatically and an untouched one never does.

_cache: dict[str, tuple[tuple, Session]] = {}


def _stamp(path: Path) -> tuple:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def _load_sessions() -> list[Session]:
    """Every session on disk, newest-first.

    Re-globs the projects directory on every call, so a session started after
    the page was opened shows up on the next refresh with no restart. A session
    that is still being appended to has a changed mtime and is re-parsed; an
    untouched one is served straight from cache.
    """
    sessions: list[Session] = []
    live: set[str] = set()

    for project_slug, path in claude_code.discover_sessions():
        key = str(path)
        live.add(key)
        stamp = _stamp(path)
        hit = _cache.get(key)
        if hit is not None and hit[0] == stamp:
            sessions.append(hit[1])
            continue
        try:
            session = claude_code.load_session(project_slug, path)
        except Exception:
            # One broken transcript must not empty the whole dashboard.
            continue
        if session is None:
            continue
        session.summary = judge.cached_summary(session.user_prompts)
        _cache[key] = (stamp, session)
        sessions.append(session)

    # Drop transcripts that have gone away, so a long-running server does not
    # hold every session it has ever seen.
    for stale in _cache.keys() - live:
        del _cache[stale]

    sessions.sort(key=lambda s: (s.last_ts is None, s.last_ts), reverse=True)
    return sessions


def _find(session_id: str) -> Session:
    for session in _load_sessions():
        if session.session_id == session_id:
            return session
    raise HTTPException(status_code=404, detail=f"no session {session_id}")


# ----------------------------------------------------------------------
# Background summarisation
# ----------------------------------------------------------------------
# Summaries fill themselves in. The list request never waits on the LLM — it
# returns whatever is cached and reports how many are still pending, and the
# page quietly re-polls until that reaches zero.

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="summarize")
_inflight: set[str] = set()
_inflight_lock = threading.Lock()


def _summarize_one(session: Session) -> None:
    try:
        text = judge.summarize(session.user_prompts)
        if text:
            session.summary = text
    except Exception:
        # A failed summary is not worth breaking the page over; the row simply
        # keeps showing the Claude Code title until the next attempt.
        pass
    finally:
        with _inflight_lock:
            _inflight.discard(session.session_id)


def _kick_off_summaries(sessions: list[Session]) -> int:
    """Queue missing summaries. Returns how many are outstanding."""
    if not judge.llm_available():
        return 0

    pending = 0
    for session in sessions:
        if session.summary or not session.user_prompts:
            continue
        pending += 1
        with _inflight_lock:
            if session.session_id in _inflight:
                continue
            _inflight.add(session.session_id)
        _pool.submit(_summarize_one, session)
    return pending


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "projects_root": str(claude_code.projects_root()),
        "llm": llm_diagnostics(),
    }


@app.get("/api/projects")
def projects() -> dict:
    counts: dict[str, int] = {}
    for session in _load_sessions():
        counts[session.project] = counts.get(session.project, 0) + 1
    items = [
        {
            "slug": slug,
            "label": claude_code.unslug_project(slug),
            "count": count,
        }
        for slug, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return {"projects": items}


@app.get("/api/sessions")
def sessions(
    project: Optional[str] = Query(None, description="project slug; omit for all"),
    severity: Optional[str] = Query(None, description="minimum problem severity"),
) -> dict:
    """Sessions newest-first.

    ``severity`` is accepted and threaded through now so the UI filter is real,
    but every session currently reports ``none`` — problem detection lands in a
    later step.
    """
    items = _load_sessions()

    # Fill in whatever is missing, in the background, for every session — not
    # just the filtered view, so switching projects never restarts the work.
    pending = _kick_off_summaries(items)

    if project:
        items = [s for s in items if s.project == project]

    if severity and severity != "any":
        order = ["none", "info", "low", "medium", "high"]
        try:
            floor = order.index(severity)
        except ValueError:
            floor = 0
        items = [s for s in items if order.index(s.max_severity) >= floor]

    return {
        "sessions": [
            dict(s.summary_dict(), project_label=claude_code.unslug_project(s.project))
            for s in items
        ],
        "total": len(items),
        # >0 means summaries are still arriving; the client re-polls until 0.
        "pending_summaries": pending,
    }


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict:
    """One session, with its bar blocks built.

    Blocks are computed here rather than in ``_load_sessions`` because the list
    page never needs them and classification can call the judge.
    """
    session = _find(session_id)
    if not session.blocks:
        session.blocks = build_blocks(session)
    data = session.as_dict()
    data["project_label"] = claude_code.unslug_project(session.project)
    data["kind_labels"] = KIND_LABELS
    return data


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/session/{session_id}")
def session_page(session_id: str) -> FileResponse:
    # The id is read client-side from the URL; one static page serves them all.
    return FileResponse(WEB_DIR / "session.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
