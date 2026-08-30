"""FastAPI backend for the TraceLens dashboard.

Run it with:

    uvicorn Final_app.api.app:app --reload --port 8765

Endpoints
    GET  /                        the dashboard (two tabs)
    GET  /session/{id}            one session's page
    GET  /api/health              LLM configuration diagnostics
    GET  /api/llm-check           live probe: does the LLM actually answer?
    GET  /api/projects            project slugs + session counts, for the filter
    GET  /api/sessions            session list, newest first
    GET  /api/problems            detected problems, newest-session-first
    GET  /api/sessions/{id}       one full session
    GET  /api/sessions/{id}/blocks/{n}   one block's steps, for its page
    GET  /api/sessions/{id}/agents/{aid} one subagent's own blocks
    GET  /api/sessions/{id}/agents/{aid}/blocks/{n}   one of those blocks
    GET  /session/{id}/block/{n}         that block's page
    GET  /session/{id}/problem/{pid}     one detected problem's own page
    GET  /session/{id}/agent/{aid}       one subagent's own bar
    GET  /session/{id}/agent/{aid}/block/{n}
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from Final_app.adapters import claude_code
from Final_app.analysis.attribution import attribute
from Final_app.analysis.blocks import KIND_LABELS, build_blocks
from Final_app.analysis.problems import detect_problems
from Final_app.analysis.steps import block_steps, step_summary
from Final_app.config import llm_diagnostics, probe_llm
from Final_app.ir.models import Block, Session

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="TraceLens", docs_url="/api/docs")

# ----------------------------------------------------------------------
# Session cache
# ----------------------------------------------------------------------
# Cache by (path, mtime, size) so an appended-to session reparses automatically
# and an untouched one never does.
#
# The list request parses **only the page it returns**. Parsing everything cost
# ~0.2s for the 39.7 MB of transcripts on this machine, which is fine — but it
# scales with total bytes on disk, and a user with a few gigabytes of history
# waited many seconds for a page showing twenty rows. Ordering therefore has to
# come from something free.

_cache: dict[str, tuple[tuple, Session]] = {}

# Rows per page. The list is a "what did I do lately" view, so the first screen
# is what matters and the rest is opt-in.
PAGE_SIZE = 20

# ``/api/problems`` scans sessions through a thread pool: each detector's work
# is a network call to the judge LLM, so the GIL releases for most of it and
# threads genuinely overlap. This matters a lot with a reasoning model on the
# other end (e.g. ``openai/gpt-5.6-luna``) — its per-call latency is seconds,
# not milliseconds, and scanning a page of sessions serially made that latency
# additive across the whole page instead of paid once.
PROBLEM_SCAN_WORKERS = 8


def _stamp(path: Path) -> tuple:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def _candidates(project: Optional[str] = None) -> list[tuple[str, Path]]:
    """``(slug, path)`` newest-first, **without opening a single transcript**.

    Ordered by file mtime, which is when the last message was appended — the
    same thing ``last_ts`` measures, and available from ``stat``. The two can
    disagree if a transcript is touched without being written, so the page that
    is actually parsed is re-sorted by real ``last_ts`` before it is returned.

    Re-globs on every call, so a session started after the page was opened
    shows up on the next refresh with no restart.
    """
    found = claude_code.discover_sessions()
    if project:
        found = [(slug, path) for slug, path in found if slug == project]

    # Drop transcripts that have gone away, so a long-running server does not
    # hold every session it has ever seen. Done against the unfiltered set.
    if not project:
        live = {str(path) for _, path in found}
        for stale in _cache.keys() - live:
            del _cache[stale]

    found.sort(key=lambda item: _stamp(item[1])[0], reverse=True)
    return found


def _load_one(project_slug: str, path: Path) -> Optional[Session]:
    """One session, from cache when the file has not moved. None if unusable."""
    key = str(path)
    stamp = _stamp(path)
    hit = _cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]

    try:
        session = claude_code.load_session(project_slug, path)
    except Exception:
        # One broken transcript must not empty the whole dashboard.
        return None
    if session is None:
        return None

    # Spread the billed context window over the Events that caused it. Done at
    # load rather than in ``session_detail`` because it is pure arithmetic over
    # the already-parsed stream — no LLM, no network — and caching it with the
    # session means a re-opened page pays nothing.
    session.overhead_tokens = attribute(session)
    _cache[key] = (stamp, session)
    return session


def _load_page(candidates: list[tuple[str, Path]], offset: int,
               limit: int) -> list[Session]:
    """Parse one window of candidates, newest-first by real ``last_ts``.

    A transcript that fails to parse is skipped, which makes the page shorter
    than ``limit`` rather than pulling an extra one forward — keeping the
    window aligned with ``offset`` matters more than filling every row, or
    "load more" would skip sessions.
    """
    sessions = [
        session
        for project_slug, path in candidates[offset:offset + limit]
        if (session := _load_one(project_slug, path)) is not None
    ]
    sessions.sort(key=lambda s: (s.last_ts is None, s.last_ts), reverse=True)
    return sessions


def _find(session_id: str) -> Session:
    """One session by id, parsing only that transcript.

    The id is the file stem, so the transcript is found by name rather than by
    parsing every session and comparing.
    """
    for project_slug, path in claude_code.discover_sessions():
        if path.stem != session_id:
            continue
        session = _load_one(project_slug, path)
        if session is not None:
            return session
        break
    raise HTTPException(status_code=404, detail=f"no session {session_id}")


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


@app.get("/api/llm-check")
def llm_check(force: bool = Query(False, description="ignore the cached result")) -> dict:
    """Does the LLM actually answer, right now.

    Separate from ``/api/health`` so the page paints immediately: health is a
    config read, this makes a real call and can take seconds to fail. The
    client renders the page first and fills this in when it lands.
    """
    return probe_llm(force=force)


@app.get("/api/projects")
def projects() -> dict:
    # Counted from filenames, never by parsing: this endpoint only needs to
    # know how many transcripts each project has.
    counts: dict[str, int] = {}
    for project_slug, _ in claude_code.discover_sessions():
        counts[project_slug] = counts.get(project_slug, 0) + 1
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
    offset: int = Query(0, ge=0, description="rows to skip, for 'load more'"),
    limit: int = Query(PAGE_SIZE, ge=1, le=200, description="rows to return"),
) -> dict:
    """One page of sessions, newest-first.

    Only this page's transcripts are opened. ``total`` counts every candidate
    on disk, so the client knows there is more without anything being parsed to
    find out.

    ``severity`` is accepted and threaded through now, but this endpoint never
    builds blocks or runs a detector — every session it returns reports
    ``none`` unless its detail page has already been opened this server run
    (see ``session_detail``, which is where detection actually happens). The
    Problems tab gets its real answer from ``/api/problems`` instead, which
    does run detection. It is applied *within* the page rather than before it,
    since filtering across all sessions would mean parsing all of them, which
    is the cost this endpoint exists to avoid.
    """
    candidates = _candidates(project)
    items = _load_page(candidates, offset, limit)

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
        "total": len(candidates),
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < len(candidates),
    }


@app.get("/api/problems")
def problems(
    project: Optional[str] = Query(None, description="project slug; omit for all"),
    offset: int = Query(0, ge=0, description="sessions to skip, for 'load more'"),
    limit: int = Query(PAGE_SIZE, ge=1, le=200, description="sessions to scan"),
) -> dict:
    """One page of detected problems, scanning sessions newest-first.

    Unlike ``/api/sessions``, this always builds each scanned session's blocks
    and runs its detectors — there is no problem list to page over without
    doing that work, and this is the endpoint that actually does it (the
    Problems tab's real source of truth; ``/api/sessions``' own severity
    filter is only ever as fresh as whichever sessions this — or a detail
    page — has already scanned).

    ``limit``/``offset`` page over SESSIONS scanned, not problems found, so a
    stretch of clean sessions can come back as a short or empty page with
    ``has_more`` still true — the client should keep clicking "load more"
    rather than reading a short page as the end of the list.

    Sessions are scanned through ``PROBLEM_SCAN_WORKERS`` threads, since each
    session's detectors are dominated by a network call to the judge LLM
    rather than local computation.
    """
    candidates = _candidates(project)
    sessions_scanned = _load_page(candidates, offset, limit)

    def scan(session: Session) -> Session:
        if not session.blocks:
            session.blocks = build_blocks(session)
        if not session.problems:
            session.problems = detect_problems(session)
        return session

    with ThreadPoolExecutor(max_workers=PROBLEM_SCAN_WORKERS) as pool:
        sessions_scanned = list(pool.map(scan, sessions_scanned))

    rows = []
    for session in sessions_scanned:
        # The same summary shape `/api/sessions` rows use (title, tokens,
        # tool_call_count, duration_s, subagent_count, ...), so a problem row
        # can show the session it belongs to without a second request.
        summary = session.summary_dict()
        for problem in session.problems:
            rows.append(dict(
                summary,
                project_label=claude_code.unslug_project(session.project),
                problem=problem.as_dict(),
            ))

    return {
        "problems": rows,
        "total_sessions": len(candidates),
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < len(candidates),
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
    if not session.problems:
        # Detectors reason about blocks, so this runs after they're built.
        # Guarded like blocks: a cached Session only runs detection once.
        session.problems = detect_problems(session)
    data = session.as_dict()
    data["project_label"] = claude_code.unslug_project(session.project)
    data["kind_labels"] = KIND_LABELS
    return data


@app.get("/api/sessions/{session_id}/blocks/{index}")
def block_detail(session_id: str, index: int) -> dict:
    """One block's steps: what was read, run or written, and what came back.

    Addressed by position in the session's block list rather than by an id of
    its own. Blocks are derived, not stored — re-running the merge on an
    appended-to session can change how many there are — so any id would be just
    as unstable as the index while pretending otherwise. The response repeats
    ``block_count`` so a stale link can say "block 47 of 30" instead of
    silently showing the wrong stretch of work.
    """
    session = _find(session_id)
    if not session.blocks:
        session.blocks = build_blocks(session)

    if index < 0 or index >= len(session.blocks):
        raise HTTPException(
            status_code=404,
            detail=f"block {index} of {len(session.blocks)}"
                   f" — this session has no such block",
        )

    return _block_payload(session, session.blocks[index], index,
                          len(session.blocks))


def _block_payload(session: Session, block: Block, index: int, block_count: int,
                   agent_id: Optional[str] = None,
                   agent_description: str = "") -> dict:
    """One block's page payload, whether it came from a session or a subagent.

    Both bars are the same thing at different depths, so they answer with the
    same shape and the page needs no second renderer. ``agent_id`` is set only
    for a block of a subagent's own bar, and is what its links point back to.
    """
    steps = block_steps(block)

    # A delegation band's events are its spawns *plus* every event of every
    # child, because that is what makes its size on the bar honest. Listing all
    # of them here would repeat, one level up, exactly the timeline each
    # agent's own page already shows — so the band lists only the calls that
    # did the delegating. The summary above it still covers the whole band.
    own_steps = ([step for step in steps
                  if step["tool"]
                  and step["tool"]["name"] in claude_code.SUBAGENT_TOOLS]
                 if block.agents else steps)

    return {
        "session_id": session.session_id,
        "session_title": session.title,
        "project_label": claude_code.unslug_project(session.project),
        "agent_id": agent_id,
        "agent_description": agent_description,
        "index": index,
        "block_count": block_count,
        "kind": block.kind,
        "kind_label": KIND_LABELS.get(block.kind, block.kind),
        "label": block.label,
        "description": block.description,
        "confidence": block.confidence,
        "t_start": block.t_start.isoformat() if block.t_start else None,
        "t_end": block.t_end.isoformat() if block.t_end else None,
        "duration_s": block.duration_s,
        "tokens": block.tokens.as_dict(),
        "attributed_tokens": block.attributed_tokens,
        "attributed_cache_read": block.attributed_cache_read,
        "attributed_total": block.attributed_total,
        "context_tokens": block.context_tokens,
        "message_count": block.message_count,
        "summary": step_summary(steps),
        "steps": own_steps,
        # One entry per subagent under this band; empty for every other kind.
        # Summaries only — each links to its own page for the detail.
        "agents": [_agent_detail(agent, summary_only=True)
                   for agent in block.agents],
    }


def _agent_detail(agent: Block, summary_only: bool = False) -> dict:
    """One subagent inside a delegation band.

    ``summary_only`` is what the band's own page needs: it lists the agents and
    links to them, and shipping every agent's full step list would make one
    request carry the whole subtree.
    """
    steps = block_steps(agent)
    detail = {
        "agent_id": agent.agent_id,
        "description": agent.description,
        "label": agent.label,
        "t_start": agent.t_start.isoformat() if agent.t_start else None,
        "t_end": agent.t_end.isoformat() if agent.t_end else None,
        "duration_s": agent.duration_s,
        "tokens": agent.tokens.as_dict(),
        "attributed_tokens": agent.attributed_tokens,
        "attributed_cache_read": agent.attributed_cache_read,
        "attributed_total": agent.attributed_total,
        "attributed_cost": agent.attributed_cost,
        "context_tokens": agent.context_tokens,
        "message_count": agent.message_count,
        "summary": step_summary(steps),
        "block_count": len(agent.inner_blocks),
    }
    if not summary_only:
        detail["steps"] = steps
        # The agent's own timeline, in the shape the bar renders. This is what
        # makes a subagent page a session page in miniature.
        detail["blocks"] = [child.as_dict() for child in agent.inner_blocks]
    return detail


def _find_agent(session: Session, agent_id: str) -> Block:
    """The subagent with this id, wherever its band sits in the session."""
    if not session.blocks:
        session.blocks = build_blocks(session)
    for block in session.blocks:
        for agent in block.agents:
            if agent.agent_id == agent_id:
                return agent
    raise HTTPException(status_code=404,
                        detail=f"no subagent {agent_id} in this session")


@app.get("/api/sessions/{session_id}/agents/{agent_id}")
def agent_detail(session_id: str, agent_id: str) -> dict:
    """One subagent's own timeline — its blocks, as a bar of its own.

    A subagent is a whole session in miniature: it was given a task, it worked
    through it, and it has its own stretches of reading and running. Flattening
    that into its parent's bar is what made delegated work unreadable, so it
    gets its own page and its own bar instead.
    """
    session = _find(session_id)
    agent = _find_agent(session, agent_id)

    detail = _agent_detail(agent)
    detail["session_id"] = session.session_id
    detail["session_title"] = session.title
    detail["kind_labels"] = KIND_LABELS
    return detail


@app.get("/api/sessions/{session_id}/agents/{agent_id}/blocks/{index}")
def agent_block_detail(session_id: str, agent_id: str, index: int) -> dict:
    """One block of one subagent's bar, addressed within that agent."""
    session = _find(session_id)
    agent = _find_agent(session, agent_id)

    if index < 0 or index >= len(agent.inner_blocks):
        raise HTTPException(
            status_code=404,
            detail=f"block {index} of {len(agent.inner_blocks)}"
                   f" — subagent {agent_id} has no such block",
        )

    return _block_payload(
        session, agent.inner_blocks[index], index, len(agent.inner_blocks),
        agent_id=agent_id, agent_description=agent.description,
    )


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
# Asset URLs carry a content version, so a browser physically cannot reuse a
# stale script: the URL changes when the file does.
#
# ``Cache-Control: no-cache`` on /static was not enough on its own. There is no
# build step here, so ``app.js`` keeps its URL forever, and a copy Safari had
# already stored survived both the new header and a reload — it kept calling
# ``trackPending`` after the summaries feature was deleted and reported
# "Failed to load sessions" against a healthy backend. Versioning the URL is
# the only fix that does not depend on the browser choosing to revalidate.

_ASSET_HREF = re.compile(r'(?P<attr>src|href)="(?P<url>/static/[^"?]+)"')


def _asset_version() -> str:
    """A short digest of every file in ``web/``.

    Content-based rather than a server start time, so restarting does not
    needlessly bust the cache and an edit always does. These are a handful of
    small files, so hashing them per page request is cheaper than the round
    trips it saves.
    """
    digest = hashlib.sha256()
    for path in sorted(WEB_DIR.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(str(_stamp(path)).encode())
    return digest.hexdigest()[:12]


def _page(name: str) -> HTMLResponse:
    """One of the two static pages, with versioned asset URLs."""
    version = _asset_version()
    html = (WEB_DIR / name).read_text(encoding="utf-8")
    html = _ASSET_HREF.sub(
        lambda m: f'{m["attr"]}="{m["url"]}?v={version}"', html)
    # The HTML itself must never be cached either, or it would keep pointing at
    # the old version string and undo all of this.
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/")
def index() -> HTMLResponse:
    return _page("index.html")


@app.get("/session/{session_id}")
def session_page(session_id: str) -> HTMLResponse:
    # The id is read client-side from the URL; one static page serves them all.
    return _page("session.html")


@app.get("/session/{session_id}/block/{index}")
def block_page(session_id: str, index: int) -> HTMLResponse:
    # Opened in its own tab from the bar, so the session stays put behind it.
    return _page("block.html")


@app.get("/session/{session_id}/problem/{problem_id}")
def problem_page(session_id: str, problem_id: str) -> HTMLResponse:
    """One detected problem's own page — the Problems tab's link target.

    Draws the session's plain bar beside the same bar with that problem's
    split cut into it (see ``web/plan_mode.js``); ``session.html`` itself
    shows only the plain bar, so this is the one place that comparison lives.
    """
    return _page("problem.html")


@app.get("/session/{session_id}/agent/{agent_id}")
def agent_page(session_id: str, agent_id: str) -> HTMLResponse:
    """One subagent's own bar — a session page one level down."""
    return _page("agent.html")


@app.get("/session/{session_id}/agent/{agent_id}/block/{index}")
def agent_block_page(session_id: str, agent_id: str, index: int) -> HTMLResponse:
    # Same page as a session block; block.js reads the agent from the URL.
    return _page("block.html")


class _NoCacheStatic(StaticFiles):
    """Serve ``web/`` with revalidation forced on every request.

    Without this, FastAPI sends only ``ETag`` and ``Last-Modified`` and leaves
    freshness to the browser's heuristics — Safari happily reuses a cached
    ``app.js`` without asking. There is no build step and no content hash in
    the filename here, so an edited file keeps its URL and a stale copy sticks:
    after the summaries were removed, a cached ``app.js`` kept calling
    ``trackPending`` and blew up on the ``#pending`` element that no longer
    exists, showing "Failed to load sessions" on a working backend.

    ``no-cache`` means "revalidate", not "do not store" — the ETag still turns
    an unchanged file into a 304, so this costs a round trip, not a re-download.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _NoCacheStatic(directory=WEB_DIR), name="static")
