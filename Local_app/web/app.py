"""
web/app.py — the local FastAPI dashboard for the Claude Split Advisor.

Everything runs on localhost against your own transcripts; nothing is uploaded.

Endpoints
---------
  GET  /                  -> the single-page dashboard (static HTML/JS/CSS)
  GET  /api/sessions      -> whatever is already CACHED (instant, no analysis/LLM)
  POST /api/refresh       -> INCREMENTAL analysis: analyse only new/changed
                             sessions, reuse the cache for the rest, return stats
  GET  /api/session/{id}  -> one session's full analysis dict

The heavy lifting (discovery + the incremental JSON cache) lives in
`analysis_api.py`; this file is only the HTTP layer. Because `/api/sessions` reads
the cache directly, opening the page is always fast — the (possibly slow, LLM-bound)
work happens only when you press Refresh, and even then only for sessions you
created or changed since last time.
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analysis_api as api
import split_advisor as sa

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

APP_CONFIG = {
    "source": sa.DEFAULT_SOURCE,
    "projects_dir": sa.DEFAULT_PROJECTS_DIR,
    "project_filter": None,
    "dataset_repo": sa.DEFAULT_SWECHAT_REPO,
    "dataset_split": sa.DEFAULT_SWECHAT_SPLIT,
}

app = FastAPI(title="Claude Split Advisor")


# ======================================================================
# API endpoints
# ======================================================================

def _cached_sessions():
    """All analysis dicts currently in the cache, biggest as-is cost first."""
    cache = api.load_cache(cache_path=_cache_path())
    rows = [entry["analysis"] for entry in cache["sessions"].values()
            if isinstance(entry, dict) and entry.get("analysis")]
    rows.sort(key=lambda a: a.get("as_is_cost", 0), reverse=True)
    return rows


def _cache_path():
    return api.cache_path_for_source(APP_CONFIG["source"], APP_CONFIG["dataset_repo"],
                                     APP_CONFIG["dataset_split"])


@app.get("/api/sessions")
def get_sessions():
    """Serve the cached analyses instantly (no analysis, no LLM). Empty on first
    ever run until the user presses Refresh."""
    rows = _cached_sessions()
    return {"sessions": rows, "stats": {"cached": len(rows)}}


@app.post("/api/refresh")
def refresh(use_llm: bool = True, force: bool = False, source: str = "local",
            projects_dir: Optional[str] = None, project_filter: Optional[str] = None,
            dataset_repo: Optional[str] = None, dataset_split: Optional[str] = None):
    """Run the incremental analysis and return the updated sessions + stats.

    Only new/changed sessions are (re)analysed; `force=true` re-analyses all.
    `use_llm=false` runs the structural-only path (no task forest, no quota)."""
    APP_CONFIG["source"] = source
    APP_CONFIG["projects_dir"] = projects_dir or sa.DEFAULT_PROJECTS_DIR
    APP_CONFIG["project_filter"] = project_filter
    APP_CONFIG["dataset_repo"] = dataset_repo or sa.DEFAULT_SWECHAT_REPO
    APP_CONFIG["dataset_split"] = dataset_split or sa.DEFAULT_SWECHAT_SPLIT
    analyses, stats = api.analyze_all(use_llm=use_llm, force=force,
                                      source=APP_CONFIG["source"],
                                      projects_dir=APP_CONFIG["projects_dir"],
                                      project_filter=APP_CONFIG["project_filter"],
                                      dataset_repo=APP_CONFIG["dataset_repo"],
                                      dataset_split=APP_CONFIG["dataset_split"],
                                      cache_path=_cache_path())
    rows = [api.analysis_to_dict(a) for a in analyses]
    rows.sort(key=lambda a: a["as_is_cost"], reverse=True)
    return {"sessions": rows, "stats": stats}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """One session's full analysis dict (accepts the short 8-char id or the full id)."""
    for row in _cached_sessions():
        if row["session_id"] == session_id or row["session_id"].startswith(session_id):
            return row
    raise HTTPException(status_code=404, detail="session not found in cache")


# ======================================================================
# Static page + server launcher
# ======================================================================

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Static assets (JS/CSS). Mounted last so it doesn't shadow the API routes.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _find_free_port(host, port, tries=20):
    """Return the first free port at/after `port` (so a second `serve` doesn't die
    with 'address already in use'), or None if none is free in the range."""
    import socket
    for candidate in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return None


def serve(host="127.0.0.1", port=8000, source="local", projects_dir=None,
          project_filter=None, dataset_repo=None, dataset_split=None):
    """Launch the dashboard with uvicorn (used by `split-advisor serve`).

    If the requested port is busy (e.g. an earlier dashboard is still running) we
    fall back to the next free port and say so, instead of crashing. The URL is
    printed only once a port is secured, so it's never misleading."""
    import uvicorn
    APP_CONFIG["source"] = source
    APP_CONFIG["projects_dir"] = projects_dir or sa.DEFAULT_PROJECTS_DIR
    APP_CONFIG["project_filter"] = project_filter
    APP_CONFIG["dataset_repo"] = dataset_repo or sa.DEFAULT_SWECHAT_REPO
    APP_CONFIG["dataset_split"] = dataset_split or sa.DEFAULT_SWECHAT_SPLIT
    chosen = _find_free_port(host, port)
    if chosen is None:
        print(f"Ports {port}–{port + 19} are all in use. Free one, or pass --port <n>.",
              flush=True)
        return
    if chosen != port:
        print(f"Port {port} is in use — starting on {chosen} instead "
              f"(is a dashboard already open? use --port to pick your own).", flush=True)
    # Pre-flight LLM check at startup, so the user sees on screen whether their key
    # was picked up BEFORE they press Refresh — no more silent structural-only.
    import split_advisor as sa
    sa.print_llm_diagnostics(header="startup: LLM check")
    print(f"Source         : {APP_CONFIG['source']}", flush=True)
    if APP_CONFIG["source"] == "local":
        print(f"Projects dir   : {APP_CONFIG['projects_dir']}", flush=True)
    else:
        print(f"Dataset repo   : {APP_CONFIG['dataset_repo']} [{APP_CONFIG['dataset_split']}]",
              flush=True)
    if APP_CONFIG["project_filter"]:
        print(f"Project filter : {APP_CONFIG['project_filter']}", flush=True)
    print(f"Claude Split Advisor dashboard → http://{host}:{chosen}", flush=True)
    print("  (all local; press Refresh in the page to analyse new sessions; Ctrl-C to stop)",
          flush=True)
    uvicorn.run(app, host=host, port=chosen, log_level="warning")


if __name__ == "__main__":
    serve()
