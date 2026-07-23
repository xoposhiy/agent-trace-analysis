"""
analysis_api.py — the shared, cached analysis orchestrator behind the web UI.

WHY THIS EXISTS
---------------
`split_advisor.py` analyses every session from scratch on each run. For the web
dashboard that is wasteful: analysing a session runs an LLM judge call, so
re-analysing the *same* sessions on every page refresh would burn quota and time.

This module adds an INCREMENTAL JSON CACHE. Each analysed session is stored keyed
by its id together with a lightweight file signature (mtime + size). On a refresh
we re-analyse ONLY sessions that are new or whose transcript changed (e.g. you kept
working in them and they grew); every unchanged session is served straight from the
cache with no LLM call. So the first run analyses everything, and later runs touch
only what you actually created since.

The cache is a single JSON file under `~/.claude` (per-user, resolves at runtime),
so nothing about it is machine-specific and it can be deleted at any time to force a
full re-analysis.

Both the web app (`web/app.py`) and, if wanted, the CLI can call `analyze_all()`.
"""
import dataclasses
import json
import os

import session_core as core
import split_advisor as sa

# Cache lives next to the transcripts' root, per-user (never in the install dir).
DEFAULT_CACHE_PATH = os.path.join(
    os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")),
    "split_advisor_cache.json")


def cache_path_for_source(source=sa.DEFAULT_SOURCE, dataset_repo=sa.DEFAULT_SWECHAT_REPO,
                          dataset_split=sa.DEFAULT_SWECHAT_SPLIT):
    """Return a source-specific cache path so local and SWE-chat runs stay separate."""
    if source == "local":
        return DEFAULT_CACHE_PATH
    repo_slug = dataset_repo.replace("/", "__")
    split_slug = dataset_split.replace("/", "_")
    return os.path.join(
        os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")),
        f"split_advisor_cache.{repo_slug}.{split_slug}.json")

# Bump when the analysis logic changes (not just the transcripts), so an old cache
# built by a previous version is discarded and everything is re-analysed once.
# v7: terminology rename in split-point labels ("message" -> "user prompt",
#     "turn" -> "agent step"), so old cached labels are regenerated on next refresh.
# v8: sub-agent candidates now carry split_end_fraction (for the excise-and-rejoin
#     preview), so re-analyse to populate it on existing sessions.
CACHE_VERSION = 8


# ----------------------------------------------------------------------
# File signature — how we decide a session is "unchanged"
# ----------------------------------------------------------------------

def file_signature(path):
    """A cheap fingerprint of a transcript: (mtime, size). If either changes the
    session is treated as new work and re-analysed. Cheap enough to stat every
    session on every refresh."""
    st = os.stat(path)
    return {"mtime": st.st_mtime, "size": st.st_size}


# ----------------------------------------------------------------------
# (De)serialising a SessionAnalysis <-> plain JSON dict
# ----------------------------------------------------------------------

def candidate_to_dict(candidate):
    """Serialise a Candidate, adding the computed dollars/pct so the frontend
    doesn't have to re-derive them from the raw saving dict."""
    data = dataclasses.asdict(candidate)   # source, split_fraction, saving, label, detail
    data["dollars"] = candidate.dollars
    data["pct"] = candidate.pct
    return data


def candidate_from_dict(data):
    """Rebuild a Candidate from its serialised form (dollars/pct are properties,
    so they are recomputed from `saving` and not passed in)."""
    return sa.Candidate(
        source=data["source"],
        split_fraction=data["split_fraction"],
        saving=data["saving"],
        label=data["label"],
        detail=data.get("detail", ""),
        split_end_fraction=data.get("split_end_fraction"),
    )


def analysis_to_dict(analysis):
    """Serialise a SessionAnalysis to a JSON-safe dict (for the cache + the API)."""
    return {
        "session_id": analysis.session_id,
        "project": analysis.project,
        "path": analysis.path,
        "turns": analysis.turns,
        "tool_calls": analysis.tool_calls,
        "peak_context": analysis.peak_context,
        "as_is_cost": analysis.as_is_cost,
        "pattern": analysis.pattern,
        "already_plan_mode": analysis.already_plan_mode,
        "candidates": [candidate_to_dict(c) for c in analysis.candidates],
        "best": candidate_to_dict(analysis.best) if analysis.best else None,
        "modelled": analysis.modelled,
        "task_summary": analysis.task_summary,
        "task_forest": analysis.task_forest,
        "full_split": analysis.full_split,
        "phases": analysis.phases,
        "seq_turns": analysis.seq_turns,
    }


def analysis_from_dict(data):
    """Rebuild a SessionAnalysis from a cached dict so the rest of the pipeline
    (suggestion gate, ranking, rendering) treats cached and fresh results alike."""
    analysis = sa.SessionAnalysis(
        session_id=data["session_id"], project=data["project"], path=data["path"],
        turns=data["turns"], tool_calls=data["tool_calls"],
        peak_context=data["peak_context"], as_is_cost=data["as_is_cost"],
        pattern=data["pattern"], already_plan_mode=data["already_plan_mode"],
    )
    analysis.candidates = [candidate_from_dict(c) for c in data.get("candidates", [])]
    analysis.best = candidate_from_dict(data["best"]) if data.get("best") else None
    analysis.modelled = data.get("modelled", True)
    analysis.task_summary = data.get("task_summary", "")
    analysis.task_forest = data.get("task_forest")
    analysis.full_split = data.get("full_split")
    analysis.phases = data.get("phases", [])
    analysis.seq_turns = data.get("seq_turns", 0)
    return analysis


# ----------------------------------------------------------------------
# Cache load / save
# ----------------------------------------------------------------------

def load_cache(cache_path=DEFAULT_CACHE_PATH):
    """Load the cache, or a fresh empty one if missing/unreadable/old-version."""
    try:
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("version") == CACHE_VERSION and isinstance(cache.get("sessions"), dict):
            return cache
    except Exception:
        pass
    return {"version": CACHE_VERSION, "sessions": {}}


def save_cache(cache, cache_path=DEFAULT_CACHE_PATH):
    """Write the cache atomically (temp file + rename) so a crash mid-write can't
    corrupt it."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)


# ----------------------------------------------------------------------
# The incremental analysis itself
# ----------------------------------------------------------------------

def analyze_all(projects_dir=sa.DEFAULT_PROJECTS_DIR, project_filter=None,
                use_llm=True, alpha=core.SUMMARY_ALPHA,
                model=sa.DEFAULT_JUDGE_MODEL, base_url=sa.DEFAULT_LLM_BASE_URL,
                cache_path=None, force=False, progress=None,
                source=sa.DEFAULT_SOURCE, dataset_repo=sa.DEFAULT_SWECHAT_REPO,
                dataset_split=sa.DEFAULT_SWECHAT_SPLIT, max_workers=8):
    """Analyse every discovered session, reusing the cache for unchanged ones.

    Only sessions that are NEW or whose transcript changed since last time are
    (re)analysed — everything else comes from the cache with no LLM call. Pass
    force=True to ignore the cache and re-analyse all. `progress(session_id, n)` is
    called after each freshly analysed session (for a live progress log).

    Returns (analyses, stats) where stats = {analyzed, reused, total}. The LLM
    client is built lazily, so a fully-cached refresh makes no network calls at all.
    """
    if cache_path is None:
        cache_path = cache_path_for_source(source, dataset_repo, dataset_split)

    sessions = sa.discover_sessions(projects_dir=projects_dir, project_filter=project_filter,
                                    source=source, dataset_repo=dataset_repo,
                                    dataset_split=dataset_split)
    cache = load_cache(cache_path)
    entries = cache["sessions"]

    # Pre-flight: say up front whether the LLM will run, and if not, exactly why.
    # This is the piece that was missing — the web path used to degrade to
    # structural-only in total silence, so a user who supplied a key had no way to
    # tell it was never picked up.
    diag = None
    if use_llm:
        diag = sa.print_llm_diagnostics(base_url, model, header="refresh: LLM check")

    client = None
    client_tried = False
    client_error = None          # why make_llm_client failed, if it did
    judge_errors = 0             # sessions whose judge call errored this run
    first_judge_error = None     # the first such error, verbatim (bad key/model/timeout)
    llm_judged = 0              # sessions actually judged by the LLM this run
    analyses = []
    analyzed = reused = 0

    # --- Phase 1: partition sessions into reuse-from-cache vs (re)analyse ---
    # Reuse when the transcript is unchanged, being careful about LLM mode:
    #   - a cached LLM result satisfies ANY request — we never throw away a forest
    #     just because the toggle is off, so flipping LLM off/on never re-runs a
    #     session already analysed with the LLM;
    #   - a cached structural-only result satisfies only a no-LLM request; if the
    #     caller now wants the LLM, we fall through and UPGRADE it.
    to_analyze = []
    for session_id, path, project in sessions:
        try:
            sig = file_signature(path)
        except OSError:
            continue      # transcript vanished between discovery and stat — skip
        cached = entries.get(session_id)
        if not force and cached and cached.get("sig") == sig:
            cached_llm = bool(cached.get("llm"))
            if cached_llm or not use_llm:
                analyses.append(analysis_from_dict(cached["analysis"]))
                reused += 1
                continue
        to_analyze.append((session_id, path, project, sig))

    # --- Phase 2: (re)analyse the rest, with the judge calls run CONCURRENTLY ---
    # The per-session LLM judge is a network call, so a big-repo run is dominated by
    # waiting on I/O. A thread pool turns an N×~5s sequential crawl into ~N/workers,
    # which is the difference between ~73 min and a few minutes on an 860-session repo.
    # We also CHECKPOINT the cache every few completions so a long run's progress
    # survives an interruption — the old code saved only once, at the very end.
    if to_analyze:
        if use_llm:
            client_tried = True
            try:
                client = sa.make_llm_client(base_url)
            except Exception as e:
                client = None      # degrade to structural-only, like the CLI does
                client_error = str(e)
                print(f"  [LLM] could not build client — running structural-only: {e}",
                      flush=True)

        def _run(item):
            session_id, path, project, sig = item
            a = sa.analyze_session(session_id, path, project, alpha=alpha,
                                   client=client, model=model)
            return item, a

        # Only fan out when the LLM (the slow, I/O-bound part) is in play; structural
        # analysis is CPU-bound and already fast, so a single worker keeps it simple.
        workers = max(1, max_workers) if client is not None else 1

        # Results are consumed on THIS thread (as_completed), so the cache dict, the
        # counters and `analyses` are only ever mutated single-threaded — no locks
        # needed. Worker threads only read transcripts and call the (thread-safe) client.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        since_save = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, item) for item in to_analyze]
            for fut in as_completed(futures):
                (session_id, path, project, sig), a = fut.result()
                # Track what the judge actually did, so the end-of-run verdict can tell
                # the user whether the LLM was really used (a present client is not
                # enough — the endpoint/model can still reject calls).
                if client is not None:
                    if a.llm_error and first_judge_error is None:
                        first_judge_error = a.llm_error
                    if a.llm_error:
                        judge_errors += 1
                        print(f"  [LLM] judge error on {session_id[:8]}: {a.llm_error}",
                              flush=True)
                    elif a.task_forest:
                        llm_judged += 1
                # Mark the entry as an LLM result only if the client was present AND the
                # judge didn't error. Sessions too short to judge keep llm_ok True, so
                # they're reused; a genuine judge failure leaves llm_ok False so that
                # session is retried next healthy refresh, not stored as an empty result.
                got_llm = client is not None and a.llm_ok
                entries[session_id] = {
                    "sig": sig,
                    "llm": got_llm,
                    "analysis": analysis_to_dict(a),
                }
                analyses.append(a)
                analyzed += 1
                since_save += 1
                if progress:
                    progress(session_id, analyzed)
                if since_save >= 25:      # checkpoint so a long run isn't all-or-nothing
                    save_cache(cache, cache_path)
                    since_save = 0

    save_cache(cache, cache_path)

    # End-of-run verdict: was the LLM ACTUALLY used, and if not, why? This is the
    # single line that answers "did it run with the LLM or not?".
    #
    # Order matters. The client is built LAZILY (only when a session actually needs
    # analysing), so `client is None` on its own does NOT mean failure — if nothing
    # needed re-analysing (all sessions served from cache) the client is simply never
    # built. We check `client_tried` first so that all-cached refreshes read as a
    # healthy "nothing to do", not a scary "no client". A genuine build failure is
    # caught separately via `client_error`.
    if not use_llm:
        llm_active = False
        llm_status = "off (structural-only requested)"
    elif not client_tried:
        # No new/changed session this refresh → everything came from the cache. The
        # LLM wasn't exercised, but the cached results ARE LLM results (a structural
        # cache entry would have forced a re-analysis), so this is fine, not a failure.
        llm_active = True
        llm_status = ("not needed this refresh — all sessions served from cache "
                      "(existing LLM results preserved; use Force to re-run)")
    elif client_error is not None:
        llm_active = False
        llm_status = f"OFF — client build failed: {client_error}"
    elif client is None:
        llm_active = False
        llm_status = "off (no client)"
    elif judge_errors and not llm_judged:
        # EVERY judge call errored and none succeeded → the LLM genuinely isn't working.
        # This is the case that usually IS a key/base_url/model problem.
        llm_active = False
        llm_status = (f"FAILING — all {judge_errors} judge call(s) errored "
                      f"(first: {first_judge_error}). "
                      "Check the API key, base_url and model name for your endpoint.")
    elif llm_judged:
        # The LLM worked. Some big sessions may still have errored (e.g. a response too
        # long to fit the token budget) — note that, but don't call the whole run failed.
        llm_active = True
        if judge_errors:
            llm_status = (f"ACTIVE — {llm_judged} session(s) judged; "
                          f"{judge_errors} errored (first: {first_judge_error})")
        else:
            llm_status = f"ACTIVE — {llm_judged} session(s) judged this run"
    else:
        llm_active = True
        llm_status = ("enabled, but no session this run needed a judge call "
                      "(all cached or too short) — nothing to report")
    print(f"[refresh done] analyzed={analyzed} reused={reused} "
          f"total={len(sessions)} | LLM: {llm_status}", flush=True)

    stats = {
        "analyzed": analyzed,
        "reused": reused,
        "total": len(sessions),
        # LLM diagnostics so the web UI can surface them too (not just the console).
        "llm": {
            "requested": use_llm,
            "active": llm_active,
            "status": llm_status,
            "judged": llm_judged,
            "judge_errors": judge_errors,
            "first_error": first_judge_error,
            "client_error": client_error,
            "config": diag,
        },
    }
    return analyses, stats


def suggestions_for(analyses, min_pct=sa.DEFAULT_MIN_PCT, min_dollars=sa.DEFAULT_MIN_DOLLARS):
    """Apply the % + $ suggestion gate to a list of analyses (reuses the CLI's
    decide_suggestion so the web UI and the report agree)."""
    return [s for s in (sa.decide_suggestion(a, min_pct, min_dollars) for a in analyses) if s]
