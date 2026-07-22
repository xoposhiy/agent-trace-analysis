"""
split_advisor.py — a LOCAL CLI that scans your own Claude Code sessions and tells
you which ones were worth splitting (and how much money a split would have saved).

WHAT IT DOES
------------
Claude Code writes every session as a JSONL transcript under
`~/.claude/projects/<encoded-project-path>/<session-id>.jsonl`. This tool reads
those transcripts (fully offline), and for each session:

  1. Reconstructs the ordered tool-call sequence and detects the structural
     pattern (session_core.classify_session_pattern):
       A — front-loaded reading phase  -> "plan-mode" opportunity
       C — mid-session reading burst   -> "sub-agent" opportunity
  2. Prices the best semantic split at that boundary with the Research7 chunk-split
     math model (session_core.saving_for_split), charging the true summary-carry cost.
  3. SUGGESTS a split only when the modelled saving clears BOTH a percentage floor
     and a dollar floor (so tiny sessions don't nag).

Whenever an LLM endpoint is reachable (the default), it ALSO asks an LLM-as-judge
to segment the session into its FOREST of tasks: the ordered sequence of distinct
goals, allowing tasks to recur (T1 -> T2 -> T1 -> T3), plus within-task "sub-agent
opportunities" (tangents the main session could have offloaded). Every task boundary
and every tangent is priced as a split (full pricing on every boundary), the best is
kept, and a one-line summary labels the session (session IDs alone are meaningless).
`--no-llm` turns all of that off for a fully offline structural run.

All the analysis/pricing lives in `session_core.py` (shared with the dataset
script `main.py`); this file only adds local discovery, gating, and reporting.

USAGE
-----
    python split_advisor.py                     # scan all local sessions, write report
    python split_advisor.py --summary-only      # console summary only, no file
    python split_advisor.py --project agent-trace   # only sessions of one project
    python split_advisor.py --min-pct 8 --min-dollars 0.25
    python split_advisor.py --no-llm            # fully offline, structural heuristics only

By default the summaries and the task-switch judge call an OpenAI-compatible endpoint
(e.g. a LiteLLM proxy); set OPENAI_API_KEY (and --llm-base-url / --llm-model) first. If
no endpoint is reachable the tool degrades to a structural-only report instead of failing.
"""
import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import chunk_split_model as csm
import session_core as core


# Records what happened when we tried to load a project `.env`, so the diagnostics
# can explain a silently-missing key (dotenv not installed, no .env found, or a .env
# that existed but had no usable non-empty values).
#   dotenv: was python-dotenv importable?
#   found:  .env files that existed on disk (whether or not they had values)
#   loaded: .env files that actually contributed at least one non-empty value
_ENV_LOAD_INFO = {"dotenv": False, "found": [], "loaded": []}


def _load_project_env():
    """Load a project `.env` (if python-dotenv is installed) so LLM settings can
    live in a file instead of shell exports. Runs at import time, BEFORE the
    env-derived defaults below are read.

    Precedence, from strongest to weakest:
      1. a real (non-empty) shell export        — always wins, never overridden;
      2. the `.env` next to this code;
      3. the `.env` in the current directory.

    We merge manually with `dotenv_values` rather than `load_dotenv` on purpose, to
    fix a genuinely nasty trap: a BLANK `OPENAI_API_KEY=` line (the shipped template
    and `.env.example` both have one) loaded via `load_dotenv(override=False)` writes
    an empty string into the environment, which then BLOCKS a real key from a second
    `.env` (or makes the app look like the key "wasn't picked up"). Here an empty
    value never overrides anything — only a real, non-empty value fills a gap.

    Silently doing nothing if python-dotenv isn't installed is recorded in
    `_ENV_LOAD_INFO` so the diagnostics can flag it (a missing python-dotenv is the
    #1 reason a key in `.env` is never picked up)."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    _ENV_LOAD_INFO["dotenv"] = True
    # the .env shipped next to this code (works from any CWD), then one in the CWD
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    seen = set()
    for path in candidates:
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        _ENV_LOAD_INFO["found"].append(path)
        applied = False
        for key, value in dotenv_values(path).items():
            # only fill a var that isn't already a real value, and only with a real
            # value — so shell exports win and blank lines never clobber anything.
            if value and not os.environ.get(key):
                os.environ[key] = value
                applied = True
        if applied:
            _ENV_LOAD_INFO["loaded"].append(path)


_load_project_env()

# ----------------------------------------------------------------------
# Defaults / config
# ----------------------------------------------------------------------
# Where Claude Code stores transcripts. This resolves per-user at runtime (via
# $HOME / $CLAUDE_CONFIG_DIR), so it is correct on any machine, not just the author's.
DEFAULT_PROJECTS_DIR = os.path.join(
    os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")), "projects")
DEFAULT_MIN_PCT = 10.0        # suggest only if modelled saving is >= this % of cost
DEFAULT_MIN_DOLLARS = 0.50    # ...AND at least this many dollars
# Report goes to the current working directory by default — never into the install
# location (which may be read-only when pip-installed).
DEFAULT_OUT = os.path.join(os.getcwd(), "split_report.md")

# --llm settings. The judge uses the OpenAI-compatible chat-completions API, so it
# works against ANY OpenAI-compatible endpoint (OpenAI itself, OpenRouter, a LiteLLM
# proxy, a local server, ...). Nothing provider-specific is baked in: the base URL
# defaults to $OPENAI_BASE_URL (unset -> the OpenAI SDK's own default endpoint), and
# the model defaults to $SPLIT_ADVISOR_MODEL. Set --llm-model to match your endpoint
# (e.g. "gpt-4o-mini" for OpenAI, "anthropic/claude-haiku-4-5" for a Claude proxy).
DEFAULT_JUDGE_MODEL = os.environ.get("SPLIT_ADVISOR_MODEL", "gpt-4o-mini")
DEFAULT_LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL")   # None -> SDK default


# ======================================================================
# Local session discovery + reading
# ======================================================================

def discover_sessions(projects_dir=DEFAULT_PROJECTS_DIR, project_filter=None):
    """Find every session transcript. Returns [(session_id, path, project), ...]."""
    found = []
    for project_dir in sorted(glob.glob(os.path.join(projects_dir, "*"))):
        if not os.path.isdir(project_dir):
            continue
        project = os.path.basename(project_dir)
        if project_filter and project_filter not in project:
            continue
        for path in sorted(glob.glob(os.path.join(project_dir, "*.jsonl"))):
            session_id = os.path.splitext(os.path.basename(path))[0]
            found.append((session_id, path, project))
    return found


def read_local_transcript(path):
    """Parse one local JSONL transcript into a list of events (skip bad lines)."""
    events = []
    with open(path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue      # rare malformed line — safe to skip
    return events


def used_plan_mode(events):
    """True if the user ever switched this session into plan mode — detected via
    Claude Code's `permission-mode` events (permissionMode == 'plan')."""
    for event in events:
        if event.get("type") == "permission-mode" and event.get("permissionMode") == "plan":
            return True
    return False


# ======================================================================
# Per-session analysis
# ======================================================================

@dataclass
class Candidate:
    """One possible split point for a session, already priced."""
    # "plan-mode" (front-loaded reading) | "task-switch" (independent task, priced as
    # a fresh session) | "sub-agent" (a child T1.x or a structural reading burst,
    # priced with the excise-and-rejoin sub-agent model)
    source: str
    split_fraction: float
    saving: dict                # from core.saving_for_split
    label: str                  # short human description of WHERE the split is
    detail: str = ""            # extra context (e.g. reason / call count)
    # For a "sub-agent" candidate: the END fraction of the excised segment (its start
    # is split_fraction). Lets the UI draw the excise-and-rejoin shape rather than a
    # two-session split. None for task-switch / plan-mode (they split at one point).
    split_end_fraction: Optional[float] = None

    @property
    def dollars(self):
        return self.saving["dollar_saving"]

    @property
    def pct(self):
        c = self.saving["as_is_cost"]
        return 100 * self.saving["dollar_saving"] / c if c else 0.0


@dataclass
class SessionAnalysis:
    session_id: str
    project: str
    path: str
    turns: int
    tool_calls: int
    peak_context: float
    as_is_cost: float
    pattern: str
    already_plan_mode: bool
    candidates: list = field(default_factory=list)
    best: Optional[Candidate] = None
    modelled: bool = True       # False if the session doesn't fit the cost model
    task_summary: str = ""      # one-line LLM summary of the task (only with --llm)
    # The LLM task FOREST for this session (only with --llm). None when no LLM ran
    # or the session was too short to judge. Shape (see _build_task_forest):
    #   {tasks, assignments, timeline, num_switches, recurring}  (ids are hierarchical)
    task_forest: Optional[dict] = None
    # Headline saving from splitting the WHOLE forest into separate sessions at once
    # (every task boundary), priced via core.saving_for_multi_split. None if unpriceable.
    full_split: Optional[dict] = None
    # STRUCTURAL activity phases (always available, no LLM): the reading/editing/execution
    # runs used by the plan-mode view. [{category, start_turn, end_turn}, ...] in turn
    # space; `seq_turns` is the total tool-call turns those phases span.
    phases: list = field(default_factory=list)
    seq_turns: int = 0
    # Transient (NOT serialised): False only if the LLM judge was called and ERRORED
    # (e.g. endpoint down). Sessions too short to judge keep it True. The cache uses it
    # so a real result is reused but a failed one is retried next refresh.
    llm_ok: bool = True
    # Transient (NOT serialised): the raw error string from the judge call when it
    # failed (e.g. a 401 from a wrong key, a 404 for an unknown model, a timeout with
    # no VPN). Surfaced by the diagnostics so a silent degrade becomes explainable.
    llm_error: str = ""


def _plan_mode_candidate(buckets, seq_turns, expl_phases, alpha):
    """Pattern A: split at the end of the opening exploration phase."""
    first = expl_phases[0]
    frac = first["end_turn"] / seq_turns if seq_turns else 0.0
    saving = core.saving_for_split(buckets, frac, alpha)
    if not saving:
        return None
    return Candidate(
        source="plan-mode",
        split_fraction=frac,
        saving=saving,
        label=f"after agent step {first['end_turn']} (end of opening reading phase)",
        detail=f"~{first['length']} reads before the first edit",
    )


def _subagent_candidate(buckets, seq_turns, expl_phases, alpha):
    """Pattern C: offload the mid-session reading burst to a sub-agent. Priced with
    the excise-and-rejoin model over the burst's turn span (the main thread resumes
    without the burst's context), matching the LLM sub-agent candidates."""
    burst = expl_phases[0]
    if not seq_turns:
        return None
    f_start = burst["start_turn"] / seq_turns
    f_end = burst["end_turn"] / seq_turns
    saving = core.saving_for_subagent(buckets, f_start, f_end, alpha)
    if not saving:
        return None
    return Candidate(
        source="sub-agent",
        split_fraction=f_start,
        split_end_fraction=f_end,
        saving=saving,
        label=f"agent steps {burst['start_turn']}–{burst['end_turn']} (mid-session reading burst)",
        detail=f"~{burst['length']}-call reading burst that a sub-agent could handle",
    )


def _build_task_forest(forest, prompts):
    """Turn a raw judge result into the compact task-forest structure stored on
    SessionAnalysis. `assignments` is trimmed to the number of prompts (the LLM can
    occasionally return a different length); ids are hierarchical (T1, T1.1)."""
    assignments = (forest.get("assignments") or [])[:len(prompts)]
    interleave = core.detect_interleaving(assignments)
    return {
        "tasks": forest.get("tasks") or [],
        "assignments": assignments,
        "timeline": interleave["runs"],            # [(task_id, start_msg, end_msg), ...]
        "num_switches": interleave["num_switches"],
        "recurring": interleave["recurring"],      # TOP-LEVEL ids the user returned to
    }


def _run_spans(timeline, prompts, total_turns):
    """Attach a turn span to each run of the compressed timeline: a run runs from its
    first message's turn until the NEXT run's first message (or session end). Returns
    [{id, start_msg, end_msg, f_start, f_end}, ...] with fractions in 0..1."""
    spans = []
    for i, (tid, start_msg, end_msg) in enumerate(timeline):
        start_turn = prompts[start_msg - 1][0] if start_msg - 1 < len(prompts) else 0
        if i + 1 < len(timeline):
            nxt_start = timeline[i + 1][1]
            end_turn = prompts[nxt_start - 1][0] if nxt_start - 1 < len(prompts) else total_turns
        else:
            end_turn = total_turns
        spans.append({
            "id": tid, "start_msg": start_msg, "end_msg": end_msg,
            "f_start": start_turn / total_turns if total_turns else 0.0,
            "f_end": end_turn / total_turns if total_turns else 0.0,
        })
    return spans


def _forest_candidates(task_forest, buckets, prompts, total_turns, alpha):
    """--llm only: build every priced Candidate implied by the HIERARCHICAL forest.

    The role of each run falls straight out of its id, so nothing is double-counted:
      * a run whose id is a child (T1.1) -> a SUB-AGENT: its segment is priced with
        the excise-and-rejoin model (core.saving_for_subagent) over the run's turn span.
      * a transition into a NEW top-level id -> an INDEPENDENT task switch, priced as a
        fresh session there (core.saving_for_split).
      * a transition back into a top-level id seen before -> a RETURN: no split point.
    """
    candidates = []
    if not task_forest or total_turns <= 0:
        return candidates

    labels = {t.get("id"): t.get("label", "") for t in task_forest.get("tasks", [])}
    spans = _run_spans(task_forest["timeline"], prompts, total_turns)
    seen_top, prev_top = set(), None

    for span in spans:
        tid = span["id"]
        tl = core.top_level(tid)

        if core.is_subtask(tid):
            # a sub-agent child: price offloading its segment to a sub-agent. Skip when
            # the modelled saving is <= 0 (a tiny segment whose summary overhead exceeds
            # what excising it saves — not a real opportunity).
            saving = core.saving_for_subagent(buckets, span["f_start"], span["f_end"], alpha)
            if saving and saving["dollar_saving"] > 0:
                parent_label = labels.get(tl, tl)
                candidates.append(Candidate(
                    source="sub-agent", split_fraction=span["f_start"],
                    split_end_fraction=span["f_end"], saving=saving,
                    label=(f"{tid}: user prompts {span['start_msg']}–{span['end_msg']} "
                           f"(sub-agent within {tl})"),
                    detail=(labels.get(tid) or f"a side-task within {parent_label}")))
        elif prev_top is not None and tl != prev_top and tl not in seen_top:
            # a genuinely new independent task -> a fresh-session split point
            saving = core.saving_for_split(buckets, span["f_start"], alpha)
            if saving and saving["dollar_saving"] > 0:
                candidates.append(Candidate(
                    source="task-switch", split_fraction=span["f_start"], saving=saving,
                    label=(f"at user prompt {span['start_msg']} "
                           f"({prev_top}→{tl}, independent task)"),
                    detail=f"independent task switch {prev_top}→{tl}"))
        # else: a return to an earlier top-level task -> not a split point

        prev_top = tl
        seen_top.add(tl)

    return candidates


def _top_level_fractions(task_forest, prompts, total_turns):
    """Split fractions for the whole-forest 'into k sessions' headline: one cut at
    each TOP-LEVEL task change (children stay with their parent)."""
    fracs, prev_top = [], None
    for span in _run_spans(task_forest["timeline"], prompts, total_turns):
        tl = core.top_level(span["id"])
        if prev_top is not None and tl != prev_top:
            fracs.append(span["f_start"])
        prev_top = tl
    return fracs


def analyze_session(session_id, path, project, alpha=core.SUMMARY_ALPHA,
                    client=None, model=DEFAULT_JUDGE_MODEL):
    """Analyse one session end-to-end. Returns a SessionAnalysis.

    When `client` is set (an LLM endpoint is reachable) the analysis also runs
    the task-switch judge and writes a one-line task summary for any session
    that has a split option. With no client it is a purely structural analysis."""
    events = read_local_transcript(path)
    buckets = core.session_token_buckets(events)
    input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls = buckets
    as_is = csm.full_session_cost(input_tok, output_tok, cache_write_tok, cache_read_tok)

    sequence, seq_turns = core.extract_tool_sequence(events)
    phases = core.detect_phases(sequence)
    pattern, expl_phases = core.classify_session_pattern(phases, sequence)

    analysis = SessionAnalysis(
        session_id=session_id,
        project=project,
        path=path,
        turns=api_calls,
        tool_calls=len(sequence),
        peak_context=(csm.peak_context_tokens(cache_read_tok, api_calls)
                      if cache_read_tok > 0 and api_calls >= 2 else 0.0),
        as_is_cost=as_is,
        pattern=pattern,
        already_plan_mode=used_plan_mode(events),
    )
    # structural phase strip (reading/editing/execution) for the plan-mode view
    analysis.phases = [{"category": p["category"], "start_turn": p["start_turn"],
                        "end_turn": p["end_turn"]} for p in phases]
    analysis.seq_turns = seq_turns

    # A session must have real cache-read + enough calls to be priced at all.
    if cache_read_tok <= 0 or api_calls < 2:
        analysis.modelled = False
        return analysis

    candidates = []
    if pattern == "A" and expl_phases:
        c = _plan_mode_candidate(buckets, seq_turns, expl_phases, alpha)
        if c:
            candidates.append(c)
    elif pattern == "C" and expl_phases:
        c = _subagent_candidate(buckets, seq_turns, expl_phases, alpha)
        if c:
            candidates.append(c)

    judge_forest = None
    prompts = []
    if client is not None:
        prompts, total_turns = core.user_prompts_with_turns(events)
        # One judge call per session segments the whole task forest AND returns the
        # session summary — sessions with enough prompts cost no extra request.
        if len(prompts) >= 4 and total_turns > 0:
            judge_forest = judge_task_forest(client, [t for _, t in prompts],
                                             model=model)
            if judge_forest and "error" not in judge_forest:
                analysis.task_forest = _build_task_forest(judge_forest, prompts)
                # per-run turn-fraction spans, so the dashboard graph can position the
                # task bands in the same turn-space as the split markers and cost ramp.
                analysis.task_forest["spans"] = _run_spans(
                    analysis.task_forest["timeline"], prompts, total_turns)
                # hierarchical ids drive the roles: children -> sub-agent (excise math),
                # new top-level -> task switch, returns -> skipped.
                candidates.extend(_forest_candidates(analysis.task_forest, buckets,
                                                     prompts, total_turns, alpha))
                # headline: split the whole forest into one session per INDEPENDENT task
                fracs = _top_level_fractions(analysis.task_forest, prompts, total_turns)
                analysis.full_split = core.saving_for_multi_split(buckets, fracs, alpha)
            else:
                # the judge was called but errored (e.g. endpoint down) — mark it so the
                # cache retries this session next time instead of storing an empty result,
                # and keep the raw error so the diagnostics can explain WHY (bad key, bad
                # model, unreachable endpoint, timeout, ...).
                analysis.llm_ok = False
                if judge_forest and judge_forest.get("error"):
                    analysis.llm_error = str(judge_forest["error"])

    analysis.candidates = candidates
    if candidates:
        analysis.best = max(candidates, key=lambda c: c.dollars)

    # Chosen scope: only summarise sessions that produced a split option.
    if client is not None and candidates:
        analysis.task_summary = _session_summary(client, model, prompts, judge_forest)
    return analysis


def _session_summary(client, model, prompts, judge_result):
    """The session's one-line task summary: reuse the judge call's summary if we
    have one, else make a small dedicated summary call. Never raises."""
    if judge_result and "error" not in judge_result:
        summary = (judge_result.get("summary") or "").strip()
        if summary:
            return summary
    if prompts:
        return summarize_session(client, [t for _, t in prompts], model=model)
    return ""


# ======================================================================
# Suggestion decision (the % + $ gate)
# ======================================================================

@dataclass
class Suggestion:
    analysis: SessionAnalysis
    candidate: Candidate
    message: str
    informational: bool = False   # True = already acted on (e.g. plan mode used)


def _suggestion_message(cand):
    if cand.source == "plan-mode":
        return (f"Did {cand.detail} — running that opening exploration in plan mode "
                f"(or a separate reading session) would cut re-read cost by "
                f"~${cand.dollars:.2f} ({cand.pct:.0f}%).")
    if cand.source == "sub-agent":
        reason = f" ({cand.detail})" if cand.detail else ""
        return (f"Has a self-contained side-task{reason}; handing it to a sub-agent "
                f"(so the main thread never carries its context) would save "
                f"~${cand.dollars:.2f} ({cand.pct:.0f}%).")
    if cand.source == "task-switch":
        reason = f" ({cand.detail})" if cand.detail else ""
        return (f"Switched to an independent task{reason}; starting a fresh session "
                f"there would save ~${cand.dollars:.2f} ({cand.pct:.0f}%).")
    return f"Splitting {cand.label} would save ~${cand.dollars:.2f} ({cand.pct:.0f}%)."


def decide_suggestion(analysis, min_pct=DEFAULT_MIN_PCT, min_dollars=DEFAULT_MIN_DOLLARS):
    """Return a Suggestion if this session clears both floors, else None."""
    cand = analysis.best
    if cand is None:
        return None
    if cand.pct < min_pct or cand.dollars < min_dollars:
        return None
    # If the win is a plan-mode split but the user ALREADY used plan mode here,
    # downgrade to an informational note rather than nagging them again.
    if cand.source == "plan-mode" and analysis.already_plan_mode:
        return Suggestion(analysis, cand,
                          message=("Front-loaded reading detected, but this session "
                                   "already used plan mode — the opportunity was taken."),
                          informational=True)
    return Suggestion(analysis, cand, message=_suggestion_message(cand))


# ======================================================================
# LLM task-switch judge (only used with --llm)
# ======================================================================

def make_llm_client(base_url=DEFAULT_LLM_BASE_URL):
    """Lazily build the OpenAI-compatible client (needs OPENAI_API_KEY). Works
    against any OpenAI-compatible endpoint via `base_url`; when `base_url` is None
    the SDK uses its own default endpoint."""
    from openai import OpenAI      # imported here so the default path needs no openai
    return OpenAI(base_url=base_url) if base_url else OpenAI()


# ======================================================================
# LLM diagnostics — make a silent degrade to structural-only explainable
# ======================================================================

def _mask_key(key):
    """Show just enough of a key to confirm the right one is loaded, never the whole
    thing: 'sk-abcd…wxyz (43 chars)'. Returns '(not set)' when empty."""
    if not key:
        return "(not set)"
    body = f"{key[:5]}…{key[-4:]}" if len(key) > 12 else "…"
    return f"{body} ({len(key)} chars)"


def llm_config_diagnostics(base_url=None, model=None):
    """Inspect the current LLM configuration WITHOUT making a network call, and
    return a dict describing exactly what the tool will do:

        {dotenv, env_files, key_set, key_masked, base_url, model, openai_installed,
         will_use_llm, reasons}

    `reasons` is a list of human-readable notes explaining anything that would stop
    the LLM from being used (no key, dotenv missing, .env not found, ...). This is
    the single source of truth for both the CLI and the web dashboard so their
    "why is the LLM off?" messages always agree."""
    key = os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url if base_url is not None else DEFAULT_LLM_BASE_URL
    model = model or DEFAULT_JUDGE_MODEL
    try:
        import openai  # noqa: F401
        openai_installed = True
    except ImportError:
        openai_installed = False

    reasons = []
    if not openai_installed:
        reasons.append("the `openai` package is not installed — run "
                       "`pip install \".[llm]\"` (or `pip install openai`).")
    if not key:
        reasons.append("OPENAI_API_KEY is not set in the environment.")
        if not _ENV_LOAD_INFO["dotenv"]:
            reasons.append("python-dotenv is NOT installed, so any key in a `.env` "
                           "file was ignored — run `pip install \".[llm]\"` or export "
                           "the key in your shell.")
        elif not _ENV_LOAD_INFO["found"]:
            reasons.append("no `.env` file was found next to the code or in the "
                           "current directory (checked both). Copy .env.example to "
                           ".env and put your key in it, or export OPENAI_API_KEY.")
        else:
            found = ", ".join(_ENV_LOAD_INFO["found"])
            reasons.append(f"a `.env` was found ({found}) but its OPENAI_API_KEY line "
                           "is blank — put your key after the '=' (no quotes, no spaces).")

    return {
        "dotenv": _ENV_LOAD_INFO["dotenv"],
        "env_files": list(_ENV_LOAD_INFO["loaded"]),
        "env_files_found": list(_ENV_LOAD_INFO["found"]),
        "key_set": bool(key),
        "key_masked": _mask_key(key),
        "base_url": base_url or "(OpenAI SDK default)",
        "model": model,
        "openai_installed": openai_installed,
        "will_use_llm": openai_installed and bool(key),
        "reasons": reasons,
    }


def print_llm_diagnostics(base_url=None, model=None, header="LLM configuration"):
    """Print the pre-flight LLM diagnostics (no network call). Called at startup so a
    user immediately sees whether their key was picked up and, if not, precisely why.
    Returns the diagnostics dict so callers can branch on `will_use_llm`."""
    d = llm_config_diagnostics(base_url, model)
    print(f"[{header}]", flush=True)
    print(f"  openai package : {'yes' if d['openai_installed'] else 'NO'}", flush=True)
    print(f"  python-dotenv  : {'yes' if d['dotenv'] else 'no (.env files ignored)'}",
          flush=True)
    if d["env_files"]:
        print(f"  .env loaded    : {', '.join(d['env_files'])}", flush=True)
    elif d["env_files_found"]:
        print(f"  .env found     : {', '.join(d['env_files_found'])} "
              f"(no non-empty values used)", flush=True)
    else:
        print("  .env loaded    : none found", flush=True)
    print(f"  OPENAI_API_KEY : {d['key_masked']}", flush=True)
    print(f"  base_url       : {d['base_url']}", flush=True)
    print(f"  model          : {d['model']}", flush=True)
    if d["will_use_llm"]:
        print("  → LLM ENABLED (task summaries + task-forest judge will run).",
              flush=True)
        print("    If summaries are still missing, the endpoint/model may be "
              "rejecting calls — watch for per-refresh judge errors below.", flush=True)
    else:
        print("  → LLM OFF — falling back to structural-only. Why:", flush=True)
        for r in d["reasons"]:
            print(f"      - {r}", flush=True)
    return d


def judge_task_forest(client, prompt_texts, model=DEFAULT_JUDGE_MODEL):
    """One LLM call segmenting the session into its task forest (tasks +
    per-message assignments + within-task sub-agent opportunities + a one-line
    summary). Returns the parsed dict, or {'error': ...} on any failure.

    max_tokens is generous because the response scales with the message count
    (one assignment per message) rather than the fixed old yes/no answer."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=900,
            temperature=0,
            messages=[{"role": "user",
                       "content": core.build_task_forest_prompt(prompt_texts)}],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def summarize_session(client, prompt_texts, model=DEFAULT_JUDGE_MODEL):
    """One LLM call returning a one-line task summary (used for sessions too
    short for the judge). Returns the sentence, or '' on any failure."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=120,
            temperature=0,
            messages=[{"role": "user",
                       "content": core.build_summary_prompt(prompt_texts)}],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return (json.loads(text).get("summary") or "").strip()
    except Exception:
        return ""


# ======================================================================
# Reporting
# ======================================================================

def _fmt_k(tokens):
    return f"{tokens/1000:.0f}k" if tokens else "—"


def _forest_timeline_str(forest):
    """The compact forest string, e.g. 'T1 → T2 → T1 → T3', from the run list."""
    return " → ".join(tid for tid, _, _ in forest.get("timeline", [])) or "—"


def _forest_detail_lines(a):
    """The task-forest bullets for ONE session, folded into its Details entry:
    the hierarchical T1→T1.1→T1→T2 timeline, a task legend (children indented under
    their parent), an interleaving callout when a top-level task recurs, and the
    whole-forest 'split into k sessions' headline. Returns [] if the session has no
    forest (offline run, or too short to judge). The session id / task summary are
    NOT repeated here — the Details block already prints them."""
    f = a.task_forest
    if not (f and f.get("timeline")):
        return []

    switches = f["num_switches"]
    lines = [f"- Forest: {_forest_timeline_str(f)} "
             f"({switches} switch{'es' if switches != 1 else ''})"]
    # legend: id -> label; sub-agent children (dotted ids) are indented under parents
    if f.get("tasks"):
        lines.append("- Tasks (independent) and their sub-agent children:")
        for task in f["tasks"]:
            tid = task.get("id", "?")
            indent = "        " if core.is_subtask(tid) else "    "
            kind_tag = " (sub-agent)" if core.is_subtask(tid) else ""
            lines.append(f"{indent}- `{tid}`{kind_tag} — {task.get('label','')}")
    # interleaving callout (returning to a top-level/main task)
    if f.get("recurring"):
        ids = ", ".join(f"`{t}`" for t in f["recurring"])
        lines.append(f"- ⚠️ Interleaving: returned to independent task {ids} after "
                     f"leaving it — these likely belonged in separate sessions.")
    # whole-forest headline saving
    if a.full_split:
        fs = a.full_split
        pct = 100 * fs["dollar_saving"] / fs["as_is_cost"] if fs["as_is_cost"] else 0.0
        lines.append(f"- Splitting the whole forest into {fs['num_chunks']} sessions "
                     f"≈ **${fs['dollar_saving']:.2f}** ({pct:.0f}%).")
    return lines


def render_report(analyses, suggestions, min_pct, min_dollars, used_llm):
    """Build the full Markdown report string. `used_llm` = an LLM endpoint was
    reachable, so the report has task summaries and the per-session task forest.

    The report is intentionally lean: it shows only the suggestions that clear both
    floors, ranked biggest-saving first, plus the task forest for those sessions. It
    does NOT dump every below-floor / non-best split option."""
    suggested = [s for s in suggestions if not s.informational]
    informational = [s for s in suggestions if s.informational]
    total_saving = sum(s.candidate.dollars for s in suggested)
    total_as_is = sum(a.as_is_cost for a in analyses)

    lines = []
    lines.append("# Local Split Advisor — session report\n")
    lines.append(f"- Sessions scanned: **{len(analyses)}**")
    lines.append(f"- Sessions worth splitting (≥ {min_pct:.0f}% and ≥ ${min_dollars:.2f}): "
                 f"**{len(suggested)}**")
    lines.append(f"- Total modelled cost of all sessions: **${total_as_is:,.2f}**")
    lines.append(f"- Total modelled saving from acting on suggestions: **${total_saving:,.2f}**")
    if informational:
        lines.append(f"- Sessions that already used plan mode (no action needed): "
                     f"**{len(informational)}**")
    detection = ("heuristic + LLM task-forest judge + task summaries" if used_llm
                 else "heuristic only (no LLM endpoint reachable)")
    lines.append(f"- Detection: {detection}")
    lines.append("")

    # ---- ranked suggestions table ----
    if suggested:
        lines.append("## Suggestions (ranked by modelled saving)\n")
        lines.append("| Session | Project | Agent steps | Peak | Pattern | Split point | Save $ | Save % |")
        lines.append("|---|---|---:|---:|:---:|---|---:|---:|")
        for s in sorted(suggested, key=lambda x: (x.candidate.dollars, x.candidate.pct), reverse=True):
            a, c = s.analysis, s.candidate
            lines.append(
                f"| `{a.session_id[:8]}` | {a.project[-28:]} | {a.turns} | "
                f"{_fmt_k(a.peak_context)} | {c.source} | {c.label} | "
                f"${c.dollars:.2f} | {c.pct:.0f}% |")
        lines.append("")

        # Details: one entry per suggested session, biggest saving first, each with
        # its full task forest folded in (no separate forest section).
        lines.append("### Details\n")
        for s in sorted(suggested, key=lambda x: (x.candidate.dollars, x.candidate.pct), reverse=True):
            a = s.analysis
            lines.append(f"**`{a.session_id}`** — {a.project}")
            if a.task_summary:
                lines.append(f"- Task: {a.task_summary}")
            lines.append(f"- {a.turns} agent steps, {a.tool_calls} tool calls, "
                         f"peak context ~{_fmt_k(a.peak_context)} tokens, "
                         f"as-is cost ${a.as_is_cost:.2f}")
            lines.append(f"- {s.message}")
            lines.extend(_forest_detail_lines(a))   # the task forest for this session
            lines.append("")
    else:
        lines.append("## Suggestions\n\nNo session cleared both thresholds. "
                      "Lower `--min-pct` / `--min-dollars` to see marginal opportunities.\n")

    # ---- informational (already handled) ----
    if informational:
        lines.append("## Already using plan mode\n")
        for s in informational:
            a = s.analysis
            lines.append(f"- `{a.session_id[:8]}` ({a.project[-28:]}): {s.message}")
        lines.append("")

    return "\n".join(lines)


def print_summary(analyses, suggestions, min_pct, min_dollars):
    suggested = [s for s in suggestions if not s.informational]
    total_saving = sum(s.candidate.dollars for s in suggested)
    print(f"\nScanned {len(analyses)} session(s).")
    print(f"Worth splitting (≥{min_pct:.0f}% and ≥${min_dollars:.2f}): {len(suggested)}")
    print(f"Total modelled saving if acted on: ${total_saving:,.2f}\n")
    for s in sorted(suggested, key=lambda x: (x.candidate.dollars, x.candidate.pct), reverse=True):
        a, c = s.analysis, s.candidate
        print(f"  • {a.session_id[:8]}  [{c.source}]  ${c.dollars:.2f} ({c.pct:.0f}%)")
        if a.task_summary:
            print(f"      task: {a.task_summary}")
        if a.task_forest and a.task_forest.get("timeline"):
            note = _forest_timeline_str(a.task_forest)
            if a.task_forest.get("recurring"):
                note += "  (interleaved)"
            print(f"      forest: {note}")
        print(f"      {s.message}")


# ======================================================================
# CLI
# ======================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Scan local Claude Code sessions and suggest worthwhile semantic splits.")
    p.add_argument("--projects-dir", default=DEFAULT_PROJECTS_DIR,
                   help="root of Claude Code project transcripts (default: ~/.claude/projects)")
    p.add_argument("--project", default=None,
                   help="only sessions whose project dir name contains this substring")
    p.add_argument("--min-pct", type=float, default=DEFAULT_MIN_PCT,
                   help=f"minimum saving %% to suggest a split (default: {DEFAULT_MIN_PCT})")
    p.add_argument("--min-dollars", type=float, default=DEFAULT_MIN_DOLLARS,
                   help=f"minimum saving $ to suggest a split (default: {DEFAULT_MIN_DOLLARS})")
    p.add_argument("--alpha", type=float, default=core.SUMMARY_ALPHA,
                   help=f"carried-summary size as a fraction of the first chunk "
                        f"(default: {core.SUMMARY_ALPHA})")
    p.add_argument("--no-llm", action="store_true",
                   help="disable all LLM use (no task summaries, no task-switch judge); "
                        "structural heuristics only, fully offline")
    p.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL,
                   help="OpenAI-compatible endpoint for the LLM (default: $OPENAI_BASE_URL, "
                        "else the OpenAI SDK default). Point it at your own provider.")
    p.add_argument("--llm-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"model name for summaries + the judge (default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Markdown report path (default: {DEFAULT_OUT})")
    p.add_argument("--summary-only", action="store_true",
                   help="print the console summary only; do not write the report file")
    return p


def _explain_no_llm(err, model):
    """Print a friendly, actionable hint when the LLM client can't be built, so a
    first-time user knows exactly how to turn on task summaries + task-switch
    detection (rather than seeing a raw exception)."""
    msg = str(err).lower()
    missing_key = "api_key" in msg or "credential" in msg or "openai_api_key" in msg
    print("Note: no LLM endpoint configured — running structural-only "
          "(no task summaries, no task-switch detection).")
    if missing_key:
        print("  To enable them, set an API key for any OpenAI-compatible provider "
              "and re-run:")
        print("      export OPENAI_API_KEY=sk-...            # e.g. an OpenAI key")
        print("  For a non-OpenAI provider, also set the endpoint and model:")
        print("      export OPENAI_BASE_URL=https://your-endpoint/v1")
        print(f"      export SPLIT_ADVISOR_MODEL=your-model   # default: {model}")
    else:
        print(f"  (endpoint error: {err})")
    print("  Pass --no-llm to silence this and always run offline.")


def _serve_command(argv):
    """Handle `split-advisor serve [--host H] [--port N]` — launch the local web
    dashboard. Kept as a tiny separate parser so the main analysis CLI (and its
    flags) are untouched."""
    import argparse as _argparse
    p = _argparse.ArgumentParser(prog="split-advisor serve",
                                 description="Launch the local web dashboard.")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: localhost)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    args = p.parse_args(argv)
    from web.app import serve
    serve(host=args.host, port=args.port)


def main():
    # `serve` is a subcommand handled before argparse so the analysis flags below
    # stay exactly as they were; everything else falls through to the normal CLI.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _serve_command(sys.argv[2:])
        return

    args = build_arg_parser().parse_args()

    sessions = discover_sessions(args.projects_dir, args.project)
    if not sessions:
        print(f"No sessions found under {args.projects_dir}"
              + (f" matching '{args.project}'." if args.project else "."))
        return

    # By default an LLM client is built so every suggestion carries a task
    # summary AND task switches are detected (session IDs alone are meaningless).
    # --no-llm skips it; if the endpoint is simply unreachable we degrade to a
    # structural-only report rather than crashing.
    client = None
    if not args.no_llm:
        # Pre-flight diagnostics up front: shows whether the key was picked up and,
        # if not, exactly why (dotenv missing, no .env, key unset, ...).
        print_llm_diagnostics(args.llm_base_url, args.llm_model)
        try:
            client = make_llm_client(args.llm_base_url)
        except Exception as e:
            _explain_no_llm(e, args.llm_model)

    print(f"Analysing {len(sessions)} session(s)"
          + (" with LLM task summaries + task-forest judge…" if client
             else " (structural heuristics only; no LLM)…"))

    analyses = []
    judge_errors = 0
    first_judge_error = None
    for session_id, path, project in sessions:
        try:
            a = analyze_session(session_id, path, project, alpha=args.alpha,
                                client=client, model=args.llm_model)
            analyses.append(a)
            if a.llm_error:
                judge_errors += 1
                if first_judge_error is None:
                    first_judge_error = a.llm_error
        except Exception as e:
            print(f"  skipped {session_id[:8]}: {e}")

    # If the client built but every judge call failed, that is the classic
    # "key set but wrong endpoint/model/VPN" case — call it out explicitly rather
    # than letting the report quietly show no summaries.
    if client is not None and judge_errors:
        print(f"\n⚠ LLM client built, but {judge_errors} judge call(s) FAILED "
              f"— first error: {first_judge_error}")
        print("  The key was found but the endpoint/model rejected the calls. "
              "Check --llm-base-url and --llm-model match your provider "
              "(and any VPN the endpoint needs).")

    suggestions = [s for s in (decide_suggestion(a, args.min_pct, args.min_dollars)
                               for a in analyses) if s]

    print_summary(analyses, suggestions, args.min_pct, args.min_dollars)

    if not args.summary_only:
        report = render_report(analyses, suggestions, args.min_pct, args.min_dollars,
                               used_llm=client is not None)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
