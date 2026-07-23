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
whether the user genuinely switched tasks mid-session — considering a split at that
switch too, keeping whichever candidate saves more — and labels every considered
split with a one-line summary of what the session was about (session IDs alone are
meaningless). `--no-llm` turns all of that off for a fully offline structural run.

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

def decode_project(dir_name):
    """Claude Code encodes a project's absolute path by replacing '/' with '-'.
    Decode it back to something readable for the report (best-effort: real
    dashes in path segments are indistinguishable from separators)."""
    return dir_name.replace("-", "/")


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
    source: str                 # "plan-mode" | "sub-agent" | "task-switch"
    split_fraction: float
    saving: dict                # from core.saving_for_split
    label: str                  # short human description of WHERE the split is
    detail: str = ""            # extra context (e.g. reason / call count)

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
        label=f"after turn {first['end_turn']} (end of opening reading phase)",
        detail=f"~{first['length']} reads before the first edit",
    )


def _subagent_candidate(buckets, seq_turns, expl_phases, alpha):
    """Pattern C: split at the start of the mid-session reading burst."""
    burst = expl_phases[0]
    frac = burst["start_turn"] / seq_turns if seq_turns else 0.0
    saving = core.saving_for_split(buckets, frac, alpha)
    if not saving:
        return None
    return Candidate(
        source="sub-agent",
        split_fraction=frac,
        saving=saving,
        label=f"at turn {burst['start_turn']} (mid-session reading burst)",
        detail=f"~{burst['length']}-call reading burst that a sub-agent could handle",
    )


def _task_switch_candidate(result, buckets, prompts, total_turns, alpha):
    """--llm only: build a split Candidate from a judge result, or None.
    `result` is the parsed judge dict (already fetched by the caller so the
    same LLM call can also supply the session summary)."""
    if not result or "error" in result or not result.get("has_switch"):
        return None
    switch_num = result.get("switch_message_number")
    if not (isinstance(switch_num, int) and 1 <= switch_num <= len(prompts)):
        return None
    switch_turn = prompts[switch_num - 1][0]
    frac = switch_turn / total_turns
    saving = core.saving_for_split(buckets, frac, alpha)
    if not saving:
        return None
    return Candidate(
        source="task-switch",
        split_fraction=frac,
        saving=saving,
        label=f"at message {switch_num} (turn {switch_turn}, task switch)",
        detail=result.get("reason", "").strip(),
    )


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

    judge_result = None
    prompts = []
    if client is not None:
        prompts, total_turns = core.user_prompts_with_turns(events)
        # The judge runs on sessions with enough prompts; ask it for the summary
        # in the SAME call so those sessions cost no extra request.
        if len(prompts) >= 4 and total_turns > 0:
            judge_result = judge_task_switch(client, [t for _, t in prompts],
                                             model=model, with_summary=True)
            c = _task_switch_candidate(judge_result, buckets, prompts,
                                       total_turns, alpha)
            if c:
                candidates.append(c)

    analysis.candidates = candidates
    if candidates:
        analysis.best = max(candidates, key=lambda c: c.dollars)

    # Chosen scope: only summarise sessions that produced a split option.
    if client is not None and candidates:
        analysis.task_summary = _session_summary(client, model, prompts, judge_result)
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
        return (f"Has a {cand.detail}; offloading it to a sub-agent instead of the "
                f"main model would save ~${cand.dollars:.2f} ({cand.pct:.0f}%).")
    if cand.source == "task-switch":
        reason = f" ({cand.detail})" if cand.detail else ""
        return (f"Switched to an unrelated task{reason}; starting a fresh session "
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


def judge_task_switch(client, prompt_texts, model=DEFAULT_JUDGE_MODEL,
                      with_summary=False):
    """One LLM call judging whether the user switches tasks (optionally also
    returning a one-line task summary). Returns a dict or {'error': ...}."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=300 if with_summary else 200,
            temperature=0,
            messages=[{"role": "user",
                       "content": core.build_judge_prompt(prompt_texts, with_summary)}],
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


def _candidate_status(analysis, cand, min_pct, min_dollars):
    """Short status for one candidate in the 'all considered options' table.
    Only the best candidate that clears BOTH floors is ever suggested."""
    is_best = analysis.best is cand
    clears = cand.pct >= min_pct and cand.dollars >= min_dollars
    if is_best and clears:
        if cand.source == "plan-mode" and analysis.already_plan_mode:
            return "already plan mode"
        return "✅ suggested"
    if clears:
        return "clears floor (not best)"
    return "below floor"


def render_report(analyses, suggestions, min_pct, min_dollars, used_llm):
    """Build the full Markdown report string. `used_llm` = an LLM endpoint was
    reachable, so the report has task summaries and task-switch options."""
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
    detection = ("heuristic + LLM task-switch judge + task summaries" if used_llm
                 else "heuristic only (no LLM endpoint reachable)")
    lines.append(f"- Detection: {detection}")
    lines.append("")

    # ---- ranked suggestions table ----
    if suggested:
        lines.append("## Suggestions (ranked by modelled saving)\n")
        lines.append("| Session | Project | Turns | Peak | Pattern | Split point | Save $ | Save % |")
        lines.append("|---|---|---:|---:|:---:|---|---:|---:|")
        for s in sorted(suggested, key=lambda x: x.candidate.dollars, reverse=True):
            a, c = s.analysis, s.candidate
            lines.append(
                f"| `{a.session_id[:8]}` | {a.project[-28:]} | {a.turns} | "
                f"{_fmt_k(a.peak_context)} | {c.source} | {c.label} | "
                f"${c.dollars:.2f} | {c.pct:.0f}% |")
        lines.append("")

        lines.append("### Details\n")
        for s in sorted(suggested, key=lambda x: x.candidate.dollars, reverse=True):
            a = s.analysis
            lines.append(f"**`{a.session_id}`** — {a.project}")
            if a.task_summary:
                lines.append(f"- Task: {a.task_summary}")
            lines.append(f"- {a.turns} turns, {a.tool_calls} tool calls, "
                         f"peak context ~{_fmt_k(a.peak_context)} tokens, "
                         f"as-is cost ${a.as_is_cost:.2f}")
            lines.append(f"- {s.message}")
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

    # ---- everything scanned, for transparency ----
    lines.append("## All scanned sessions\n")
    lines.append("| Session | Turns | Peak | Pattern | Modelled saving | Suggested? |")
    lines.append("|---|---:|---:|:---:|---:|:---:|")
    suggested_ids = {s.analysis.session_id for s in suggested}
    for a in sorted(analyses, key=lambda x: x.as_is_cost, reverse=True):
        if not a.modelled:
            save_str, mark = "n/a (no cache-read)", "—"
        elif a.best:
            save_str = f"${a.best.dollars:.2f} ({a.best.pct:.0f}%)"
            mark = "✅" if a.session_id in suggested_ids else "below floor"
        else:
            save_str, mark = "no split point", "—"
        lines.append(f"| `{a.session_id[:8]}` | {a.turns} | {_fmt_k(a.peak_context)} | "
                     f"{a.pattern} | {save_str} | {mark} |")
    lines.append("")

    # ---- every option considered, incl. below-floor & non-best (transparency) ----
    lines.append("## All considered split options\n")
    note = ("Every split point the tool priced, including options below the "
            "suggestion floor and options that were not the best for their session. ")
    note += ("LLM-detected task switches appear here too."
             if used_llm else
             "(No LLM endpoint reachable, so LLM task switches are not included.)")
    lines.append(note + "\n")
    # The "Task" column (LLM summary of what the session was about) appears
    # whenever an LLM endpoint was reachable — session IDs alone say nothing.
    task_col = " Task |" if used_llm else ""
    task_sep = ":---|" if used_llm else ""
    lines.append(f"| Session |{task_col} Option | Split point | Save $ | Save % | Status |")
    lines.append(f"|---|{task_sep}:---:|---|---:|---:|:---:|")
    any_cand = False
    for a in sorted(analyses, key=lambda x: x.as_is_cost, reverse=True):
        task_cell = f" {a.task_summary or '—'} |" if used_llm else ""
        for c in sorted(a.candidates, key=lambda x: x.dollars, reverse=True):
            any_cand = True
            status = _candidate_status(a, c, min_pct, min_dollars)
            lines.append(f"| `{a.session_id[:8]}` |{task_cell} {c.source} | {c.label} | "
                         f"${c.dollars:.2f} | {c.pct:.0f}% | {status} |")
    if not any_cand:
        empty = f"| — |{' — |' if used_llm else ''} — | no priceable split points found in any session | — | — | — |"
        lines.append(empty)
    lines.append("")

    return "\n".join(lines)


def print_summary(analyses, suggestions, min_pct, min_dollars):
    suggested = [s for s in suggestions if not s.informational]
    total_saving = sum(s.candidate.dollars for s in suggested)
    print(f"\nScanned {len(analyses)} session(s).")
    print(f"Worth splitting (≥{min_pct:.0f}% and ≥${min_dollars:.2f}): {len(suggested)}")
    print(f"Total modelled saving if acted on: ${total_saving:,.2f}\n")
    for s in sorted(suggested, key=lambda x: x.candidate.dollars, reverse=True):
        a, c = s.analysis, s.candidate
        print(f"  • {a.session_id[:8]}  [{c.source}]  ${c.dollars:.2f} ({c.pct:.0f}%)")
        if a.task_summary:
            print(f"      task: {a.task_summary}")
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


def main():
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
        try:
            client = make_llm_client(args.llm_base_url)
        except Exception as e:
            _explain_no_llm(e, args.llm_model)

    print(f"Analysing {len(sessions)} session(s)"
          + (" with LLM task summaries + task-switch judge…" if client
             else " (structural heuristics only; no LLM)…"))

    analyses = []
    for session_id, path, project in sessions:
        try:
            analyses.append(analyze_session(session_id, path, project,
                                             alpha=args.alpha, client=client,
                                             model=args.llm_model))
        except Exception as e:
            print(f"  skipped {session_id[:8]}: {e}")

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
