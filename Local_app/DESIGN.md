# Claude Split Advisor

A local tool that scans your own Claude Code sessions, uses an LLM to segment each into a
hierarchical **forest of tasks**, and prices — with a token-cost model — where you could
have saved money: by splitting into separate sessions, offloading a tangent to a
sub-agent, or running the opening reading phase in plan mode. Results are shown as a
Markdown report and a local web dashboard. It runs against the transcripts Claude Code
already writes (`~/.claude/projects/**/*.jsonl`); nothing is uploaded, and with no API key
it still runs fully offline (structural heuristics only).

## Main ideas

**Message vs turn.** A *message* is one prompt you typed; a *turn* is one assistant step
(a tool call or reply). One message spans many turns, and the cost math runs in turns.

**Cost model (linear context ramp).** Context grows ~linearly from 0 to a peak across the
turns; every turn re-reads the whole context, so total cost ≈ the area under that ramp.
A split replaces one big triangle with smaller ones, minus the cost of a carried summary.

**Hierarchical task forest.** Every message gets a task id: top-level ids (`T1`, `T2`) are
independent goals that may recur (`T1 → T2 → T1`); dotted children (`T1.1`) are
related-but-tangential side-tasks — especially an interruption you returned to.

**Three kinds of split, each priced.**
- **Task switch** — a move to a new top-level task you never return from → start a fresh
  session there.
- **Sub-agent** — a `T1.x` child (or a structural mid-session reading burst) → offload the
  segment to an isolated sub-agent so the main thread never carries its context.
- **Plan mode** — a front-loaded reading phase at the session start → run that reading in
  plan mode next time.

These are **priced by different formulas**, both living in the math model: a task switch
(or a whole-forest split into separate sessions) is a *fresh-session* split — the context
triangle is cut into smaller independent triangles; a sub-agent uses an *excise-and-rejoin*
model — the segment runs in its own isolated context and the main thread resumes at its
pre-segment height, carrying back only a small output summary. So a sub-agent and a task
switch at the same point cost different amounts.

A split is only suggested when its modelled saving clears both a percentage and a dollar
floor.

**Incremental caching.** Analysis results are cached per session (keyed by the
transcript's file signature). On a dashboard refresh the app re-analyses only the sessions
that are new or have changed since last time and reloads everything else from the cache —
so it never re-spends LLM quota on sessions already seen.

## Files

- **`chunk_split_model.py`** — the pure pricing math: the linear-ramp cost model and the
  per-split / per-sub-agent token-and-dollar formulas. No I/O.
- **`session_core.py`** — the pure analysis core: tool categorisation, activity-phase
  detection, the task-forest prompt + hierarchy helpers, and the three saving functions.
  Shared by the CLI and the dataset script.
- **`split_advisor.py`** — the CLI: discovers sessions, runs the structural + LLM analysis
  into priced split candidates, gates them behind the floors, writes the report, and hosts
  the `serve` command for the dashboard.
- **`analysis_api.py`** — the cached orchestrator behind the web UI: analyses each session
  once and caches it by file signature, so a refresh re-analyses only new or changed
  sessions and never re-spends LLM quota on unchanged ones.
- **`web/app.py`** — a local FastAPI server exposing the cached analysis and an incremental
  refresh, and launching the dashboard on `localhost`.
- **`web/static/`** — the self-contained dashboard page: two tabs (split suggestions vs
  plan-mode opportunities), colored task-forest strips and phase bars, live thresholds, and
  a per-session graph of the context ramp with the task bands and split points marked.
- **`main.py`** — a separate legacy script that ran the original task-switch analysis over
  the SWE-chat dataset; reuses `session_core` but is not part of the app.

## Next steps

- **Validate on SWE-chat, one repo.** Run the analysis over the sessions of a single
  repository in the SWE-chat dataset to check the forest segmentation and the modelled
  savings against real, varied traces at scale.
- **Find the best-fit client.** Identify which kind of repository / workflow benefits most
  from the tool — the ideal use case where splitting, sub-agents, and plan mode save the
  most — so we know who it's for.
