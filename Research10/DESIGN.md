# Split Advisor — Design & Report

## 0. Introduction — the idea

The earlier project proved one thing: most long Claude Code sessions open with a
reading phase that plan mode could have handled (73% of long sessions). But it
reported this as a *retrospective dollar figure* — small, and already spent.

The new idea: use that same signal *prospectively*. Mine the user's past sessions to
(1) suggest plan mode for future work, grounded in their own history, and (2) name the
*root cause* — why that session needed so much upfront reading — so the advice is an
explanation, not a nag.

**Build now:** a pre-session briefing with root-cause explanation. **Later:** a live
in-terminal nudge. **Non-goals:** GUI, uploads, pricing changes.

---

## 1. What the earlier CLI already does

A working, tested CLI ([`split_advisor.py`](split_advisor.py)) that discovers every local
session, detects its shape (front-loaded reading → plan mode, or mid-session burst →
sub-agent), prices the best split, gates tiny sessions behind a floor, and reports.
The engine is pure/offline; an optional `--llm` judge is opt-in.

**Honest result:** of 32 sessions, only 2–3 split candidates, all saving 5–8% — below
the 10% floor. Structural reason: the reading phase ends early, so a single split only
removes a small fraction of re-read cost. **Sessions are long because the task is big,
not because they're splittable.** Hence the reframe: the dollar saving is a weak
headline, but the *pattern* is a strong per-user signal worth surfacing as advice.

It reads the standard Claude Code transcript format (one JSONL per session), parsing
token usage, the ordered tool-call sequence, and plan-mode events — version-tolerant and
fully offline.

---

## 2. What's new — the plan

Shift from "here's what a split would have saved" to "here's why your sessions need plan
mode, and what to do next time." Three stages:

- **Stage 1 — Detect.** Reuse existing analysis; keep the plan-mode (Pattern A) sessions.
- **Stage 2 — Root-cause (the new core).** From file paths and search patterns we already
  parse, compute signals per session and map them to root-cause labels (table below).
- **Stage 3 — Briefing (the deliverable).** A `brief` subcommand, per project, that
  aggregates the above into actionable advice: how often it happens, the dominant root
  cause, the files you *always* open first, and next-session advice ("start in plan mode
  and name A, B, C").

Heuristics compute the signals offline; an optional `--llm` upgrades a label into a
written narrative *why*. Off by default.

### Signals

| Signal | How it's computed | What it tells us |
|---|---|---|
| `reads_before_first_edit` | length of the opening exploration phase | how big the missed plan-mode window was |
| `distinct_files_read` vs `files_edited` | unique files read before first edit, vs edited | read 15, edited 2 → over-exploration |
| `re_reads` | same file read more than once before editing | exploration **thrashing**, not just volume |
| `prompt_named_files` | did the opening prompt contain the paths the agent read? | were reads *locating* work the prompt could have avoided? |

### Root-cause labels

| Label | Signal pattern | Prospective fix |
|---|---|---|
| **Under-specified opening prompt** | agent read files the prompt never named | name the files / start in plan mode |
| **Large surface area** | many distinct files, few re-reads | genuine complexity — offload reading to plan mode / a sub-agent |
| **Exploration thrashing** | high re-reads of the same files | the reading was inefficient, not merely large |

---

## 3. Next idea — live mode (future)

A `watch` command tailing the active session, detecting the same front-loaded reading
shape as it forms and nudging while plan mode is still actionable. Deferred because the
pre-session briefing already puts the advice where the user can act.

---

## 4. Build order

1. **File-path extraction** in [`session_core.py`](session_core.py) → the Stage-2 signals.
2. **Root-cause labeler** (heuristic) + optional `--llm` narrative.
3. **`brief` subcommand** — per-project aggregation with global fallback.

---

## 5. Design decisions & open questions

**Decided:** heuristic labels by default, `--llm` narrative opt-in; per-project briefing
(global fallback); include the recurring-opening-files list; live mode parked.

**Open:**
- **α calibration** — the 0.10 summary size is a hardcoded guess; a `--calibrate` mode
  measuring real summary sizes would harden the dollar claims.
- **Prompt-named-files detection** — regex path-matching is crude; is it enough offline,
  with `--llm` for the semantic cases?
- **Cross-session context** — some opening reads re-establish what a *previous* session
  saw; detecting it needs session linking, and may be a distinct fourth label.
