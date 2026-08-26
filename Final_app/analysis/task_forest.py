"""Detect a session that actually pursued several independent tasks at once.

WHAT THIS DETECTS
------------------
Unlike a missed plan-mode opportunity (``analysis.plan_mode``), "independent
tasks in one session" has no structural shape a rule can catch: telling "two
genuinely unrelated goals" apart from "two phases of the same feature" needs
to understand what the user actually asked for. So this detector has **no
offline fallback** — with no LLM available, ``detect`` always returns
``None``. That is a real limitation, not an oversight: there is nothing else
this pattern could be priced or flagged from.

THE SEGMENTATION
-----------------
Ported near-verbatim from a prior prototype
(``Local_app/session_core.py``'s ``build_task_forest_prompt``/`top_level`/
``is_subtask``/``compress_timeline``/``detect_interleaving``, and
``Local_app/split_advisor.py``'s ``judge_task_forest``): one LLM call
segments every user prompt into a hierarchical task id — top-level ids
(``T1``, ``T2``, …) are independent goals, and an id may recur later if the
user returns to it; dotted ids (``T1.1``) are tangents within a parent task —
both self-contained detours AND, just as often, the user asking about,
clarifying, or verifying that same task's own work (verified against a real
session, 2026-08-22: a run of "explain this", "where is this in code", "did I
commit everything" after a feature was built was mis-judged as two new
top-level tasks before the prompt called this out explicitly). This detector
only prices and visualizes **top-level** switches (``_top_level_bands`` folds
a child dip like ``T1 → T1.1 → T1`` back into ``T1``'s band) — the
child/tangent pattern is a different, not-yet-built "sub-agent opportunity"
detector.

THE PRICING MODEL
------------------
Same unmodified ``analysis.chunk_split_model`` math ``analysis.plan_mode``
already uses, generalised from one cut to N cuts — a direct port of the
prototype's ``session_core.saving_for_multi_split``. Reuses
``plan_mode``'s own ``_session_buckets``/``_call_times``/``_main_thread_events``
helpers rather than duplicating them.

Where a cut is actually proposed is coarser than every band boundary,
though: ``_cluster_independent_spans`` prices a split only at a genuinely
independent task's start, never in the middle of a stretch of back-and-forth
between RECURRING tasks (``T1, T2, T1, T2`` prices as one span, not four tiny
ones nobody would want as four separate sessions — the whole point of
"recurring" is that the user kept coming back, so there was never a clean
handoff there). ``runs`` (the task-lane visualization and the problem's own
narrative) stays band-by-band regardless — only the pricing input is
clustered.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from Final_app.adapters import claude_code
from Final_app.analysis import chunk_split_model as csm
from Final_app.analysis import classify
from Final_app.analysis import plan_mode as pm
from Final_app.config import CACHE_DIR, JUDGE_MODEL, LLM_BASE_URL, LLM_TIMEOUT_S, chat_completion
from Final_app.ir.models import EV_USER, Event, Problem, Session

# ----------------------------------------------------------------------
# Tunable thresholds
# ----------------------------------------------------------------------

SUMMARY_ALPHA = pm.SUMMARY_ALPHA
MIN_PCT_SAVING = 10.0
MIN_DOLLAR_SAVING = 0.50

# The prototype's own gate for bothering to call the judge at all — below
# this a session's "story" is too short to plausibly be two goals.
MIN_PROMPTS_FOR_JUDGE = 4

# Need at least this many distinct TOP-LEVEL tasks for "independent tasks in
# one session" to mean anything; one task, however long, is not this problem.
MIN_TASKS = 2


# ----------------------------------------------------------------------
# Segmentation — ported from the prototype's session_core.py
# ----------------------------------------------------------------------

def top_level(task_id: str) -> str:
    """The top-level (independent) task an id belongs to: ``T1.2`` -> ``T1``."""
    return str(task_id).split(".", 1)[0]


def is_subtask(task_id: str) -> bool:
    """True if the id is a self-contained tangent within a parent, e.g. ``T1.2``."""
    return "." in str(task_id)


def _join_natural(items: list[str]) -> str:
    """``["T1"]`` -> ``"T1"``; ``["T1","T2"]`` -> ``"T1 and T2"``;
    ``["T1","T2","T3"]`` -> ``"T1, T2, and T3"``."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def compress_timeline(assignments: list[str]) -> list[tuple[str, int, int]]:
    """Collapse a per-message task-id list into consecutive runs.

    ``["T1","T1","T2","T1"]`` -> ``[("T1",1,2),("T2",3,3),("T1",4,4)]``, each
    run being ``(task_id, start_message, end_message)`` with 1-based message
    numbers.
    """
    runs: list[list] = []
    for i, task_id in enumerate(assignments, start=1):
        if runs and runs[-1][0] == task_id:
            runs[-1][2] = i
        else:
            runs.append([task_id, i, i])
    return [tuple(run) for run in runs]


def detect_interleaving(assignments: list[str]) -> dict:
    """Summarise the forest's shape from its per-message assignments.

    Returns ``{runs, num_switches, recurring}``: ``runs`` is
    ``compress_timeline``'s result; ``num_switches`` counts transitions
    between exact-id runs; ``recurring`` is the sorted TOP-LEVEL ids the user
    returned to (a child dip like ``T1 -> T1.1 -> T1`` still counts as one
    continuous stretch of ``T1``, not a return).
    """
    runs = compress_timeline(assignments)
    counts: dict[str, int] = {}
    prev = None
    for task_id, _, _ in runs:
        tl = top_level(task_id)
        if tl != prev:
            counts[tl] = counts.get(tl, 0) + 1
        prev = tl
    return {
        "runs": runs,
        "num_switches": max(len(runs) - 1, 0),
        "recurring": sorted(tl for tl, n in counts.items() if n > 1),
    }


def _cluster_independent_spans(
    bands: list[tuple[str, int, int]], recurring: set
) -> list[tuple[int, int]]:
    """Group ``bands`` (top-level, from ``_top_level_bands``) into the spans
    worth pricing a split at — one per GENUINELY independent task, but a
    whole stretch of back-and-forth between recurring tasks as a single
    span, never split at every switch within it.

    Splitting a real interleaving (``T1, T2, T1, T2``) at every switch prices
    four tiny chunks nobody would actually want as four separate sessions —
    the whole point of "recurring" is that the user kept coming back, so
    there was never a clean handoff to price in the first place. A switch
    INTO a task that never recurs again, though, is a real candidate: the
    user is genuinely done with whatever came before.

    Implemented as each top-level id's last-occurrence index acting as a
    "reach": while walking bands left to right, a recurring id currently in
    the open span extends how far that span must run (to its own last
    occurrence, and transitively to any other recurring id's last occurrence
    absorbed along the way) — the same shape as merging overlapping
    intervals. A span closes the moment the walk passes every currently
    known reach, and the next band starts a new one.

    Returns ``(start_band_index, end_band_index)`` pairs, inclusive, in
    order. Empty in, empty out.
    """
    if not bands:
        return []

    last_occurrence: dict[str, int] = {}
    for index, (task_id, _, _) in enumerate(bands):
        last_occurrence[task_id] = index

    spans: list[tuple[int, int]] = []
    start = 0
    reach = last_occurrence[bands[0][0]] if bands[0][0] in recurring else 0
    for index in range(1, len(bands)):
        if index > reach:
            spans.append((start, index - 1))
            start = index
            reach = last_occurrence[bands[index][0]] if bands[index][0] in recurring else index
        else:
            task_id = bands[index][0]
            if task_id in recurring:
                reach = max(reach, last_occurrence[task_id])
    spans.append((start, len(bands) - 1))
    return spans


def _top_level_bands(timeline: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Fold ``timeline`` runs by TOP-LEVEL id, merging a child dip back into
    its parent's band. This detector only visualizes independent top-level
    tasks, not the tangents within them.
    """
    bands: list[list] = []
    for task_id, start_msg, end_msg in timeline:
        tl = top_level(task_id)
        if bands and bands[-1][0] == tl:
            bands[-1][2] = end_msg
        else:
            bands.append([tl, start_msg, end_msg])
    return [tuple(band) for band in bands]


def build_task_forest_prompt(prompt_texts: list[str]) -> str:
    """The task-forest judge prompt. Pure string; no API call.

    Ported from the prototype's ``session_core.build_task_forest_prompt``, then
    hardened: the judge model silently drops messages from its own
    ``assignments`` array once a session runs long — verified against 9 real
    sessions, 2026-08-23, 7 of 9 came back short (one 48-message session
    returned only 39-41 assignments). Stating the exact expected count twice
    (once up front, once right before the schema) and asking the model to
    self-check its own count before answering fixed every case tested.
    ``detect`` still refuses anything short, per CLAUDE.md §5 — this just makes
    "short" rare instead of routine.
    """
    count = len(prompt_texts)
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(prompt_texts))
    return (
        f"Here are the user's {count} messages from one coding session, in order:\n\n"
        f"{numbered}\n\n"
        "Each numbered item is ALREADY ONE ATOMIC MESSAGE the user actually typed "
        "— never split one, never merge two into one task boundary. A task's span "
        "starts at one numbered message and ends at another; a boundary can fall "
        "only BETWEEN two numbered messages, never inside one.\n\n"
        f"There are EXACTLY {count} messages numbered above — count them yourself "
        "before answering. Assign EVERY one of them to a task, in order, using "
        "HIERARCHICAL task ids.\n\n"
        "Top-level tasks — `T1`, `T2`, `T3`, … — are GENUINELY INDEPENDENT goals "
        "(unrelated to each other). Number them in the order they first appear, and "
        "reuse the SAME id when the user RETURNS to that task later (e.g. T1 … T2 … "
        "T1).\n\n"
        "Sub-agent children — `T1.1`, `T1.2`, … — are ANY message that is TANGENT to "
        "task `T1` rather than a genuinely new goal. This covers two shapes: (a) a "
        "self-contained detour — a big read-only investigation, a side change; and "
        "(b) — just as common, and easy to miss — the user asking ABOUT, "
        "clarifying, verifying, or double-checking work a task already did: "
        "'explain how this works', 'where is this in the code', 'what does X "
        "mean', 'is this right', 'did I commit everything'. None of these ask for "
        "anything new — they are the user looking closer at `T1`'s own work, so "
        "they are `T1`'s children, never a new top-level task, no matter how many "
        "such messages run in a row. A tangent is still its own TASK — give it its "
        "own id and its own short label in \"tasks\" exactly like a top-level task "
        "— it is simply NESTED under the task it is tangent to, rather than "
        "counted as an independent goal.\n\n"
        "Crucially: if the user LEAVES a task to do some related work — including "
        "asking questions about it — and then COMES BACK to that task, the "
        "in-between work is a CHILD of it, NOT a new independent task — because "
        "returning proves it was a side-quest that a sub-agent could have handled.\n\n"
        "Do NOT create a new task (top-level OR child) for a different PHASE of the "
        "same work: design→plan, plan→implement, implement→explain/clarify, "
        "implement→document/write-up, implement→test/fix of the SAME feature, or "
        "just different files/words for the same goal.\n\n"
        "Only use a new TOP-LEVEL id when the goal genuinely changes to something "
        "unrelated the user does NOT return from — not merely because the "
        "conversation moved from doing the work to talking about it.\n\n"
        "BEFORE you finalize, re-check every top-level id against the messages "
        "AFTER it: if the surrounding task's goal shows up again later, whatever "
        "you assigned in between was a side-quest, not a new goal — go back and "
        "renumber it as a CHILD of the task it interrupted, not a new top-level "
        "id.\n\n"
        "Answer with ONLY a JSON object, no other text:\n"
        '{"tasks": [{"id": "T1", "label": "<short concrete name>"}, '
        '{"id": "T1.1", "label": "<the tangent/side-task>"}, …], '
        '"assignments": ["T1", "T1.1", "T1", "T2", …], '
        '"summary": "<one short, concrete sentence naming the session\'s overall '
        'work>"}\n\n'
        f'"assignments" MUST have EXACTLY {count} entries — one id per message, in '
        "the same order as the numbered list above — and every id must appear in "
        '"tasks". Before you output, count the entries in your own "assignments" '
        f"array and confirm it is {count}; if it is not, fix it before answering.\n\n"
        "Output MINIFIED JSON: a single line, no spaces, no newlines, no markdown "
        "fences. Keep every \"label\" to at most 5 words."
    )


# ----------------------------------------------------------------------
# The judge — Final_app's optional/cached/degrade shape
# (same conventions as analysis.classify.judge_calls / analysis.plan_mode.justify)
# ----------------------------------------------------------------------

_CACHE_FILE = CACHE_DIR / "task_forest.json"
_cache: Optional[dict] = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(_cache, dict):
            _cache = {}
    except (OSError, json.JSONDecodeError):
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp = _CACHE_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(_load_cache(), indent=0), encoding="utf-8")
        temp.replace(_CACHE_FILE)
    except OSError:
        pass


def _forest_cache_key(prompt_texts: list[str]) -> str:
    """Content-addressed on the FULL rendered prompt, not just the raw messages.

    Hashing ``build_task_forest_prompt(prompt_texts)`` rather than
    ``prompt_texts`` means an edit to the judge prompt's wording naturally
    invalidates every prior verdict — a session judged under an old prompt
    version is never silently served that stale answer once the prompt (and
    therefore what "correct" means) has changed.
    """
    payload = json.dumps([build_task_forest_prompt(prompt_texts), JUDGE_MODEL],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def judge_task_forest(prompt_texts: list[str]) -> Optional[dict]:
    """One LLM call segmenting the session's prompts into a task forest.

    Returns the parsed ``{tasks, assignments, summary}`` dict, or ``None`` —
    never raised — on any failure: LLM unavailable, network/auth error, or a
    response that doesn't parse as JSON.

    Only a SUCCESSFUL judgment is cached. A failure (a timeout, in particular)
    used to be written to the cache as ``None`` just the same as a genuine
    "no split-worthy pattern here" verdict — permanently, since the cache is
    checked before any call is made. Against a reasoning model whose calls can
    legitimately take 40s+, that baked in real timeouts as false negatives
    forever; not caching failures means a timed-out session is simply retried
    on the next scan instead.
    """
    if not classify.llm_available():
        return None

    cache = _load_cache()
    key = _forest_cache_key(prompt_texts)
    if key in cache:
        return cache[key]

    forest: Optional[dict] = None
    try:
        from openai import OpenAI

        client = (OpenAI(base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_S)
                  if LLM_BASE_URL else OpenAI(timeout=LLM_TIMEOUT_S))
        response = chat_completion(
            client,
            model=JUDGE_MODEL, max_tokens=8000, temperature=0,
            messages=[{"role": "user", "content": build_task_forest_prompt(prompt_texts)}],
        )
        text = (response.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("assignments"), list):
            forest = parsed
    except Exception:
        forest = None

    if forest is not None:
        cache[key] = forest
        _save_cache()
    return forest


# ----------------------------------------------------------------------
# Pricing — a direct port of the prototype's saving_for_multi_split
# ----------------------------------------------------------------------

def price_multi_split(session: Session, fractions: list[float],
                      alpha: float = SUMMARY_ALPHA) -> Optional[dict]:
    """Price cutting ``session`` at every fraction in ``fractions`` at once.

    The N-way generalisation of ``plan_mode.price_split``: builds arbitrary
    chunk heights from the (deduped, sorted) cut fractions along the linear
    context ramp, using the same unmodified ``chunk_split_model`` functions
    for ``split_cost`` — priced at ``pm._rates_for_session(session)``, this
    session's own real per-model rate, plus any subagent spend
    (``pm._subagent_cost``) added on top, since a subagent's own cost does
    not change when the main thread's tasks are split apart.

    ``as_is_cost`` is simply ``session.attributed_cost`` — the same exact
    figure "Retrospective cost" shows, not a second estimate that can
    disagree with it. See ``plan_mode.price_split`` for the full reasoning.
    Returns ``None`` when the session doesn't fit the ramp model, or no
    fraction survives the 5%/95% degenerate-split guard.
    """
    input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls = pm._session_buckets(session)
    if cache_read_tok <= 0 or api_calls < 2:
        return None

    seen: set[float] = set()
    fracs: list[float] = []
    for fraction in sorted(fractions):
        if not (0.05 < fraction < 0.95):
            continue
        key = round(fraction, 4)
        if key not in seen:
            seen.add(key)
            fracs.append(fraction)
    if not fracs:
        return None

    peak = csm.peak_context_tokens(cache_read_tok, api_calls)
    if peak <= 0:
        return None

    cuts = [0.0] + [peak * f for f in fracs] + [peak]
    heights = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
    if any(h <= 0 for h in heights):
        return None

    rates = pm._rates_for_session(session)
    subagent_cost = pm._subagent_cost(session)
    calls = csm.calls_per_chunk(heights, api_calls, peak)
    cache_read_split = csm.cache_read_after_split(heights, api_calls, peak)

    overhead = 0.0
    for j in range(1, len(heights)):
        summary_tok = csm.summary_tokens(heights[j - 1], alpha)
        chunk_overhead, *_ = csm.summary_overhead_cost([calls[j]], summary_tok, rates)
        overhead += chunk_overhead

    as_is = session.attributed_cost
    split = (csm.base_cost_after_split(input_tok, output_tok, cache_write_tok,
                                       cache_read_split, rates)
             + overhead + subagent_cost)
    dollar_saving = as_is - split
    percent_saving = (100.0 * dollar_saving / as_is) if as_is else 0.0

    return {
        "as_is_cost": as_is,
        "split_cost": split,
        "dollar_saving": dollar_saving,
        "percent_saving": percent_saving,
        "cache_read_before": cache_read_tok,
        "cache_read_after": cache_read_split,
        "token_saving": cache_read_tok - cache_read_split,
        "num_chunks": len(heights),
    }


# ----------------------------------------------------------------------
# The public entry point
# ----------------------------------------------------------------------

def _human_prompt_events(session: Session) -> list[Event]:
    """The Events ``session.user_prompts`` was built from, in the same order.

    ``Session.user_prompts`` carries no positional link back to ``events`` —
    it is built in ``adapters.claude_code.parse_transcript`` in the same pass
    and order as these Events, so prompt *i* is always this list's *i*-th
    entry's text, positionally, with no stored id to look up instead.
    """
    return [e for e in session.events if e.type == EV_USER and e.is_human_prompt]


def _fraction_at(call_times: list, ts) -> float:
    """Where ``ts`` falls among ``call_times``, as a fraction — mirrors
    ``plan_mode._split_fraction_for``'s own timestamp-to-fraction mapping."""
    if not call_times:
        return 0.0
    passed = sum(1 for call_ts in call_times if call_ts <= ts)
    return passed / len(call_times)


def detect(session: Session) -> Optional[Problem]:
    """One ``Problem`` for independent tasks pursued in one session, or ``None``.

    Has no offline fallback: with no LLM available, or too few prompts to
    plausibly hold two goals, or fewer than ``MIN_TASKS`` distinct top-level
    tasks in the judge's own answer, this returns ``None``.
    """
    prompt_events = _human_prompt_events(session)
    if len(prompt_events) < MIN_PROMPTS_FOR_JUDGE:
        return None

    # ``is_human_prompt`` only checks that a line's TEXT STARTS WITH real
    # typed content, so a line where genuine text is followed by injected
    # machinery in the same line (e.g. a trailing ``<system-reminder>``)
    # still counts as human — correctly — but the injected tail must not
    # ride along into what the judge is told a person said.
    prompt_texts = [claude_code.strip_injected_machinery(e.text) for e in prompt_events]
    forest = judge_task_forest(prompt_texts)
    if forest is None:
        return None

    assignments = (forest.get("assignments") or [])[:len(prompt_events)]
    if len(assignments) < len(prompt_events):
        # A malformed or truncated response — degrade rather than guess at
        # which messages the missing assignments belonged to.
        return None

    bands = _top_level_bands(compress_timeline(assignments))
    if len({tl for tl, _, _ in bands}) < MIN_TASKS:
        return None

    interleave = detect_interleaving(assignments)
    prompt_ts = [e.ts for e in prompt_events]
    call_times = pm._call_times(pm._main_thread_events(session))

    # ``runs`` stays one entry per BAND — the task lane still shows every
    # switch the judge found, granular, for transparency. Only the PRICING
    # input is coarser: a split fraction only at a genuinely independent
    # task's start, never in the middle of a stretch of interleaving. See
    # ``_cluster_independent_spans``.
    runs: list[dict] = []
    labels = {task.get("id"): task.get("label", "") for task in forest.get("tasks", [])}
    for index, (task_id, start_msg, end_msg) in enumerate(bands):
        start_ts = prompt_ts[start_msg - 1]
        end_ts = prompt_ts[bands[index + 1][1] - 1] if index + 1 < len(bands) else None
        runs.append({
            "id": task_id,
            "label": labels.get(task_id, task_id),
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat() if end_ts else None,
        })

    spans = _cluster_independent_spans(bands, set(interleave["recurring"]))
    fractions = [
        _fraction_at(call_times, prompt_ts[bands[start_band][1] - 1])
        for start_band, _end_band in spans[1:]
    ]

    # What each PRICED chunk actually contains — one entry per span, so a
    # chunk that merged an interleaved stretch reads as "T1 and T2", not
    # silently as just whichever id happened to start it.
    chunks: list[dict] = []
    for start_band, end_band in spans:
        task_ids = list(dict.fromkeys(
            bands[i][0] for i in range(start_band, end_band + 1)))
        chunks.append({
            "task_ids": task_ids,
            "label": _join_natural(task_ids),
        })

    saving = price_multi_split(session, fractions)
    if saving is None:
        return None
    if (saving["percent_saving"] < MIN_PCT_SAVING
            or saving["dollar_saving"] < MIN_DOLLAR_SAVING):
        return None

    # Distinct top-level tasks, not bands: a recurring task's second (or
    # third) stretch is the SAME task returned to, not another independent
    # one — counting bands here would report "6 independent tasks" for a
    # session that actually pursued 4, with 2 interleaved returns.
    distinct_tasks = [{"id": task_id, "label": labels.get(task_id, task_id)}
                      for task_id in dict.fromkeys(tl for tl, _, _ in bands)]
    task_names = ", ".join(f"{task['id']} ({task['label']})" for task in distinct_tasks)
    # ``num_chunks`` is the number of SPANS actually priced
    # (``_cluster_independent_spans``), not the number of switches — a
    # stretch of back-and-forth between recurring tasks prices as one chunk,
    # so this must say how many chunks, never "at each switch" (stale
    # wording from before clustering existed, and actively wrong now: a
    # session can have far more switches than priced chunks).
    num_chunks = saving["num_chunks"]
    chunk_note = (
        " (keeping each returned-to task's back-and-forth together)"
        if interleave["recurring"] else ""
    )
    detail = (
        f"This session pursued {len(distinct_tasks)} independent tasks — "
        f"{task_names}. Splitting into {num_chunks} separate sessions"
        f"{chunk_note} could have saved an estimated "
        f"{saving['percent_saving']:.0f}% (${saving['dollar_saving']:.2f})."
    )

    return Problem(
        id="task-switch",
        title="Independent tasks in one session",
        severity="high" if interleave["recurring"] else "medium",
        detail=detail,
        data={
            **saving,
            "tasks": distinct_tasks,
            "runs": runs,
            "chunks": chunks,
            "recurring": interleave["recurring"],
            "num_switches": interleave["num_switches"],
        },
    )
