"""Detect a front-loaded reading phase that could have run in plan mode.

WHAT THIS DETECTS
------------------
A session that opens with a long run of ``read`` blocks before any real edit
or command, and never entered Claude Code's plan mode for it, paid to carry
that whole reading phase's content into every later call's context.

IMPORTANT CAVEAT, verified against the Claude Code source
(``claude-code-source-code/src/tools/ExitPlanModeTool/ExitPlanModeV2Tool.ts``):
by DEFAULT, exiting plan mode is a pure permission-mode flip — the plan text
is added on top of the existing conversation, and everything read during
planning stays in context exactly as it would outside plan mode, still billed
on every later call. The "only a short plan carries forward" behaviour this
detector prices is real, but gated behind a non-default setting
(``showClearContextOnPlanAccept``) that clears context on plan approval and
starts a fresh session — off unless the user has turned it on. So the priced
saving is what plan mode *plus clearing context on approval* would have
bought, not what toggling plan mode alone does. ``detect``'s ``detail`` text
says this explicitly rather than overpromising.

THE PRICING MODEL
------------------
This module does none of the pricing math itself. ``analysis.chunk_split_model``
is a prior prototype's linear-context-ramp cost model
(``Local_app/chunk_split_model.py``), carried over unmodified — this file only
*adapts* Final_app's own ``Session``/``Event`` IR into the plain
``(input, output, cache_write, cache_read, api_calls)`` bucket that model's
functions expect, the same way that prototype's own ``session_core.py`` adapted
raw transcript events into it. ``price_split`` below is a direct, otherwise
unmodified port of that prototype's ``saving_for_split`` — with one exception:
it now passes ``_rates_for_session(session)`` instead of that model's built-in
flat, model-agnostic default (a blended Opus-4.x guess) into every
``chunk_split_model`` call, so ``as_is_cost`` prices this session's own real
model and agrees with ``analysis.attribution``'s "Retrospective cost" instead
of disagreeing with it by the ratio between the two rates.

Only a single split (this detector never proposes more than one cut) is
needed here, so the general N-way chunking and the sub-agent excise-and-rejoin
model in ``chunk_split_model`` go unused — both are for other problem types.

JUSTIFYING THE SUGGESTION
--------------------------
The pattern match and the priced saving explain the *mechanism* ("you read
for N steps before any edit"), but not why THIS reading looked like planning
rather than work that would have happened in any mode. ``justify`` asks an
LLM, grounded in the actual file paths/searches the flagged block touched and
the task's first prompt, for one concrete sentence — the same optional,
cache-and-degrade shape as ``analysis.classify``'s Stage B judge for
ambiguous Bash commands: skipped with no key/network, cached on disk by
content hash, and never load-bearing for detection itself (a session is
flagged from the structural pattern and the price alone; the justification
only adds colour to something already decided).
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from Final_app.analysis import chunk_split_model as csm
from Final_app.analysis import classify
from Final_app.analysis.pricing import price_for_model
from Final_app.config import CACHE_DIR, JUDGE_MODEL, LLM_BASE_URL, LLM_TIMEOUT_S, chat_completion
from Final_app.ir.models import (
    EV_ASSISTANT,
    EV_TOOL_USE,
    EXECUTE,
    READ,
    WRITE,
    Block,
    Event,
    Problem,
    Session,
)

# ----------------------------------------------------------------------
# Tunable thresholds
# ----------------------------------------------------------------------

# Carried over from the prototype's ``session_core.SUMMARY_ALPHA``: the carried
# summary is modelled as this fraction of the first chunk's peak height.
SUMMARY_ALPHA = 0.10

# A split is only suggested when it clears both floors — otherwise a tiny
# session would get nagged over a saving nobody would act on. Lowered from
# the prototype's original 10.0 after a real session came back at 9.99% —
# a real $2.93 saving excluded by 0.014 percentage points. 9% still filters
# genuinely negligible cases while catching that one.
MIN_PCT_SAVING = 9.0
MIN_DOLLAR_SAVING = 0.50

# Final_app's Block-granularity stand-ins for the prototype's raw-tool-call
# ``MIN_PHASE_LENGTH`` (5) and its 20%-of-the-session start gate. A Block here
# already merges a whole run of same-kind calls, so the same absolute numbers
# do not carry over unchanged — these are a first approximation, not measured
# against real transcripts.
MIN_READ_BLOCK_MESSAGES = 3
START_WITHIN_FRACTION = 0.20


# ----------------------------------------------------------------------
# Adapting Session/Event into the math model's plain token buckets
# ----------------------------------------------------------------------

def _main_thread_events(session: Session) -> list[Event]:
    """The main thread only — a subagent bills and ramps independently."""
    return [e for e in session.events if e.agent_id is None]


def _call_times(events: list[Event]) -> list:
    """One timestamp per API call (first-seen), oldest first.

    Grouped by ``message_id`` the same way ``analysis.attribution._calls_for``
    groups a call — one call is written as one line per content block, all
    sharing that id — but only the timestamps are needed here, so this stays a
    small local pass rather than importing that module's private helper.
    """
    seen: dict[str, object] = {}
    for event in events:
        if event.type not in (EV_ASSISTANT, EV_TOOL_USE):
            continue
        key = event.message_id or f"uuid:{event.uuid}"
        if key not in seen:
            seen[key] = event.ts
    return sorted(seen.values())


def _session_buckets(session: Session) -> tuple[int, int, int, int, int]:
    """``(input, output, cache_write, cache_read, api_calls)`` for the main
    thread, matching the shape ``chunk_split_model``'s functions expect.
    """
    main_events = _main_thread_events(session)
    input_tok = sum(e.tokens.input for e in main_events)
    output_tok = sum(e.tokens.output for e in main_events)
    cache_write_tok = sum(e.tokens.cache_creation for e in main_events)
    cache_read_tok = sum(e.tokens.cache_read for e in main_events)
    api_calls = len(_call_times(main_events))
    return input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls


def _rates_for_session(session: Session) -> csm.Rates:
    """This session's real per-token rates, in ``chunk_split_model``'s shape.

    ``chunk_split_model`` was carried over with one hardcoded, model-agnostic
    rate table (``csm.DEFAULT_RATES``, a blended Opus-4.x guess) — pricing
    every session identically regardless of which model actually ran it, and
    disagreeing with ``analysis.attribution``'s real per-model ledger
    ("Retrospective cost") by exactly the ratio between the two rates. This
    converts ``pricing.price_for_model``'s real $/token rates into the
    $/MTok-plus-multiplier shape ``chunk_split_model`` expects, so both
    pricing paths agree for a single-model session.

    ``chunk_split_model`` prices a session as one lump sum, not call by call
    like ``analysis.attribution`` does, so a session that switched models
    mid-way is still priced at one blended rate here — ``session.model``,
    the same single label already shown in the session header.
    """
    price = price_for_model(session.model)
    return csm.Rates(
        input_per_mtok=price.input * 1_000_000,
        output_per_mtok=price.output * 1_000_000,
        cache_write_mult=price.cache_write / price.input,
        cache_read_mult=price.cache_read / price.input,
    )


def _subagent_cost(session: Session) -> float:
    """Every dollar a subagent spent, exactly as ``analysis.attribution``
    already priced it (real per-call rates, no re-estimation needed here).

    ``chunk_split_model``'s linear-ramp geometry only describes the MAIN
    thread's own context growth — a subagent runs in its own, separate,
    isolated context window, so there is no sound way to fold its tokens
    into the same ramp. It is added to ``split_cost`` on top of that estimate
    instead (see ``price_split``): splitting the main thread does not change
    what a subagent cost, so it is priced once, exactly, and carried through
    unchanged rather than modelled on the ramp at all. ``as_is_cost`` needs no
    such addition — it already includes every subagent dollar, being
    ``session.attributed_cost`` itself.
    """
    return sum(e.attributed_cost for e in session.events if e.agent_id is not None)


def price_split(session: Session, split_fraction: float,
                alpha: float = SUMMARY_ALPHA) -> Optional[dict]:
    """Price splitting ``session`` at ``split_fraction`` of the way through.

    Adapted from the prototype's ``session_core.saving_for_split`` — but
    where that prototype (and this function, until this was fixed) priced
    the CURRENT, unsplit session with the same rough estimate used for the
    hypothetical split, ``as_is_cost`` here is simply ``session.attributed_cost``:
    the exact bill ``analysis.attribution`` already computed, the same number
    the session header calls "Retrospective cost". There is no reason to
    re-estimate a number that is already known exactly, and doing so was
    exactly why ``as_is_cost`` used to disagree with "Retrospective cost" —
    first by rate, then by scope (subagents) — for what should always have
    been the same session, priced the same way, everywhere in the app.

    ``split_cost`` stays the ``chunk_split_model`` estimate: there is no
    "exact" figure for a split that never happened, only a geometric
    approximation of the linear context ramp, plus real subagent dollars
    (``_subagent_cost``) added on top since a subagent's cost does not change
    when the main thread's reading is split. ``percent_saving`` is now a
    percentage of the session's one real total bill, not of a second
    estimate — the number a user sees anywhere in the app for "what did this
    cost" is this same ``as_is_cost``, never a different figure that happens
    to answer the same question.

    Returns ``None`` when the session does not fit the ramp model at all: no
    real cache-read, too few calls, or a split point too close to either end
    to model sensibly.
    """
    input_tok, output_tok, cache_write_tok, cache_read_tok, api_calls = _session_buckets(session)
    if cache_read_tok <= 0 or api_calls < 2:
        return None
    if not (0.05 < split_fraction < 0.95):
        return None

    peak = csm.peak_context_tokens(cache_read_tok, api_calls)
    if peak <= 0:
        return None

    rates = _rates_for_session(session)
    subagent_cost = _subagent_cost(session)
    heights = [peak * split_fraction, peak * (1.0 - split_fraction)]
    calls = csm.calls_per_chunk(heights, api_calls, peak)
    cache_read_split = csm.cache_read_after_split(heights, api_calls, peak)

    summary_tok = csm.summary_tokens(heights[0], alpha)
    overhead, *_ = csm.summary_overhead_cost(calls[1:], summary_tok, rates)

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
    }


# ----------------------------------------------------------------------
# Pattern detection — the front-loaded reading phase
# ----------------------------------------------------------------------

def find_front_loaded_reading(session: Session) -> Optional[Block]:
    """The session's opening reading run, if it looks like a missed plan-mode
    opportunity, else ``None``.

    Walks ``session.blocks`` in order and returns the first ``read`` block
    that: starts within the first ``START_WITHIN_FRACTION`` of the session (by
    cumulative message count), is at least ``MIN_READ_BLOCK_MESSAGES`` long,
    and is genuinely followed by real work (a later ``write``/``execute``
    block) — a session that is all reading is not a missed opportunity, it is
    just reading.
    """
    blocks = session.blocks
    if not blocks:
        return None

    total = sum(block.message_count for block in blocks)
    if total <= 0:
        return None

    cumulative = 0
    for index, block in enumerate(blocks):
        start_fraction = cumulative / total
        cumulative += block.message_count

        if block.kind != READ:
            continue
        if block.message_count < MIN_READ_BLOCK_MESSAGES:
            continue
        if start_fraction > START_WITHIN_FRACTION:
            continue
        if not any(b.kind in (WRITE, EXECUTE) for b in blocks[index + 1:]):
            continue
        return block

    return None


def _split_fraction_for(session: Session, read_block: Block) -> float:
    """Where ``read_block`` ends, as a fraction of the main thread's API calls."""
    call_times = _call_times(_main_thread_events(session))
    if not call_times:
        return 0.0
    end = read_block.t_end
    if end is None:
        return 0.0
    passed = sum(1 for ts in call_times if ts <= end)
    return passed / len(call_times)


# ----------------------------------------------------------------------
# LLM-as-judge — grounding the suggestion in what was actually read
# ----------------------------------------------------------------------

MAX_JUSTIFY_TARGETS = 12


def _read_block_targets(read_block: Block) -> list[str]:
    """What the flagged read block actually looked at, in order, deduped.

    File paths for ``Read``, patterns for ``Grep``/``Glob``, queries for
    ``WebSearch`` — whichever the call's own input carries.
    """
    targets: list[str] = []
    for event in read_block.events:
        if event.tool is None:
            continue
        value = (event.tool.input.get("file_path")
                 or event.tool.input.get("pattern")
                 or event.tool.input.get("query")
                 or event.tool.input.get("url"))
        if value:
            targets.append(str(value))

    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            ordered.append(target)
    return ordered


def _next_work_target(session: Session, read_block: Block) -> str:
    """What the session actually changed or ran right after the flagged read.

    Identity, not equality: ``Block`` is a plain dataclass, and two blocks
    could in principle compare equal by field values — the caller means THIS
    object, wherever it sits in ``session.blocks``.
    """
    blocks = session.blocks
    try:
        index = next(i for i, block in enumerate(blocks) if block is read_block)
    except StopIteration:
        return ""

    for block in blocks[index + 1:]:
        if block.kind not in (WRITE, EXECUTE):
            continue
        for event in block.events:
            if event.tool is None:
                continue
            target = event.tool.input.get("file_path") or event.tool.input.get("command")
            if target:
                return str(target)
        return block.label
    return ""


def build_justification_prompt(task: str, targets: list[str], next_target: str) -> str:
    """The one-sentence-justification prompt. Pure string; no API call.

    Explicitly tells the judge that plan mode alone does not clear context by
    default (see the module docstring's caveat) — without this, the model's
    natural framing ("do this in plan mode to save tokens") repeats the same
    overstatement ``detect``'s own detail sentence had to be corrected for.
    """
    listed = "\n".join(f"- {t}" for t in targets[:MAX_JUSTIFY_TARGETS])
    return (
        "A coding agent was given this task by its user:\n"
        f"\"{task}\"\n\n"
        "Before making any change, it read the following, in order:\n"
        f"{listed or '(no file paths or search terms were recorded)'}\n\n"
        f"Its first real change afterwards was: {next_target or '(not recorded)'}.\n\n"
        "IMPORTANT: by default, exiting Claude Code's plan mode does NOT "
        "clear the conversation's context — it only stops the agent from "
        "editing. The token saving being suggested here assumes the user "
        "ALSO clears context when approving the plan (a non-default option), "
        "so that only the plan carries forward instead of the full reading. "
        "Do not imply plan mode alone reduces cost.\n\n"
        "In ONE short, concrete sentence, explain why that opening reading "
        "looks like it was for planning/understanding rather than work that "
        "would have happened regardless of mode — reference what was "
        "actually being looked for, not a generic reason. Answer with only "
        "the sentence: no quotes, no preamble."
    )


_JUSTIFY_CACHE_FILE = CACHE_DIR / "plan_mode_justifications.json"
_justify_cache: Optional[dict] = None


def _load_justify_cache() -> dict:
    global _justify_cache
    if _justify_cache is not None:
        return _justify_cache
    try:
        _justify_cache = json.loads(_JUSTIFY_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(_justify_cache, dict):
            _justify_cache = {}
    except (OSError, json.JSONDecodeError):
        _justify_cache = {}
    return _justify_cache


def _save_justify_cache() -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp = _JUSTIFY_CACHE_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(_load_justify_cache(), indent=0), encoding="utf-8")
        temp.replace(_JUSTIFY_CACHE_FILE)
    except OSError:
        pass


def _justify_cache_key(task: str, targets: list[str], next_target: str) -> str:
    """Content-addressed, so the same situation is judged once ever."""
    payload = json.dumps([task, targets, next_target, JUDGE_MODEL],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def justify(session: Session, read_block: Block) -> Optional[str]:
    """One LLM-written sentence grounding the suggestion, or ``None``.

    ``None`` — never raised — when the LLM is unavailable, the session has no
    prompt to ground the question in, or the call itself fails; the caller
    falls back to the structural detail sentence, the same degrade
    ``analysis.classify.judge_calls`` uses for an ambiguous Bash command.

    Only a non-empty result is cached, so a failed call (a timeout, in
    particular — see ``analysis.task_forest.judge_task_forest``) is retried on
    the next scan rather than permanently losing its justification.
    """
    if not classify.llm_available():
        return None

    task = session.user_prompts[0] if session.user_prompts else session.title
    if not task:
        return None

    targets = _read_block_targets(read_block)
    next_target = _next_work_target(session, read_block)

    cache = _load_justify_cache()
    key = _justify_cache_key(task, targets, next_target)
    if key in cache:
        return cache[key] or None

    text = ""
    try:
        from openai import OpenAI

        client = (OpenAI(base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_S)
                  if LLM_BASE_URL else OpenAI(timeout=LLM_TIMEOUT_S))
        response = chat_completion(
            client,
            model=JUDGE_MODEL, max_tokens=120, temperature=0.2,
            messages=[{"role": "user",
                      "content": build_justification_prompt(task, targets, next_target)}],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        text = ""

    if text:
        cache[key] = text
        _save_justify_cache()
    return text or None


# ----------------------------------------------------------------------
# The public entry point
# ----------------------------------------------------------------------

def detect(session: Session) -> Optional[Problem]:
    """One ``Problem`` for a missed plan-mode opportunity, or ``None``.

    A session that already used plan mode is never flagged — the opportunity
    was taken, so there is nothing to suggest.
    """
    if session.used_plan_mode:
        return None

    read_block = find_front_loaded_reading(session)
    if read_block is None:
        return None

    fraction = _split_fraction_for(session, read_block)
    saving = price_split(session, fraction)
    if saving is None:
        return None
    if (saving["percent_saving"] < MIN_PCT_SAVING
            or saving["dollar_saving"] < MIN_DOLLAR_SAVING):
        return None

    detail = (
        f"This session read for {read_block.message_count} steps before any "
        f"edit or command ran. Running that reading in plan mode AND clearing "
        f"context when approving the plan could have saved an estimated "
        f"{saving['percent_saving']:.0f}% (${saving['dollar_saving']:.2f}) — "
        f"plan mode alone does not clear context by default."
    )

    data = {
        **saving,
        "split_fraction": fraction,
        # The event-uuid anchor DESIGN.md calls for, plus the read block's
        # own end timestamp: ``Block.as_dict()`` carries ``t_start``/
        # ``t_end`` but not individual event uuids, and the frontend needs
        # to locate this same block among ``session.blocks`` without a
        # schema change there — the timestamp already round-trips exactly.
        "split_after_event_uuid": read_block.events[-1].uuid,
        "split_after_ts": read_block.t_end.isoformat() if read_block.t_end else None,
        "read_block_message_count": read_block.message_count,
    }

    # Optional: grounds the mechanism above in what was actually read. Never
    # gates the suggestion — a session is flagged from the structural pattern
    # and the price alone, this only adds a reason when one is available.
    justification = justify(session, read_block)
    if justification:
        data["justification"] = justification

    return Problem(
        id="plan-mode",
        title="Front-loaded reading could have run in plan mode",
        severity="medium",
        detail=detail,
        data=data,
    )
