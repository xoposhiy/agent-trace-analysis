"""Place every billed token on the block that caused it.

The problem
-----------
``usage`` is reported once per assistant *message*, so the adapter has to
divide it. Its first cut (``adapters.claude_code._attribute_tokens``) splits
``output`` across the message's own content blocks and then dumps ``input`` and
``cache_creation`` wholesale on the message's *first* Event. That last part is
not an attribution, it is a placement of convenience, and it shows: the largest
block on a real 239-block bar was a single ``Read /tmp/subagent_preview.png``
carrying 325,412 prompt-side tokens, none of which it caused — it was simply
the first thing the model did after a 1h47m gap, so it inherited the cost of
re-establishing the whole context.

The model
---------
Two observations make a real attribution possible.

1. ``output`` for a call belongs to the content that call *generated*. Already
   handled; this module keeps that split and reuses it.

2. ``input + cache_creation`` for a call is the **fresh** part of its prompt —
   the content appended to the conversation since the previous call. That is a
   specific, identifiable set of pieces, not a proportional guess:

       · results of the previous call's tool calls  -> that tool call's block
       · a user prompt typed in between             -> that user_chat block
       · the previous call's own output re-entering -> those blocks

Verified against real transcripts: the characters appended between consecutive
calls, over that call's ``input + cache_creation``, gives a stable ratio across
four independent sessions —

    51db4d3e  n=249  median 4.61 chars/token   (p25 3.79, p75 5.87)
    1c4504f8  n=344  median 4.55               (p25 3.69, p75 6.70)
    d2b6b2bb  n=211  median 4.72               (p25 3.77, p75 7.53)
    edad9dce  n=107  median 4.65               (p25 3.48, p75 13.10)

The spread does not matter. Sizes are used only as *relative weights* to divide
a per-call total that is known exactly from ``usage``, so a mis-sized piece
shifts tokens between siblings but never changes the sum.

``cache_read`` is deliberately excluded, so this totals ``input + output +
cache_creation`` — the ``working`` figure the header already shows. Counting
re-reads would make every late block outgrow every early one regardless of what
it did (CLAUDE.md §4: one session was 95% cache reads).

What this is not
----------------
Exact per-block truth. The API bills per request, so no per-block figure exists
to be right about. This divides real totals by defensible weights: the session
sum is exact, an individual block is an attribution.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

from Final_app.ir.models import (
    EV_ASSISTANT,
    EV_TOOL_USE,
    EV_USER,
    Event,
    Session,
)

# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------

# Characters per token, measured on messages containing exactly one kind of
# content and no thinking block, where the whole billed ``output`` provably
# belongs to what is visible (n=120 prose, n=367 tool_use across all
# transcripts on this machine, 2026-08-05):
#
#     prose      median 2.70 chars/token   (p10 2.48, p90 2.99)
#     tool_use   median 1.82               (p10 1.01, p90 2.57)
#
# Tool-call JSON tokenises far denser than prose — quoting and structure — so a
# single characters-per-token constant would systematically under-weight tool
# calls against the prose beside them.
CHARS_PER_TOKEN_PROSE = 2.70
CHARS_PER_TOKEN_TOOL_CALL = 1.82

# Tool results are mostly file contents, code and command output: denser than
# prose, lighter than argument JSON. Not separately measurable — a result is
# never billed on its own line — so this is an interpolation, and it only ever
# affects how a call's fresh tokens divide between siblings.
CHARS_PER_TOKEN_RESULT = 3.0


class Calibration:
    """Characters per token, for one session.

    The constants above are population defaults, measured on the transcripts
    that happened to be available while this was written. A session whose
    content is mostly minified JSON, or mostly Chinese, or mostly ASCII tables
    tokenises nothing like those, so each session re-derives its own figures
    from its own billing wherever it can and only falls back to the defaults
    when it cannot.
    """

    __slots__ = ("prose", "tool_call", "result", "prose_samples", "tool_samples")

    def __init__(self, prose: float = CHARS_PER_TOKEN_PROSE,
                 tool_call: float = CHARS_PER_TOKEN_TOOL_CALL,
                 result: float = CHARS_PER_TOKEN_RESULT,
                 prose_samples: int = 0, tool_samples: int = 0):
        self.prose = prose
        self.tool_call = tool_call
        self.result = result
        self.prose_samples = prose_samples
        self.tool_samples = tool_samples


DEFAULT_CALIBRATION = Calibration()

# Below this, a session's own ratio is noise and the population default is the
# better estimate. Chosen so a handful of unusual messages cannot swing a
# session; the measured spread within a type is tight (prose p10-p90 was
# 2.48-2.99), so a small sample is enough once there is one at all.
MIN_CALIBRATION_SAMPLES = 5


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def calibrate(session: Session) -> Calibration:
    """Derive this session's characters-per-token from its own billing.

    Uses only messages where the billed ``output`` provably belongs to content
    that is visible and of a single kind:

      * no thinking block — thinking is billed but stored empty, so part of
        what was paid for cannot be seen;
      * one kind of content, so the figure is not shared with another type.

    A result's ratio cannot be derived this way — a tool result is never billed
    on its own, it only ever arrives inside some later call's prompt — so it
    keeps the default.
    """
    by_message: dict[str, list[Event]] = defaultdict(list)
    for event in session.events:
        if event.message_id:
            by_message[event.message_id].append(event)

    prose_ratios: list[float] = []
    tool_ratios: list[float] = []

    for message_id, events in by_message.items():
        if message_id in session.thinking_message_ids:
            continue
        billed = sum(event.tokens.output for event in events)
        if billed <= 0:
            continue

        kinds = {event.type for event in events}
        if kinds == {EV_ASSISTANT}:
            chars = sum(len(event.text) for event in events)
            if chars > 50:
                prose_ratios.append(chars / billed)
        elif kinds == {EV_TOOL_USE}:
            chars = sum(
                len(event.tool.name) + len(json.dumps(event.tool.input, default=str))
                for event in events if event.tool is not None
            )
            if chars > 50:
                tool_ratios.append(chars / billed)

    return Calibration(
        prose=(_median(prose_ratios) if len(prose_ratios) >= MIN_CALIBRATION_SAMPLES
               else CHARS_PER_TOKEN_PROSE),
        tool_call=(_median(tool_ratios) if len(tool_ratios) >= MIN_CALIBRATION_SAMPLES
                   else CHARS_PER_TOKEN_TOOL_CALL),
        prose_samples=len(prose_ratios),
        tool_samples=len(tool_ratios),
    )


def _size(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / max(chars_per_token, 0.1)))


def generated_weight(event: Event,
                     calibration: Optional[Calibration] = None) -> int:
    """Size of the content this Event *generated* — its own call or prose.

    Prefers a real measurement from ``analysis.tokens`` when one exists, since
    that is exact for the block; falls back to the calibrated character
    estimate so the whole model works with no network.
    """
    if event.content_tokens:
        return event.content_tokens
    calibration = calibration or DEFAULT_CALIBRATION
    if event.type == EV_TOOL_USE and event.tool is not None:
        payload = event.tool.name + json.dumps(event.tool.input, default=str)
        return _size(payload, calibration.tool_call)
    return _size(event.text, calibration.prose)


def result_weight(event: Event,
                  calibration: Optional[Calibration] = None) -> int:
    """Size of what this Event put *back into* the conversation.

    Only tool calls have one. This is the number that makes a ``Read`` of a big
    file expensive: its call is ~40 tokens, its result is thousands.
    """
    if event.type != EV_TOOL_USE or event.tool is None or event.tool.result is None:
        return 0
    # ``size_chars``, not ``len(output)``: a Read's file body is in the
    # envelope and never reaches ``output``, and it is the whole reason a Read
    # is expensive. Falls back for results parsed before the field existed.
    chars = event.tool.result.size_chars or len(event.tool.result.output)
    if not chars:
        return 0
    calibration = calibration or DEFAULT_CALIBRATION
    return max(1, round(chars / max(calibration.result, 0.1)))


# ----------------------------------------------------------------------
# Exact division
# ----------------------------------------------------------------------

def split_exact(amount: int, weights: list[int]) -> list[int]:
    """Divide ``amount`` by ``weights`` so the shares sum to it exactly.

    Largest-remainder. Plain rounding drifts, and across the several hundred
    calls in a session the drift is what stops block totals matching the header.
    """
    if not weights or amount <= 0:
        return [0] * len(weights)
    total = sum(weights)
    if total <= 0:
        base, extra = divmod(amount, len(weights))
        return [base + (1 if i < extra else 0) for i in range(len(weights))]

    shares = [amount * weight // total for weight in weights]
    order = sorted(range(len(weights)),
                   key=lambda i: (amount * weights[i]) % total, reverse=True)
    for i in order[: amount - sum(shares)]:
        shares[i] += 1
    return shares


# ----------------------------------------------------------------------
# Calls
# ----------------------------------------------------------------------

class _Call:
    """One API request: the Events it generated, and what it was billed."""

    __slots__ = ("key", "events", "output", "fresh", "first_ts")

    def __init__(self, key: str, event: Event):
        self.key = key
        self.events: list[Event] = [event]
        self.output = 0
        self.fresh = 0                    # input + cache_creation
        self.first_ts = event.ts


def _calls_for(events: list[Event]) -> list[_Call]:
    """Group an ordered Event stream into API calls, newest last.

    Grouped by ``message_id`` because one call is written to disk as one line
    per content block, all sharing that id — verified: 3324 of 3325 assistant
    lines across every transcript here carry exactly one content block, so the
    call boundary is the id, never the line.
    """
    calls: dict[str, _Call] = {}
    for event in events:
        if event.type not in (EV_ASSISTANT, EV_TOOL_USE):
            continue
        key = event.message_id or f"uuid:{event.uuid}"
        call = calls.get(key)
        if call is None:
            calls[key] = call = _Call(key, event)
        else:
            call.events.append(event)
        # The adapter already put the message's billed figures on its Events:
        # ``output`` spread over all of them, the prompt side on the first.
        call.output += event.tokens.output
        call.fresh += event.tokens.input + event.tokens.cache_creation

    ordered = sorted(calls.values(), key=lambda c: c.first_ts)
    return ordered


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------

def attribute(session: Session) -> int:
    """Set ``attributed_tokens`` on every Event. Returns what stayed unplaced.

    Every billed working token lands on a block: the return value is 0 except
    for a thread with no attributable Events at all. Three rules, in order of
    how directly the cost can be traced:

      1. Content that appeared since the last call pays for that call's fresh
         prompt tokens — the direct case, and the common one.
      2. A call with nothing new in front of it (the session's first call, or
         one after a long pause where the whole context is re-cached) is paid
         for by everything already sitting in the prompt, in proportion to how
         much of it each block occupies. Re-caching is caused by the size of
         the context, so the context pays.
      3. Whatever is still unplaced at the end — the system prompt and tool
         definitions, which are in *every* call's prompt and belong to no
         single block — is spread across the whole session in proportion to
         what each block already carries.

    Rule 1 is an attribution. Rules 2 and 3 are allocations of cost that is
    genuinely shared; they are applied only to what rule 1 cannot place.
    """
    for event in session.events:
        event.attributed_tokens = 0

    # Subagents are billed on their own transcripts, so each thread attributes
    # independently; mixing them would charge a parent call for a child's work.
    by_thread: dict[Optional[str], list[Event]] = defaultdict(list)
    for event in session.events:
        by_thread[event.agent_id].append(event)

    calibration = calibrate(session)

    unplaced = 0
    for thread_events in by_thread.values():
        unplaced += _attribute_thread(thread_events, calibration)

    if unplaced > 0:
        # Rule 3.
        placed = [e for e in session.events if e.attributed_tokens > 0]
        if not placed:
            return unplaced
        for event, share in zip(
            placed, split_exact(unplaced, [e.attributed_tokens for e in placed])
        ):
            event.attributed_tokens += share
        unplaced = 0

    _raise_floor(session, calibration)
    return unplaced


def _raise_floor(session: Session, calibration: Calibration) -> None:
    """No Event with real content may report zero. Total is left unchanged.

    A session can be billed less ``input + cache_creation`` than its content
    accounts for, because cache reads covered the rest — 0.81x in a real
    session here. Everything is then scaled down proportionally, and the
    smallest pieces round to nothing: a user message is short and arrives
    behind a long assistant turn, so "0 tokens" landed on real prompts a user
    had actually typed.

    Zero is the one number that reads as a bug rather than as a small share, so
    each starved Event is given a token and the same number is taken back from
    the largest recipients. The session total is untouched, which is what keeps
    the blocks summing to the header.
    """
    starved = [
        event for event in session.events
        if event.attributed_tokens == 0
        and (generated_weight(event, calibration) > 0
             or result_weight(event, calibration) > 0)
    ]
    if not starved:
        return

    # Largest first, so the tokens come from where they are least noticeable.
    donors = sorted(
        (e for e in session.events if e.attributed_tokens > 1),
        key=lambda e: e.attributed_tokens, reverse=True,
    )
    if not donors:
        return

    taken = 0
    for donor in donors:
        if taken >= len(starved):
            break
        # Never take a donor below 1: that would only move the zero elsewhere.
        give = min(donor.attributed_tokens - 1, len(starved) - taken)
        donor.attributed_tokens -= give
        taken += give

    for event in starved[:taken]:
        event.attributed_tokens += 1


def _attribute_thread(events: list[Event], calibration: Calibration) -> int:
    events = sorted(events, key=lambda e: e.ts)
    calls = _calls_for(events)
    if not calls:
        return 0

    user_events = [e for e in events if e.type == EV_USER]

    # Everything that has entered this thread's prompt so far, as
    # ``(event, weight)``. A tool call appears twice — once for the call it
    # emitted, once for the result it brought back — because the prompt
    # carries both and the result is usually far the larger.
    context: list[tuple[Event, int]] = []

    # Content that has entered the prompt but has not yet been paid for, oldest
    # first. See the settlement loop below for why this cannot be per-call.
    owed: list[list] = []

    unplaced = 0
    for index, call in enumerate(calls):
        # 1. What this call produced is charged to what it produced.
        generated = call.events
        for event, share in zip(
            generated, split_exact(call.output, [generated_weight(e, calibration) for e in generated])
        ):
            event.attributed_tokens += share

        # 2. What this call was newly charged to read is charged to whatever
        #    appeared in the conversation since the previous call.
        arrived: list[tuple[Event, int]] = []
        window_start = calls[index - 1].first_ts if index else None

        if index:
            for event in calls[index - 1].events:
                # The previous turn's own output is now part of the prompt...
                arrived.append((event, generated_weight(event, calibration)))
                # ...and so is whatever its tool calls returned.
                returned = result_weight(event, calibration)
                if returned:
                    arrived.append((event, returned))

        for event in user_events:
            if event.ts > call.first_ts:
                continue
            if window_start is not None and event.ts <= window_start:
                continue
            arrived.append((event, generated_weight(event, calibration)))

        # The whole of ``fresh`` goes to what arrived, uncapped. Content the IR
        # never stored still entered the prompt — attachments (13.8% of the
        # characters in a real transcript) and thinking (billed, but Claude
        # Code writes the block with an empty ``thinking`` field, so its size
        # is unrecoverable). Both belong to the turn they arrived with, so
        # letting the visible pieces of that turn absorb them puts the cost in
        # the right place even though it cannot be itemised.
        # New content joins the ledger of what has entered the prompt but has
        # not yet been paid for.
        owed.extend([event, weight] for event, weight in arrived)
        context.extend(arrived)

        # Settle against everything still outstanding, pro rata.
        #
        # Two reasons this is a running ledger rather than a per-call split.
        #
        # Cache breakpoints do not move every turn, so ``fresh`` is lumpy.
        # Real consecutive calls in session 5f12bde3:
        #
        #     call   fresh   arrived
        #       32      10       192     starved
        #       34      15       119     starved
        #       35  13,366     8,054     the lump that paid for both
        #
        # Settled per call, a 13-token user prompt landing on call 34 could
        # only receive 1 token. The ledger lets call 35's lump pay off the
        # backlog it is actually for.
        #
        # And pro rata rather than oldest-first, because a session can end up
        # with *less* ``fresh`` than content — 0.81x in session 15d437fe, where
        # cache reads covered the rest. Oldest-first then never drains, and
        # whatever sits at the back of the queue is paid nothing at all. That
        # back is always the newest arrival, and a short user prompt appended
        # after a long assistant turn is last in its batch every time: 46
        # ``user_chat`` blocks across these transcripts came out at exactly
        # zero. Pro rata scales every debt down equally instead.
        remaining = call.fresh
        if owed and remaining > 0:
            debts = [debt for _, debt in owed]
            payable = min(remaining, sum(debts))
            for entry, share in zip(owed, split_exact(payable, debts)):
                entry[0].attributed_tokens += share
                entry[1] -= share
            remaining -= payable
            owed = [entry for entry in owed if entry[1] > 0]

        if remaining <= 0:
            continue

        # More fresh than any content accounts for. That is the session's
        # baseline — system prompt and tool definitions on the first call, or a
        # whole context re-cached after a pause (one real call billed 325,412
        # for 261 tokens of new content, after a 1h47m gap). Neither is caused
        # by one block, so the context that was re-read pays in proportion to
        # how much of it each block occupies.
        if index == 0 or not context:
            unplaced += remaining
            continue

        events_, weights = zip(*context)
        for event, share in zip(events_, split_exact(remaining, list(weights))):
            event.attributed_tokens += share

    return unplaced
