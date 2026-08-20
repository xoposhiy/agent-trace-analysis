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

3. ``cache_read`` for a call is the rest of its prompt: the prefix served from
   cache. That is not a proportional guess either — it is, by definition, the
   content already sitting in the context window, which this module is already
   tracking in order to answer rule 2. So a block is charged for a re-read in
   proportion to how much of the re-read prefix it occupies, on every call that
   re-read it.

   The two channels stay separate on purpose. ``fresh`` is charged once, to
   whatever arrived; ``cache_read`` is charged repeatedly, to whatever is still
   resident. Merging them into one pot would pay for the same content twice —
   once on arrival and again as part of the prefix.

Together the three total ``input + output + cache_creation + cache_read`` —
every billed token, which is what makes the bar a picture of the whole context
window rather than of the working set alone.

Two figures, never one
----------------------
What CLAUDE.md §7 actually bans is the *naive* sum: adding up ``cache_read``
across every message in a session double-, triple-, hundred-fold-counts the
same resident content, since it gets re-billed on every later call that still
has it in its prompt — 12.5M of a real 13.2M session total was exactly that.
Shown as "tokens used", it overstates real work by ~18x.

That is not what happens here. ``attributed_tokens`` (working) and
``attributed_cache_read`` stay separate fields on every Event, Block and
payload — never collapsed into one stored number — and each cache-read token
is charged exactly once, to whichever block put that content in the context
window, not counted again at every later re-read. The bar's token axis does
display their sum, but as a distinct, labelled question — "how much of the
context window went through here" — and never without the split sitting
beside it (one hover away). A sum shown *with* its breakdown is not the trap
§7 names; a bare sum standing in for "work done" is.

What this is not
----------------
Exact per-block truth. The API bills per request, so no per-block figure exists
to be right about. This divides real totals by defensible weights: the session
sum is exact, an individual block is an attribution.

Dollars
-------
``attributed_cost`` rides the same ledger: every place a token share is handed
to an Event, a matching dollar share is computed from the *call* that paid it
— never the model of the Event receiving the blame, which can differ (a
Sonnet main thread re-reading what a Haiku subagent wrote). That keeps a
mixed-model session priced correctly without a second pass over the
transcript. See ``analysis.pricing`` for the rate table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

from Final_app.analysis.pricing import price_for_model
from Final_app.ir.models import (
    EV_ASSISTANT,
    EV_COMPACTION,
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
    """One API request: the Events it generated, and what it was billed.

    ``model`` and the ``fresh_input``/``fresh_cache_creation`` split are kept
    only for pricing — ``fresh`` alone is enough to attribute tokens, but a
    dollar figure needs to know *which* rate applies to that lump.
    """

    __slots__ = ("key", "events", "output", "fresh", "cached", "first_ts",
                 "model", "fresh_input", "fresh_cache_creation")

    def __init__(self, key: str, event: Event):
        self.key = key
        self.events: list[Event] = [event]
        self.output = 0
        self.fresh = 0                    # input + cache_creation
        self.cached = 0                   # cache_read
        self.first_ts = event.ts
        self.model = event.model
        self.fresh_input = 0
        self.fresh_cache_creation = 0


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
        call.cached += event.tokens.cache_read
        call.fresh_input += event.tokens.input
        call.fresh_cache_creation += event.tokens.cache_creation

    ordered = sorted(calls.values(), key=lambda c: c.first_ts)
    return ordered


def _fresh_rate(call: "_Call") -> float:
    """Dollars per token for this call's ``fresh`` (input + cache-write) lump.

    ``fresh`` mixes two channels priced ~1.25x apart, and the ledger later
    hands out *pieces* of it to whichever content is being paid off — never
    the whole lump to one place — so there is no single piece to look up a
    pure input-vs-cache-write split for. Blending at the call's own actual
    mix, rather than a session- or table-wide guess, is exact for any call
    settled in one step and only approximate for the rare call that pays off
    a multi-call backlog (see ``LEDGER_CALLS``) — bounded by the 1.25x spread
    between the two channels it blends, not by the size of the backlog.
    """
    if call.fresh <= 0:
        return 0.0
    price = price_for_model(call.model)
    return ((call.fresh_input * price.input
             + call.fresh_cache_creation * price.cache_write) / call.fresh)


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------

def attribute(session: Session) -> int:
    """Set ``attributed_tokens`` and ``attributed_cache_read`` on every Event.

    Returns what stayed unplaced, which is 0 except for a thread with no
    attributable Events at all. Rules, in order of how directly the cost can be
    traced:

      1. Content that appeared since the last call pays for that call's fresh
         prompt tokens — the direct case, and the common one.
      2. Content still resident in the context window pays for what each later
         call re-read from cache, in proportion to how much of the prefix it
         occupies. This is the ``attributed_cache_read`` channel.
      3. Whatever is still unplaced — the system prompt and tool definitions,
         which are in *every* call's prompt and belong to no single block, plus
         a first call's whole prompt — is spread across the whole session in
         proportion to what each block already carries. Applied to both
         channels, each against its own total.

    Rule 1 is an attribution. Rules 2 and 3 are allocations of cost that is
    genuinely shared; they are applied only to what rule 1 cannot place.

    The return value tracks the working channel only. Cache reads follow it: the
    single case that leaves either unplaced is a thread with nothing to attribute
    to, and that places neither.
    """
    for event in session.events:
        event.attributed_tokens = 0
        event.attributed_cache_read = 0
        event.attributed_cost = 0.0

    # Subagents are billed on their own transcripts, so each thread attributes
    # independently; mixing them would charge a parent call for a child's work.
    by_thread: dict[Optional[str], list[Event]] = defaultdict(list)
    for event in session.events:
        by_thread[event.agent_id].append(event)

    calibration = calibrate(session)

    unplaced = 0
    unplaced_cache_read = 0
    unplaced_cost = 0.0
    for agent_id, thread_events in by_thread.items():
        # Compaction is recorded only in the main transcript, so a subagent
        # thread gets no boundaries rather than inheriting the parent's — its
        # own context window was never rebuilt at those moments.
        boundaries = session.compaction_points if agent_id is None else []
        working, cached, cost = _attribute_thread(thread_events, calibration, boundaries)
        unplaced += working
        unplaced_cache_read += cached
        unplaced_cost += cost

    if unplaced > 0 or unplaced_cache_read > 0:
        # Rule 3. Weighted by the working figure in both cases: it is the
        # closest thing to "how much of this session was this block", and using
        # each block's own cache-read share as the weight would compound the
        # position bias that share already carries.
        placed = [e for e in session.events if e.attributed_tokens > 0]
        if not placed:
            return unplaced
        weights = [e.attributed_tokens for e in placed]
        for event, share in zip(placed, split_exact(unplaced, weights)):
            event.attributed_tokens += share
        for event, share in zip(placed, split_exact(unplaced_cache_read, weights)):
            event.attributed_cache_read += share
        unplaced = 0

        # Dollars carry no integer-exactness requirement — nothing here is
        # displayed as a token count — so the shared baseline's cost is spread
        # by the same weights directly instead of through ``split_exact``.
        if unplaced_cost > 0:
            total_weight = sum(weights)
            if total_weight > 0:
                for event, weight in zip(placed, weights):
                    event.attributed_cost += unplaced_cost * weight / total_weight
            else:
                share = unplaced_cost / len(placed)
                for event in placed:
                    event.attributed_cost += share

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


# How many calls an unpaid debt keeps asking for payment before it retires.
#
# The ledger exists because ``fresh`` is lumpy — a cache breakpoint that does not
# move leaves consecutive calls starved and then one call pays for all of them.
# The widest real instance found here spans three calls (session 5f12bde3, calls
# 32 and 34 starved at fresh 10 and 15, call 35 paying 13,366), so this leaves
# more than double that headroom while still bounding the work per call to a
# handful of entries instead of the whole session's backlog.
LEDGER_CALLS = 8


def _attribute_thread(events: list[Event], calibration: Calibration,
                      compaction_points: Optional[list] = None) -> tuple[int, int, float]:
    """Attribute one thread. Returns ``(unplaced_working, unplaced_cache_read, unplaced_cost)``."""
    events = sorted(events, key=lambda e: e.ts)
    calls = _calls_for(events)
    if not calls:
        return 0, 0, 0.0

    user_events = [e for e in events if e.type == EV_USER]

    # Forward-only cursor into ``user_events``. Both lists are ordered by ts and
    # each call's window is (previous call, this call], so the windows are
    # consecutive and non-overlapping: a user event claimed by one call can never
    # belong to a later one. Rescanning the whole list per call did the same work
    # N times for the same answer.
    user_cursor = 0

    # What is currently resident in this thread's context window, as
    # ``[event, weight, rate_at_arrival, cost_rate_at_arrival]``. A tool call
    # appears twice — once for the call it emitted, once for the result it
    # brought back — because the prompt carries both and the result is usually
    # far the larger.
    context: list[list] = []
    context_weight = 0

    # The cache-read channel, kept as a running total of "re-read per unit of
    # resident weight" — in tokens (``cumulative_rate``) and, in parallel, in
    # dollars (``cumulative_cost_rate``, advanced by that call's own model's
    # cache-read rate rather than a flat one). A block's cache-read earning is
    # its own weight times how much each total advanced while it was resident,
    # which makes the whole channel two passes instead of re-splitting the
    # prefix on every call: spreading each call's ``cache_read`` across the
    # context directly is O(N^2), and on this channel it would run on *every*
    # call rather than only the occasional surplus one.
    cumulative_rate = 0.0
    cumulative_cost_rate = 0.0
    # id(event) -> [event, tokens_earned, dollars_earned]. ``None`` marks the
    # unowned baseline, same as everywhere else in this function.
    cache_shares: dict[int, list] = {}

    boundaries = sorted(compaction_points or [])
    boundary_index = 0
    markers = {e.ts: e for e in events if e.type == EV_COMPACTION}

    # The system prompt and tool definitions sit in *every* prompt, and no Event
    # carries their weight — so without a stand-in for them, the little content
    # that happens to be resident early absorbs the whole re-read of them. That
    # is not a rounding error: a real ``hi`` of 485 working tokens came out
    # holding 165,554 cache reads, and on a short session it was a third of the
    # total. It is the same mistake the module docstring opens with (a 200-token
    # prompt inheriting a 325,412-token prompt), arriving through the new channel.
    #
    # So the baseline is resident like anything else, but owned by nobody: it
    # earns a share of every re-read and that share goes to the session-wide
    # spread in ``attribute``. Sized as the whole of the first call's prompt,
    # which overstates it by the first user message — a rounding error against
    # a system prompt and tool set, and the alternative is a magic constant.
    # It is re-seeded after each compaction, which rewrites the conversation but
    # leaves the system prompt and tool definitions exactly where they were.
    # ``None`` in an entry's event slot is what marks it unowned.
    baseline_weight = calls[0].fresh + calls[0].cached

    def settle_cache_reads(entries: list[list]) -> None:
        """Bank what these entries earned while resident, tokens and dollars.

        The caller discards the entries afterwards; banking is what makes that
        safe, since an entry's earnings are otherwise only implied by how far
        ``cumulative_rate``/``cumulative_cost_rate`` have moved since it arrived.
        """
        for event, weight, rate_at_arrival, cost_rate_at_arrival in entries:
            earned = weight * (cumulative_rate - rate_at_arrival)
            earned_cost = weight * (cumulative_cost_rate - cost_rate_at_arrival)
            if earned <= 0 and earned_cost <= 0:
                continue
            share = cache_shares.setdefault(id(event), [event, 0.0, 0.0])
            share[1] += earned
            share[2] += earned_cost

    # Content that has entered the prompt but has not yet been paid for, oldest
    # first. See the settlement loop below for why this cannot be per-call.
    owed: list[list] = []

    context.append([None, baseline_weight, 0.0, 0.0])
    context_weight += baseline_weight

    unplaced = 0
    unplaced_cache_read = 0
    unplaced_cost = 0.0
    for index, call in enumerate(calls):
        # A compaction rebuilds the prompt from a summary, so everything older
        # stops being re-read from this point on. Bank what left the window and
        # start the context again; without this, blocks from before the compaction
        # keep earning a share of re-reads of a prefix they are no longer in.
        while (boundary_index < len(boundaries)
               and boundaries[boundary_index] <= call.first_ts):
            settle_cache_reads(context)
            context = []
            context_weight = 0
            marker = markers.get(boundaries[boundary_index])
            boundary_index += 1
            context.append([None, baseline_weight, cumulative_rate, cumulative_cost_rate])
            context_weight += baseline_weight

            # The summary the compaction wrote *is* the new prefix, and no other
            # Event carries its size — the IR stores the marker with no body. Left
            # unowned, every later re-read of it falls through to the session-wide
            # spread and lands partly back on the blocks the compaction discarded,
            # which is the one thing the reset above exists to prevent. So the
            # marker holds the summary, sized from what this first call was billed
            # to read it: a weight in the same estimated-token units as every
            # other entry, and the compaction block ends up paying for the
            # context it created.
            if marker is not None and call.cached > 0:
                context.append([marker, call.cached, cumulative_rate, cumulative_cost_rate])
                context_weight += call.cached

        # 2. What this call re-read from cache is charged to the prefix it read:
        #    the context as it stood *before* this call's own arrivals, which are
        #    the fresh part of this prompt and are billed on the other channel.
        if call.cached > 0:
            if context_weight > 0:
                cache_read_rate = price_for_model(call.model).cache_read
                cumulative_rate += call.cached / context_weight
                cumulative_cost_rate += call.cached * cache_read_rate / context_weight
            else:
                # A first call's prefix is the system prompt and tool definitions
                # — cached from an earlier session and belonging to no block.
                unplaced_cache_read += call.cached
                unplaced_cost += call.cached * price_for_model(call.model).cache_read

        # 1. What this call produced is charged to what it produced.
        generated = call.events
        output_rate = price_for_model(call.model).output
        for event, share in zip(
            generated, split_exact(call.output, [generated_weight(e, calibration) for e in generated])
        ):
            event.attributed_tokens += share
            event.attributed_cost += share * output_rate

        # 3. What this call was newly charged to read is charged to whatever
        #    appeared in the conversation since the previous call.
        arrived: list[tuple[Event, int]] = []

        if index:
            for event in calls[index - 1].events:
                # The previous turn's own output is now part of the prompt...
                arrived.append((event, generated_weight(event, calibration)))
                # ...and so is whatever its tool calls returned.
                returned = result_weight(event, calibration)
                if returned:
                    arrived.append((event, returned))

        while (user_cursor < len(user_events)
               and user_events[user_cursor].ts <= call.first_ts):
            event = user_events[user_cursor]
            arrived.append((event, generated_weight(event, calibration)))
            user_cursor += 1

        # The whole of ``fresh`` goes to what arrived, uncapped. Content the IR
        # never stored still entered the prompt — attachments (13.8% of the
        # characters in a real transcript) and thinking (billed, but Claude
        # Code writes the block with an empty ``thinking`` field, so its size
        # is unrecoverable). Both belong to the turn they arrived with, so
        # letting the visible pieces of that turn absorb them puts the cost in
        # the right place even though it cannot be itemised.
        # New content joins the ledger of what has entered the prompt but has
        # not yet been paid for, and becomes resident in the context window —
        # from here on it is part of what later calls re-read.
        owed.extend([event, weight, index] for event, weight in arrived)
        for event, weight in arrived:
            context.append([event, weight, cumulative_rate, cumulative_cost_rate])
            context_weight += weight

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
        # Blended at this call's own input:cache-write mix (see ``_fresh_rate``),
        # so every piece handed out below — however this call's fresh budget
        # ends up split across old debts and new arrivals — is priced the same
        # way, without needing to know which channel a given piece came from.
        fresh_rate = _fresh_rate(call)
        if owed and remaining > 0:
            debts = [entry[1] for entry in owed]
            payable = min(remaining, sum(debts))
            for entry, share in zip(owed, split_exact(payable, debts)):
                entry[0].attributed_tokens += share
                entry[0].attributed_cost += share * fresh_rate
                entry[1] -= share
            remaining -= payable

        # Retire what is paid off, and what has been asking too long.
        #
        # The age bound is what makes this a ledger rather than a leak. Pro rata
        # means a starved call pays every debt a little and none in full, so the
        # ``debt > 0`` filter alone provably removes nothing: measured at N=2,000
        # the backlog grew by ~2 entries per call for the whole session while each
        # call re-split and re-sorted all of it. And starvation is the common
        # case, not the edge one — ``fresh`` excludes cache reads, which are ~95%
        # of a real session's tokens.
        #
        # A retired shortfall is not lost cost. ``fresh`` falls short precisely
        # when a cache breakpoint sits behind the new content, which means that
        # content is billed as ``cache_read`` from the next call onward — so the
        # remainder is charged on the cache-read channel above, where it belongs,
        # instead of waiting here for money that is never coming.
        owed = [entry for entry in owed
                if entry[1] > 0 and index - entry[2] < LEDGER_CALLS]

        if remaining <= 0:
            continue

        # More fresh than any content accounts for. That is the session's
        # baseline — system prompt and tool definitions on the first call, or a
        # whole context re-cached after a pause (one real call billed 325,412
        # for 261 tokens of new content, after a 1h47m gap). Neither is caused by
        # one block, so it goes to the session-wide spread in ``attribute``.
        #
        # This used to be charged to the resident context by size, which double
        # counted: ``context`` accumulates for the whole thread, so a block was
        # charged again at every later surplus call while a late block was
        # charged once. Across 600 calls with identical billing the first block
        # came out 1,097x the last — the mirror of the cache-read trap the module
        # docstring was written to kill. Re-read cost now has its own channel,
        # measured from ``cache_read`` rather than inferred from a surplus, so
        # this branch no longer has to tell that story.
        unplaced += remaining
        unplaced_cost += remaining * fresh_rate

    # Bank what is still resident at the end of the thread.
    settle_cache_reads(context)

    # One largest-remainder pass over the whole thread, so the cache-read total
    # lands exactly on the blocks despite the shares being floats.
    payable = sum(call.cached for call in calls) - unplaced_cache_read
    entries = list(cache_shares.values())
    weights = [max(0, round(tokens_earned)) for _, tokens_earned, _ in entries]
    if payable > 0 and sum(weights) > 0:
        # The baseline stays in the weights so every proportion is computed
        # against the real prefix; only its own share is handed on rather than
        # landing on a block.
        for (event, _, _), share in zip(entries, split_exact(payable, weights)):
            if event is None:
                unplaced_cache_read += share
            else:
                event.attributed_cache_read += share
    elif payable > 0:
        # Cache reads with nothing resident to charge them to. Session-wide.
        unplaced_cache_read += payable

    # Dollars need no largest-remainder pass — nothing downstream displays them
    # as an integer count — so each entry's banked dollar earning is handed
    # straight to its event.
    for event, _, dollars_earned in entries:
        if dollars_earned <= 0:
            continue
        if event is None:
            unplaced_cost += dollars_earned
        else:
            event.attributed_cost += dollars_earned

    return unplaced, unplaced_cache_read, unplaced_cost
