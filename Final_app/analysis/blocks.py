"""Turn classified events into the blocks the vertical bar draws.

Three passes, in order:

  1. **Split by thread.** Subagent events are pulled out of the main stream and
     grouped by ``agent_id``, so the parent timeline stays readable and each
     subagent becomes one container block holding its own inner blocks. The
     spawning call joins the container it launched rather than staying behind.
  2. **Classify** every event into one of the six kinds (``analysis.classify``).
  3. **Merge neighbours**, per the sketch's rule:

         neighbouring blocks of the same type
         (with coordination inside) - join them in one block

     So a run of the same kind collapses to one block, and coordination sitting
     *between* two blocks of the same kind is absorbed into that merge rather
     than splitting it. Coordination between two *different* kinds survives as
     its own block — it is a real boundary there.

Why absorb coordination at all: an agent reads five files, writes a TodoWrite,
then reads four more. Without absorption that is three blocks and looks like a
context switch. It is one stretch of reading.

Assistant prose counts as coordination, so the same absorption applies to it:
prose between two reads is swallowed by the read run, while prose separating two
different kinds stays visible as the boundary it marks.
"""

from __future__ import annotations

from typing import Optional

from Final_app.analysis.classify import classify_events
from Final_app.ir.models import (
    COORDINATION,
    EV_TOOL_USE,
    SUBAGENT,
    Block,
    Event,
    Session,
)

# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------

# What a block of each kind is called in the UI. The sketch's legend wording.
KIND_LABELS = {
    "read": "read",
    "write": "write",
    "execute": "execute",
    "coordination": "coordination",
    "subagent": "subagents",
    "user_chat": "chatting with user",
}


def _describe(event: Event) -> str:
    """A short human label for one event, used as the block's tooltip line."""
    if event.tool is not None:
        target = (event.tool.input.get("file_path")
                  or event.tool.input.get("pattern")
                  or event.tool.input.get("command")
                  or event.tool.input.get("description")
                  or "")
        target = " ".join(str(target).split())
        return f"{event.tool.name} {target}".strip()[:120]
    return " ".join((event.text or "").split())[:120]


def _label_for(kind: str, events: list[Event]) -> str:
    """A merged block is named by its kind plus how much it covers."""
    base = KIND_LABELS.get(kind, kind)
    if len(events) == 1:
        detail = _describe(events[0])
        return f"{base} · {detail}" if detail else base
    return f"{base} · {len(events)} steps"


# ----------------------------------------------------------------------
# Pass 3 — merging
# ----------------------------------------------------------------------

def merge_neighbours(
    kinded: list[tuple[Event, str, Optional[float]]]
) -> list[Block]:
    """Collapse runs of one kind, absorbing coordination that sits inside a run.

    ``kinded`` is ``(event, kind, confidence)`` in timeline order.

    The absorption is decided by lookahead, not by rewriting kinds in place: a
    coordination run is absorbed only when the kind *before* it equals the kind
    *after* it. At the start or end of a session there is no such pair, so it
    stays its own block.
    """
    if not kinded:
        return []

    # Group into runs of identical kind first; merging then reasons about runs
    # rather than individual events, which makes the lookahead a single step.
    runs: list[tuple[str, list[tuple[Event, Optional[float]]]]] = []
    for event, kind, confidence in kinded:
        if runs and runs[-1][0] == kind:
            runs[-1][1].append((event, confidence))
        else:
            runs.append((kind, [(event, confidence)]))

    merged: list[tuple[str, list[tuple[Event, Optional[float]]]]] = []
    index = 0
    while index < len(runs):
        kind, members = runs[index]

        if (kind == COORDINATION
                and merged
                and index + 1 < len(runs)
                and merged[-1][0] == runs[index + 1][0]
                and merged[-1][0] != COORDINATION):
            # Absorbed: this coordination and the run after it join the block
            # already open behind us.
            merged[-1][1].extend(members)
            merged[-1][1].extend(runs[index + 1][1])
            index += 2
            continue

        if merged and merged[-1][0] == kind:
            merged[-1][1].extend(members)
        else:
            merged.append((kind, list(members)))
        index += 1

    blocks: list[Block] = []
    for kind, members in merged:
        events = [event for event, _ in members]
        scores = [c for _, c in members if c is not None]
        blocks.append(Block(
            kind=kind,
            events=events,
            label=_label_for(kind, events),
            # A merged block is only as trustworthy as its least certain part.
            confidence=min(scores) if scores else None,
        ))
    return blocks


# ----------------------------------------------------------------------
# Passes 1 and 2 — threads and classification
# ----------------------------------------------------------------------

def _build_thread(events: list[Event], use_judge: bool) -> list[Block]:
    """Classify and merge one linear stream of events."""
    if not events:
        return []
    verdicts = classify_events(events, use_judge=use_judge)
    return merge_neighbours([
        (event, kind, confidence)
        for event, (kind, confidence) in zip(events, verdicts)
    ])


def build_blocks(session: Session, use_judge: bool = True) -> list[Block]:
    """The session's blocks, newest last, with subagents nested.

    The returned list is what the vertical bar renders top to bottom.
    """
    # Group subagent events by instance, preserving first-seen order.
    by_agent: dict[str, list[Event]] = {}
    for event in session.events:
        if event.is_subagent and event.agent_id:
            by_agent.setdefault(event.agent_id, []).append(event)

    # The spawning call belongs to the container it launched, not to the main
    # thread. Leaving it in produces a second, empty subagent block sitting
    # beside the real one — the delegation counted twice.
    spawn_by_agent: dict[str, Event] = {}
    main_events: list[Event] = []
    for event in session.events:
        if event.is_subagent:
            continue
        agent_id = _spawned_agent_id(event)
        if agent_id and agent_id in by_agent:
            spawn_by_agent[agent_id] = event
            continue
        main_events.append(event)

    blocks = _build_thread(main_events, use_judge)

    for agent_id, child_events in by_agent.items():
        inner = _build_thread(child_events, use_judge)
        # The spawn leads the container's events, so the container starts when
        # the delegation was issued rather than when the child's first line
        # landed, and its tokens include the call that created it.
        spawn = spawn_by_agent.get(agent_id)
        events = ([spawn] if spawn else []) + child_events
        container = Block(
            kind=SUBAGENT,
            events=events,
            agent_id=agent_id,
            description=_subagent_description(session, agent_id),
            inner_blocks=inner,
            label=f"{KIND_LABELS[SUBAGENT]} · {len(inner)} steps",
        )
        blocks = _insert_subagent(blocks, container)

    return blocks


def _spawned_agent_id(event: Event) -> Optional[str]:
    """The subagent id a spawn call produced, from its inlined result."""
    if event.tool is None or event.tool.result is None:
        return None
    return event.tool.result.spawned_agent_id or None


def _subagent_description(session: Session, agent_id: str) -> str:
    """The task this specific subagent was given.

    Matched via the spawn's ``agentId`` rather than by taking the first
    ``Agent`` call found — with several subagents in a session, the latter
    labels every one of them with the first one's task.
    """
    for event in session.events:
        if event.is_subagent or event.tool is None:
            continue
        if _spawned_agent_id(event) != agent_id:
            continue
        description = (event.tool.input.get("description")
                       or event.tool.input.get("subagent_type") or "")
        if description:
            return str(description)[:120]
    return ""


def _insert_subagent(blocks: list[Block], container: Block) -> list[Block]:
    """Place a subagent container in timeline order among the parent's blocks."""
    start = container.t_start
    if start is None:
        return blocks + [container]
    for position, block in enumerate(blocks):
        block_start = block.t_start
        if block_start is not None and block_start > start:
            return blocks[:position] + [container] + blocks[position:]
    return blocks + [container]


# ----------------------------------------------------------------------
# Rollups for the bar's Y-axis selector
# ----------------------------------------------------------------------

# The sketch's selector: "Y axis is [selector: tokens / time / money / messages]".
Y_METRICS = ("tokens", "time", "messages")


def block_metric(block: Block, metric: str) -> float:
    """One block's size under the selected Y-axis metric.

    Height is the only thing that changes when the selector moves — order and
    colour never do.
    """
    if metric == "time":
        return block.duration_s
    if metric == "messages":
        return float(block.message_count)
    return float(block.tokens.working)
