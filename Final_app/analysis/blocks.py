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

     Ahead of that grouping, one more relabel: a read of a file that comes
     *after* a write to that same file is treated as a write too, so an
     edit-then-verify loop (write, read it back, write again, read again, ...)
     collapses into one write block instead of a chain of alternating slivers.
     The same relabel also catches a test-development loop — write, run the
     test/build for that file, write again, run again — when the shell command
     names the file being worked on. See ``_relabel_verification_loop``.

Why absorb coordination at all: an agent reads five files, writes a TodoWrite,
then reads four more. Without absorption that is three blocks and looks like a
context switch. It is one stretch of reading.

Assistant prose counts as coordination, so the same absorption applies to it:
prose between two reads is swallowed by the read run, while prose separating two
different kinds stays visible as the boundary it marks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from Final_app.analysis.classify import classify_events
from Final_app.analysis.labels import clip_label, compose_label
from Final_app.ir.models import (
    COORDINATION,
    EV_TOOL_USE,
    EXECUTE,
    READ,
    SUBAGENT,
    WRITE,
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


def _label_for(kind: str, events: list[Event]) -> str:
    """A merged block is named by its kind, what it worked on, and how much.

    See ``analysis.labels`` for what counts as "what it worked on" and why the
    step count survives alongside it. A write block born from a test-loop
    relabel (see ``_relabel_verification_loop``) additionally gets a note
    naming that pattern, since "write · blocks.py x2 · 6 steps" reads as one
    edit run and hides that half those steps were test runs checking it.
    """
    note = "test development loop" if _is_test_development_loop(events) else None
    return compose_label(KIND_LABELS.get(kind, kind), events, note=note)


# ----------------------------------------------------------------------
# Pass 3 — merging
# ----------------------------------------------------------------------

# Kinds a coordination preamble may be folded into. Not ``user_chat`` (a human
# turn is never something the agent was preparing for) and not ``subagent``
# (containers are assembled separately, after merging).
_ABSORBING_KINDS = {"read", "write", "execute"}

# A coordination run this small, next to the work it introduces, is the agent
# announcing the call rather than doing anything: "Now the bar rewrite:" (20
# chars) ahead of a 10,433-char write. Measured over 407 such pairs in real
# sessions, coordination is 2.0% the size of the run it precedes at p10, 5.4%
# at p25, 17.2% at p50, 61.7% at p75 — so the preambles and the substantive
# blocks separate cleanly, and 25% absorbs 241 of the 407 while leaving
# standalone analysis (often several times larger than the call it precedes)
# untouched.
PREAMBLE_MAX_RATIO = 0.25


def _is_preamble(
    coordination: list[tuple[Event, Optional[float]]],
    work: list[tuple[Event, Optional[float]]],
) -> bool:
    """Was this coordination just introducing the work that follows it?

    Judged relative, not absolute: "Now I'll build it." ahead of a 14k-char
    write is a preamble, while the same sentence ahead of a one-line grep is
    half the substance of that stretch and stays visible.
    """
    work_chars = _content_chars([event for event, _ in work])
    if work_chars <= 0:
        # Nothing to be small *relative to*; keep the coordination visible
        # rather than folding it into a run with no content of its own.
        return False
    said = _content_chars([event for event, _ in coordination])
    return said <= work_chars * PREAMBLE_MAX_RATIO


def _content_chars(events: list[Event]) -> int:
    """How much a run actually says: prose plus the arguments of its calls.

    Characters, not tokens, on purpose — this decides a layout question about
    relative size, and a tokeniser here would make merging depend on the
    network.
    """
    total = 0
    for event in events:
        total += len(event.text)
        if event.tool is not None:
            total += len(event.tool.name)
            total += sum(len(str(value)) for value in event.tool.input.values())
    return total


# ----------------------------------------------------------------------
# Same-file "edit and verify" loops
# ----------------------------------------------------------------------
# Claude Code routinely re-reads a file right after writing it — confirming
# the edit landed, or checking the current state before the next one — then
# often edits it again: write, read it back, write again, read again. Each
# read is real, but it is work *about* the edit, not a return to exploration,
# so leaving it classified as a plain read chops one continuous edit into a
# chain of slivers (write · read · write · read · write — five blocks for one
# file touched five times in a row). This relabels every such read to write
# *before* the ordinary same-kind grouping below ever runs, so it folds the
# whole loop into a single write block the same way a run of five real writes
# would already merge.
#
# Verified against this project's own real transcripts, not assumed: two
# sessions here turned up 23 write blocks with at least one such interleaved
# re-read, the largest a 14-step Edit/Read/Edit/Read/.../Edit chain on
# ``ir/models.py`` alone that would otherwise have drawn as 7 alternating
# write/read slivers.
#
# The same shape shows up with a shell command standing in for the read: edit
# the code, run its test or build, edit again, run again. A ``Bash`` call
# carries no ``file_path`` of its own, so it cannot be matched by the equality
# check above — instead it joins the loop when its command text names the
# file being worked on (``pytest test_blocks.py`` after an edit to
# ``blocks.py``) *and* the command actually looks like a check.
#
# Checked against this project's own 35 real sessions (2026-08-27): naming the
# file alone, with no other condition, fired 45 times and both matched cleanly
# (genuine ``Edit`` -> ``pytest``/``node --check`` cycles) and misfired twice:
# a ``README.md`` edit followed by a ``grep``/``awk`` sanity check of the prose
# is a real edit-and-verify pair but is not a *test*, and a Bash call that
# restarted a server via ``Final_app.api.app:app`` joined an ``app.js`` loop
# purely because "app" appears in that unrelated module path. Requiring a
# recognised test/build/lint word in the same command fixes both: the doc
# check has none, and the server restart has none, while every real test/build
# run still does. This is more precise than the length cutoff it replaces
# would have been — a minimum stem length would have also thrown out genuine
# short names (``app.js`` is this project's own real entry point) rather than
# just the coincidence riding along with it.
#
# Still a heuristic, not exhaustive: a bare ``go test ./...`` or ``make`` with
# no filename will not join, since there is nothing here to match against. If
# that turns out to miss real loops once checked further, the fix is a judge
# call analogous to ``analysis.classify``'s Stage B, not a bigger word list.

# Tool names ``analysis.classify.AMBIGUOUS_TOOLS`` treats as shell execution —
# duplicated rather than imported so this module's relabelling logic does not
# reach into classify's judge-routing internals for one constant.
_SHELL_TOOL_NAMES = {"bash", "bashoutput", "shell", "run", "execute"}

# A command's words, lower-cased, split on anything that is not part of an
# identifier — so ``tests/test_blocks.py`` yields ``blocks`` as its own token
# rather than only as a substring of ``test_blocks.py``.
_COMMAND_TOKEN = re.compile(r"[^a-z0-9]+")

# Words that mark a command as actually running a test, build, or lint step,
# as opposed to some other action (serving, restarting, moving a file) that
# merely happens to name the file in passing. Matched as whole tokens, so
# ``check`` catches ``node --check`` without also catching every command that
# happens to contain "check" as part of a longer word.
_TEST_COMMAND_MARKERS = {
    "test", "tests", "pytest", "jest", "mocha", "vitest", "rspec", "phpunit",
    "unittest", "nosetests", "ctest", "tox",
    "check", "tsc", "mypy", "flake8", "pylint", "eslint", "lint",
    "build", "cargo",
}


def _command_mentions_stem(command: str, stems: set[str]) -> bool:
    """Does this shell command name one of these files, by its bare stem?

    Whole-token matching, not substring: a stem has to appear as one of the
    command's own words, so a short stem cannot match by accident inside an
    unrelated longer word (a path segment, a flag).
    """
    if not command or not stems:
        return False
    tokens = set(_COMMAND_TOKEN.split(command.lower()))
    return bool(tokens & stems)


def _command_verifies(command: str, stems: set[str]) -> bool:
    """Does this command both name the file *and* look like a check on it?

    The stronger condition the loop actually relabels on: naming the file is
    necessary but not sufficient, or a server restart that happens to mention
    the file's module path would count as testing it. See the section note
    above for the two real cases this rules out.
    """
    if not _command_mentions_stem(command, stems):
        return False
    tokens = set(_COMMAND_TOKEN.split(command.lower()))
    return bool(tokens & _TEST_COMMAND_MARKERS)


def _is_test_development_loop(events: list[Event]) -> bool:
    """Does this (already merged) write block pair an edit with a verifying
    shell command — the test-development half of the loop above?

    Re-derives the same match the relabel used to fold these events together,
    rather than threading a flag through the merge, so a label can ask the
    question directly from the events it already has.
    """
    stems = {Path(event.file_path).stem.lower()
             for event in events if event.file_path}
    if not stems:
        return False
    for event in events:
        if event.tool is None or event.tool.name.strip().lower() not in _SHELL_TOOL_NAMES:
            continue
        if _command_verifies(str(event.tool.input.get("command", "")), stems):
            return True
    return False


def _relabel_verification_loop(
    kinded: list[tuple[Event, str, Optional[float]]]
) -> list[tuple[Event, str, Optional[float]]]:
    """A read, or a shell command that verifies the file, becomes a write when
    it follows a write to that file.

    Only a write starts a loop — the read that happens *before* a file has
    been touched at all is exploration, not verification, and is left alone.
    From there the loop extends through any run of reads, writes, and
    verifying shell commands that keep naming the same file (see
    ``_command_verifies``), and coordination (prose, ``TodoWrite``) in between
    is transparent: it neither starts nor ends the loop, and is folded in the
    usual way once both sides of it read as the same kind. Anything else —
    a different file, a command that names the file without checking it, a
    different kind — ends the loop where it is.
    """
    kinded = list(kinded)
    index = 0
    while index < len(kinded):
        event, kind, _ = kinded[index]
        path = event.file_path if kind == WRITE else None
        if path is None:
            index += 1
            continue
        stem = {Path(path).stem.lower()}

        loop_end = index + 1
        to_relabel: list[int] = []
        cursor = index + 1
        while cursor < len(kinded):
            next_event, next_kind, _ = kinded[cursor]
            if next_kind == COORDINATION:
                cursor += 1
                continue
            if next_kind == EXECUTE:
                command = str(next_event.tool.input.get("command", "")) if next_event.tool else ""
                if not _command_verifies(command, stem):
                    break
                to_relabel.append(cursor)
                loop_end = cursor + 1
                cursor += 1
                continue
            if next_event.file_path != path:
                break
            if next_kind == WRITE:
                loop_end = cursor + 1
            elif next_kind == READ:
                to_relabel.append(cursor)
                loop_end = cursor + 1
            else:
                break
            cursor += 1

        for position in to_relabel:
            relabelled_event, _, confidence = kinded[position]
            kinded[position] = (relabelled_event, WRITE, confidence)
        index = loop_end
    return kinded


def merge_neighbours(
    kinded: list[tuple[Event, str, Optional[float]]]
) -> list[Block]:
    """Collapse runs of one kind, absorbing the coordination that glues them.

    ``kinded`` is ``(event, kind, confidence)`` in timeline order. Reads and
    matching shell commands that verify a just-written file are relabelled to
    writes first (see ``_relabel_verification_loop``), then two rules apply,
    in this order:

      1. **Preamble absorption.** A coordination run that is small next to the
         work run *after* it is that work's preamble, not a step of its own, so
         it joins it. This is what turns

             coordination · "Now the bar rewrite:"
             write · 5 steps
             coordination · "And the session page:"
             write · 3 steps

         into one ``write`` block instead of four, and it is why leading
         coordination no longer survives when it was only introducing a call.
         See ``PREAMBLE_MAX_RATIO``.

      2. **Interior absorption.** Coordination between two runs of the *same*
         kind does not split that kind, whatever its size — an agent that reads
         five files, writes a TodoWrite, then reads four more did one stretch
         of reading. Decided by lookahead on the kind before and after.

    Substantial coordination between two *different* kinds still survives as
    its own block: there it is a real transition, not glue.
    """
    if not kinded:
        return []

    kinded = _relabel_verification_loop(kinded)

    # Group into runs of identical kind first; merging then reasons about runs
    # rather than individual events, which makes the lookahead a single step.
    runs: list[tuple[str, list[tuple[Event, Optional[float]]]]] = []
    for event, kind, confidence in kinded:
        if runs and runs[-1][0] == kind:
            runs[-1][1].append((event, confidence))
        else:
            runs.append((kind, [(event, confidence)]))

    # Rule 1, as a relabel rather than a merge: a preamble simply *becomes* the
    # kind it introduces, and the regrouping below then folds it in. Doing it
    # this way keeps rule 2's lookahead reasoning about final kinds only.
    relabelled: list[tuple[str, list[tuple[Event, Optional[float]]]]] = []
    for index, (kind, members) in enumerate(runs):
        following = runs[index + 1] if index + 1 < len(runs) else None
        if (kind == COORDINATION
                and following is not None
                and following[0] in _ABSORBING_KINDS
                and _is_preamble(members, following[1])):
            kind = following[0]
        if relabelled and relabelled[-1][0] == kind:
            relabelled[-1][1].extend(members)
        else:
            relabelled.append((kind, list(members)))
    runs = relabelled

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

    return _group_subagent_runs(blocks)


def _group_subagent_runs(blocks: list[Block]) -> list[Block]:
    """Collapse a run of adjacent subagent containers into one band.

    Spawning three Explore agents in one message is a single act of delegation,
    and drawing it as three separate orange stripes reads as three separate
    decisions. They become one block; the individual agents survive in
    ``Block.agents`` for the detail page.

    Only *adjacent* containers merge. A subagent early in the session and
    another twenty blocks later are two different delegations, and nothing
    between them is swallowed.
    """
    grouped: list[Block] = []
    run: list[Block] = []

    for block in blocks:
        if block.kind == SUBAGENT:
            run.append(block)
            continue
        if run:
            grouped.append(_subagent_group(run))
            run = []
        grouped.append(block)
    if run:
        grouped.append(_subagent_group(run))
    return grouped


def _subagent_group(agents: list[Block]) -> Block:
    """One band covering ``agents``, which may be a single container.

    A run of one still gets wrapped, so the detail page has exactly one shape
    to render rather than "either a group or a lone agent".
    """
    inner = [child for agent in agents for child in agent.inner_blocks]
    events = [event for agent in agents for event in agent.events]
    only = agents[0] if len(agents) == 1 else None

    # Name the tasks, not the count: "2 agents" says nothing a glance at the
    # orange band does not already say, and the descriptions are right here.
    tasks = [agent.description for agent in agents if agent.description]
    if only is not None:
        label = (f"{KIND_LABELS[SUBAGENT]} · {tasks[0]}" if tasks
                 else only.label)
    elif tasks:
        label = (f"{KIND_LABELS[SUBAGENT]} · {', '.join(tasks)}"
                 f" · {len(agents)} agents")
    else:
        label = (f"{KIND_LABELS[SUBAGENT]} · {len(agents)} agents,"
                 f" {len(inner)} steps")
    label = clip_label(label)

    return Block(
        kind=SUBAGENT,
        events=events,
        label=label,
        agent_id=only.agent_id if only is not None else None,
        description=only.description if only is not None else "",
        inner_blocks=inner,
        agents=list(agents),
    )


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
