"""The TraceLens intermediate representation.

Three levels, all vendor-neutral. Adapters (``Final_app.adapters``) produce
level 1; everything above is computed from it.

Level 1 ``Event``  — one normalised transcript line / tool call.
Level 2 ``Block``  — one coloured rectangle on the session bar.
Level 3 ``Session`` — one session: metadata, blocks, totals.

See ``DESIGN.md`` §4. The schema starts from Entire CLI's compact transcript
format (``cli/cmd/entire/cli/transcript/compact/compact.go:27``, MIT) and adds
the fields it drops: ``uuid``/``parent_uuid``, ``agent_id``/``depth``, cache
tokens, and a mandatory timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# ----------------------------------------------------------------------
# Block kinds — the six colours from the sketch
# ----------------------------------------------------------------------

READ = "read"
WRITE = "write"
EXECUTE = "execute"
COORDINATION = "coordination"
SUBAGENT = "subagent"
USER_CHAT = "user_chat"

BLOCK_KINDS = (READ, WRITE, EXECUTE, COORDINATION, SUBAGENT, USER_CHAT)


# ----------------------------------------------------------------------
# Level 1 — Event
# ----------------------------------------------------------------------

@dataclass
class Tokens:
    """Token accounting for one assistant message.

    ``cache_read`` and ``cache_creation`` are kept separate because they are
    priced differently — cost is meaningless without them, and they are exactly
    what Entire's compact format drops.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0

    @property
    def total(self) -> int:
        """Every token, cache reads included. Rarely the number to show."""
        return self.input + self.output + self.cache_read + self.cache_creation

    @property
    def working(self) -> int:
        """Tokens excluding cache reads — the honest "how much work" figure.

        ``cache_read`` is the whole prompt prefix re-read on *every* message, so
        summing it across a session counts the same context hundreds of times:
        in a real 227-message session here, total was 13.2M of which 12.5M
        (95%) was cache reads. Showing that as "tokens used" is off by ~18x.
        Cost still needs ``cache_read`` — it is billed, just at a lower rate —
        so both are kept.
        """
        return self.input + self.output + self.cache_creation

    def __add__(self, other: "Tokens") -> "Tokens":
        return Tokens(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_creation=self.cache_creation + other.cache_creation,
        )

    def as_dict(self) -> dict:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_creation": self.cache_creation,
            "total": self.total,
            "working": self.working,
        }


@dataclass
class ToolResult:
    """A tool's result, inlined into the call that produced it.

    Inlining (rather than leaving the result as a separate line) is what makes
    "one tool call == one visual block" possible. Borrowed from Entire's
    ``inlineToolResults`` (``compact.go:453``).
    """

    output: str = ""
    is_error: bool = False
    file_path: str = ""
    num_lines: int = 0

    # Characters in the *whole* result as Claude Code recorded it, including
    # payload this IR deliberately does not keep. A ``Read``'s contents live in
    # ``toolUseResult.file.content``, which ``output`` never carries: measured
    # on a real transcript, ``output`` retains 183,415 of 5,011,535 result
    # characters (3.7%), and the 4.6M missing are file bodies. Those bodies are
    # what the next API call pays to read, so cost attribution needs their size
    # even though the dashboard has no use for their text.
    size_chars: int = 0

    # Set only on an ``Agent``/``Task`` spawn: the child's instance id, taken
    # from ``toolUseResult.agentId``. This is the parent -> subagent link, and
    # it is what anchors a subagent's block at the call that launched it.
    spawned_agent_id: str = ""

    def as_dict(self) -> dict:
        return {
            "output": self.output,
            "is_error": self.is_error,
            "file_path": self.file_path,
            "num_lines": self.num_lines,
            "size_chars": self.size_chars,
            "spawned_agent_id": self.spawned_agent_id,
        }


@dataclass
class ToolCall:
    """One tool invocation.

    ``input`` is kept **whole** and unredacted: the judge needs the actual Bash
    command text to tell ``cat foo.py`` (read) from ``pytest`` (execute).
    """

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    result: Optional[ToolResult] = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "result": self.result.as_dict() if self.result else None,
        }


# Event types.
EV_USER = "user"
EV_ASSISTANT = "assistant"
EV_TOOL_USE = "tool_use"
EV_SYSTEM = "system"
EV_COMPACTION = "compaction"


@dataclass
class Event:
    """One normalised transcript line.

    A single assistant message that calls three tools becomes three ``Event``s
    of type ``tool_use`` (one per call) plus, if it also had prose, one
    ``assistant`` event. That way an Event maps 1:1 onto a candidate block.
    """

    uuid: str
    ts: datetime
    type: str
    vendor: str = "claude-code"
    vendor_version: str = ""

    parent_uuid: Optional[str] = None
    message_id: str = ""
    model: str = ""
    tokens: Tokens = field(default_factory=Tokens)

    # None on the main thread; the subagent's instance id inside a subagent.
    agent_id: Optional[str] = None
    depth: int = 0

    tool: Optional[ToolCall] = None
    text: str = ""

    # For ``EV_USER`` only: did a person type this, or did Claude Code inject
    # it into the user role? A skill's body, IDE context and slash-command
    # echoes all arrive as user lines with real text, and counting them as
    # "chatting with user" put skill activations on the bar as human turns.
    # See ``adapters.claude_code.is_human_prompt``.
    is_human_prompt: bool = True

    # The measured size of this one content block, from ``analysis.tokens``.
    # A different number from ``tokens``, on purpose: ``tokens`` is this
    # Event's share of what the message was *billed* and sums to the session
    # total; ``content_tokens`` is what this block alone tokenises to and does
    # not. 0 means "not measured yet" (needs the VPN), never "empty".
    content_tokens: int = 0

    # This Event's share of the session's billed ``working`` tokens, placed by
    # ``analysis.attribution`` on whatever caused the cost rather than on
    # whichever Event came first in its message. Sums (with
    # ``Session.overhead_tokens``) to ``Session.tokens.working`` exactly.
    attributed_tokens: int = 0

    # This Event's share of the session's billed ``cache_read``: what later calls
    # paid to re-read this content while it sat in the context window. Held apart
    # from ``attributed_tokens`` rather than added to it — the two are priced
    # differently and answer different questions — and sums to
    # ``Session.tokens.cache_read`` exactly. See ``analysis.attribution``.
    attributed_cache_read: int = 0

    # This Event's share of the session's dollar cost, priced per call at that
    # call's own model — unlike the two fields above, this is already the sum
    # of every billed channel (output, fresh input/cache-write, cache-read),
    # because a dollar has no equivalent of CLAUDE.md §7's "don't sum cache
    # reads into a token count" trap: a dollar spent re-reading cache is exactly
    # as real as a dollar spent on output. See ``analysis.attribution`` and
    # ``analysis.pricing``.
    attributed_cost: float = 0.0

    # This Event's share of its OWN THREAD's *current* context window — the
    # real, bounded size of that thread's last call (``fresh + cached``), not
    # a cross-call sum. Answers a different question from the two fields
    # above: not "what did this cost over the whole session" (cumulative,
    # unbounded) but "what does the context look like right now" (a
    # snapshot). Never mixed across threads: a subagent's own context window
    # is separate from its parent's. See ``analysis.attribution`` and
    # DESIGN.md §7.
    context_tokens: int = 0

    @property
    def is_subagent(self) -> bool:
        return self.agent_id is not None

    @property
    def file_path(self) -> Optional[str]:
        """The file this event's tool call touched, or ``None``.

        Most tools report it on the call itself (``Edit``'s ``file_path``
        input); some only report it back on the result (``NotebookEdit``, and
        anything whose call omits the path but whose result records it).
        ``None`` for an event with no tool, or a tool that touched no file
        (``Bash``, ``Grep``'s pattern-only search, ...).
        """
        if self.tool is None:
            return None
        path = self.tool.input.get("file_path")
        if not path and self.tool.result is not None:
            path = self.tool.result.file_path
        return str(path) if path else None

    def as_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "ts": self.ts.isoformat(),
            "type": self.type,
            "vendor": self.vendor,
            "vendor_version": self.vendor_version,
            "parent_uuid": self.parent_uuid,
            "message_id": self.message_id,
            "model": self.model,
            "tokens": self.tokens.as_dict(),
            "content_tokens": self.content_tokens,
            "agent_id": self.agent_id,
            "depth": self.depth,
            "tool": self.tool.as_dict() if self.tool else None,
            "text": self.text,
            "is_human_prompt": self.is_human_prompt,
            "attributed_tokens": self.attributed_tokens,
            "attributed_cache_read": self.attributed_cache_read,
            "attributed_cost": self.attributed_cost,
            "context_tokens": self.context_tokens,
        }


# ----------------------------------------------------------------------
# Level 2 — Block
# ----------------------------------------------------------------------

@dataclass
class Block:
    """One coloured rectangle on the vertical session bar.

    Neighbouring blocks of the same kind are merged into one (the sketch's
    "neighbouring blocks of the same type — join them in one block"), so a
    Block may cover many Events.
    """

    kind: str
    events: list[Event] = field(default_factory=list)
    label: str = ""

    # Judge confidence, or None when the kind came from a rule (or when the
    # judge was unavailable — see the VPN caveat in DESIGN.md §5).
    confidence: Optional[float] = None

    # Populated when kind == SUBAGENT.
    agent_id: Optional[str] = None
    description: str = ""
    inner_blocks: list["Block"] = field(default_factory=list)

    # One entry per subagent this block covers, each carrying that agent's own
    # ``inner_blocks``. Subagents spawned back to back are drawn as a single
    # band (three parallel Explore agents are one act of delegation, not three
    # stripes), so the band needs somewhere to keep them apart for the detail
    # page. ``inner_blocks`` above stays flat because it is what the bar paints.
    agents: list["Block"] = field(default_factory=list)

    @property
    def t_start(self) -> Optional[datetime]:
        return min((e.ts for e in self.events), default=None)

    @property
    def t_end(self) -> Optional[datetime]:
        return max((e.ts for e in self.events), default=None)

    @property
    def duration_s(self) -> float:
        start, end = self.t_start, self.t_end
        if start is None or end is None:
            return 0.0
        return (end - start).total_seconds()

    @property
    def tokens(self) -> Tokens:
        total = Tokens()
        for e in self.events:
            total = total + e.tokens
        return total

    @property
    def attributed_tokens(self) -> int:
        """This block's share of the session's billed working tokens.

        The honest "what did this stretch of work cost" figure: it includes
        both what the model generated here and what this block's tool results
        made the next call pay to read.
        """
        return sum(e.attributed_tokens for e in self.events)

    @property
    def attributed_cache_read(self) -> int:
        """This block's share of the session's billed cache reads.

        What every later call paid to re-read this block's content. Large for an
        early block that put a lot into the context window and stayed there, and
        the reason a cheap-looking ``Read`` early in a session is not cheap.
        """
        return sum(e.attributed_cache_read for e in self.events)

    @property
    def attributed_total(self) -> int:
        """Every billed token attributed to this block, cache reads included.

        The figure the bar's token axis paints: the whole context-window cost of
        this block. Kept as a property over the two channels rather than a third
        stored number, so it cannot drift from them.
        """
        return self.attributed_tokens + self.attributed_cache_read

    @property
    def attributed_cost(self) -> float:
        """This block's dollar share of the session's bill.

        Every billed channel already summed (see ``Event.attributed_cost``),
        each priced at the model of the call that was actually charged for it
        — correct even when a subagent used a different, differently-priced
        model than the main thread.
        """
        return sum(e.attributed_cost for e in self.events)

    @property
    def context_tokens(self) -> int:
        """This block's share of its own thread's CURRENT context window.

        Unlike ``attributed_total`` (cumulative across the whole session),
        this is bounded by one real call's actual billed size — see
        ``Event.context_tokens`` and DESIGN.md §7. A subagent band folds in
        that subagent's own thread-scoped share alongside its spawning
        call's, the same convention ``attributed_total`` already uses.
        """
        return sum(e.context_tokens for e in self.events)

    @property
    def content_tokens(self) -> int:
        """Measured size of this block's own content. See ``Event``.

        Every Event contributes its own measurement, so unlike ``tokens`` this
        is not a share of anything — a one-step ``read`` block reports what
        that single tool call actually tokenises to.
        """
        return sum(e.content_tokens for e in self.events)

    @property
    def content_tokens_measured(self) -> int:
        """How many of this block's Events have a measurement yet."""
        return sum(1 for e in self.events if e.content_tokens)

    @property
    def content_tokens_countable(self) -> int:
        """How many of this block's Events *can* be measured.

        Compaction markers have no content block, so a block of them is fully
        measured at zero. Without this the UI could not tell "not counted yet"
        from "nothing to count".
        """
        from Final_app.analysis.tokens import content_block

        return sum(1 for e in self.events if content_block(e) is not None)

    @property
    def message_count(self) -> int:
        return len(self.events)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "confidence": self.confidence,
            "agent_id": self.agent_id,
            "description": self.description,
            "t_start": self.t_start.isoformat() if self.t_start else None,
            "t_end": self.t_end.isoformat() if self.t_end else None,
            "duration_s": self.duration_s,
            "tokens": self.tokens.as_dict(),
            "attributed_tokens": self.attributed_tokens,
            "attributed_cache_read": self.attributed_cache_read,
            "attributed_total": self.attributed_total,
            "attributed_cost": self.attributed_cost,
            "context_tokens": self.context_tokens,
            "content_tokens": self.content_tokens,
            "content_tokens_measured": self.content_tokens_measured,
            "content_tokens_countable": self.content_tokens_countable,
            "message_count": self.message_count,
            "inner_blocks": [b.as_dict() for b in self.inner_blocks],
            "agents": [b.as_dict() for b in self.agents],
        }


# ----------------------------------------------------------------------
# Level 3 — Session
# ----------------------------------------------------------------------

@dataclass
class Problem:
    """One detected issue with a session. Detectors land in a later step."""

    id: str
    title: str
    severity: str = "info"  # info | low | medium | high
    detail: str = ""

    # Generic payload for whatever this detector needs the frontend to have —
    # a priced saving, an anchor event uuid, a flag. Kept untyped rather than
    # growing a bespoke dataclass field per detector, since every detector type
    # sketched for this panel (plan-mode, task-switch, sub-agent, no-closed-
    # loop) needs a different shape here.
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class Session:
    """One agent session."""

    session_id: str
    project: str
    path: str = ""
    git_branch: str = ""
    model: str = ""
    vendor: str = "claude-code"
    vendor_version: str = ""

    # Claude Code's own generated title, from the transcript's ``ai-title``
    # line. Free and already on disk — there is deliberately no LLM-written
    # session summary alongside it.
    title: str = ""

    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None

    events: list[Event] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    compaction_points: list[datetime] = field(default_factory=list)

    user_prompts: list[str] = field(default_factory=list)
    subagent_ids: list[str] = field(default_factory=list)

    # Whether the user ever switched this session into Claude Code's plan
    # mode (any thread — main or subagent). See ``adapters.claude_code``.
    used_plan_mode: bool = False

    # Usage from assistant messages that produced no Event — in practice, ones
    # whose content was entirely ``thinking``, which the IR drops. Held here so
    # the session total stays exact even though no block can show it. Rare:
    # 1 message of 106 in real session 5f12bde3.
    orphaned_tokens: Tokens = field(default_factory=Tokens)

    # Billed working tokens no block caused: the system prompt and tool
    # definitions the first call pays for. Held apart rather than smeared over
    # the blocks, which would make the session's opening look enormous.
    # See ``analysis.attribution``.
    overhead_tokens: int = 0

    # Message ids that emitted a thinking block. Thinking is billed as output
    # but Claude Code stores the block with an empty body, so part of what was
    # billed is invisible — those messages cannot be used to calibrate
    # characters per token. See ``analysis.attribution.calibrate``.
    thinking_message_ids: set = field(default_factory=set)

    @property
    def duration_s(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return (self.last_ts - self.first_ts).total_seconds()

    @property
    def tokens(self) -> Tokens:
        total = Tokens()
        for e in self.events:
            total = total + e.tokens
        return total + self.orphaned_tokens

    @property
    def attributed_cost(self) -> float:
        """The whole session's dollar bill, attributed across its Events.

        Equal to summing every block's ``attributed_cost`` — the same Events,
        just grouped — which is why this is the number the detail page's
        header shows: it agrees with the bar underneath it by construction,
        the same guarantee ``tokens``/``attributed_total`` already give for
        token counts. Excludes ``orphaned_tokens`` for the same reason the
        token attribution does: there is no Event to hang a rate on for usage
        that produced none, so a session with any is a dollar short of exact
        — rare enough (1 message of 106 in a real session) not to chase.
        """
        return sum(e.attributed_cost for e in self.events)

    @property
    def context_window_tokens(self) -> int:
        """The MAIN thread's current context window — one real call's size.

        Main-thread events only (``agent_id is None``): a subagent's context
        window is a separate, isolated window on its own call sequence, and
        summing it in here would silently reintroduce the same cross-call
        double counting this figure exists to avoid. See DESIGN.md §7 and
        ``analysis.attribution``.
        """
        return sum(e.context_tokens for e in self.events if e.agent_id is None)

    @property
    def tool_calls(self) -> list[Event]:
        return [e for e in self.events if e.type == EV_TOOL_USE]

    @property
    def max_severity(self) -> str:
        """Highest problem severity, for the list filter. ``none`` when clean."""
        order = ["none", "info", "low", "medium", "high"]
        worst = "none"
        for p in self.problems:
            if order.index(p.severity) > order.index(worst):
                worst = p.severity
        return worst

    def summary_dict(self) -> dict:
        """The lightweight shape the session-list page renders.

        Deliberately excludes ``events`` and ``blocks``: the list page loads
        every session, and shipping full event arrays would be megabytes.
        """
        return {
            "session_id": self.session_id,
            "project": self.project,
            "title": self.title,
            "model": self.model,
            "git_branch": self.git_branch,
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "duration_s": self.duration_s,
            "tokens": self.tokens.as_dict(),
            "attributed_cost": self.attributed_cost,
            "context_window_tokens": self.context_window_tokens,
            "message_count": len(self.events),
            "tool_call_count": len(self.tool_calls),
            "subagent_count": len(self.subagent_ids),
            "problem_count": len(self.problems),
            "max_severity": self.max_severity,
        }

    def as_dict(self) -> dict:
        d = self.summary_dict()
        d.update({
            "path": self.path,
            "vendor": self.vendor,
            "vendor_version": self.vendor_version,
            "blocks": [b.as_dict() for b in self.blocks],
            "problems": [p.as_dict() for p in self.problems],
            "compaction_points": [t.isoformat() for t in self.compaction_points],
            "subagent_ids": self.subagent_ids,
            "used_plan_mode": self.used_plan_mode,
        })
        return d
