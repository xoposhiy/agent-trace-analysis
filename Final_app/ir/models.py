"""The TraceLens intermediate representation.

Three levels, all vendor-neutral. Adapters (``Final_app.adapters``) produce
level 1; everything above is computed from it.

Level 1 ``Event``  — one normalised transcript line / tool call.
Level 2 ``Block``  — one coloured rectangle on the session bar.
Level 3 ``Session`` — one session: metadata, summary, blocks, totals.

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

    @property
    def is_subagent(self) -> bool:
        return self.agent_id is not None

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
            "agent_id": self.agent_id,
            "depth": self.depth,
            "tool": self.tool.as_dict() if self.tool else None,
            "text": self.text,
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
            "message_count": self.message_count,
            "inner_blocks": [b.as_dict() for b in self.inner_blocks],
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

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "detail": self.detail,
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

    title: str = ""
    summary: str = ""

    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None

    events: list[Event] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    compaction_points: list[datetime] = field(default_factory=list)

    user_prompts: list[str] = field(default_factory=list)
    subagent_ids: list[str] = field(default_factory=list)

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
        return total

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
            "summary": self.summary,
            "model": self.model,
            "git_branch": self.git_branch,
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
            "duration_s": self.duration_s,
            "tokens": self.tokens.as_dict(),
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
        })
        return d
