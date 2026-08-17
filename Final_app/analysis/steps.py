"""One block's individual steps, as the block detail page shows them.

The bar answers *how big* a stretch of work was; this answers *what it did* —
the actual file that was read, the actual command that ran, and whether it
failed. That detail is deliberately absent from ``Block.as_dict()``: the
session payload carries every block, and a session of 300 blocks would ship
several megabytes of tool arguments to draw a bar that never displays them. It
is fetched per block instead, by ``GET /api/sessions/{id}/blocks/{index}``.

Two things are trimmed here rather than at the edge:

  * **Tool arguments.** ``ToolCall.input`` is kept whole in the IR because the
    judge needs the real Bash text. A ``Write`` call's input holds the entire
    file body, so a single step can be hundreds of kilobytes. Values are cut to
    ``MAX_ARG_CHARS`` with the original length reported alongside.
  * **Result text.** ``ToolResult.output`` is already only the head of what
    Claude Code recorded (see ``ToolResult.size_chars``), but a Bash step can
    still carry a quarter-megabyte of stdout on one line — measured on a real
    transcript, the five largest lines in one session were 258/191/174/157/151
    KB, all tool results.

Both keep a ``*_truncated`` flag so the page can say so rather than silently
showing a prefix as if it were the whole thing.
"""

from __future__ import annotations

from typing import Any, Optional

from Final_app.ir.models import Block, Event

# ----------------------------------------------------------------------
# Trimming
# ----------------------------------------------------------------------

# Enough to read a command, a path, or the first lines of a patch; short enough
# that a 40-step block stays a page rather than a download.
MAX_ARG_CHARS = 400
MAX_RESULT_CHARS = 600
MAX_TEXT_CHARS = 600


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """``(text, was_clipped)`` — never a bare prefix passed off as the whole."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def describe_event(event: Event) -> str:
    """A short human label for one event: the tool and what it acted on.

    Used both for a block's own label (``analysis.blocks``) and as the headline
    of a step on the detail page, so the two always agree.
    """
    if event.tool is not None:
        target = (event.tool.input.get("file_path")
                  or event.tool.input.get("pattern")
                  or event.tool.input.get("command")
                  or event.tool.input.get("description")
                  or "")
        target = " ".join(str(target).split())
        return f"{event.tool.name} {target}".strip()[:120]
    return " ".join((event.text or "").split())[:120]


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------

def _arguments(tool_input: dict[str, Any]) -> list[dict]:
    """A tool call's arguments, each clipped, longest-lived ones first.

    A list rather than a dict so the page can render them in a stable order
    without re-sorting, and so a duplicated key in a malformed payload cannot
    silently drop one.
    """
    arguments = []
    for name, value in tool_input.items():
        rendered = value if isinstance(value, str) else repr(value)
        clipped, truncated = _clip(rendered, MAX_ARG_CHARS)
        arguments.append({
            "name": name,
            "value": clipped,
            "truncated": truncated,
            "full_chars": len(rendered),
        })
    return arguments


def step_dict(event: Event, index: int) -> dict:
    """One step of a block, flattened for the detail page."""
    tool = event.tool
    result = tool.result if tool is not None else None

    text, text_truncated = _clip(" ".join((event.text or "").split()),
                                 MAX_TEXT_CHARS)
    output, output_truncated = _clip(result.output if result else "",
                                     MAX_RESULT_CHARS)

    return {
        "index": index,
        "uuid": event.uuid,
        "ts": event.ts.isoformat(),
        "type": event.type,
        # The API call this step was billed to. Several steps of one block
        # routinely share it — that is the unit ``usage`` is reported for.
        "message_id": event.message_id,
        "model": event.model,
        "headline": describe_event(event),
        "tool": None if tool is None else {
            "id": tool.id,
            "name": tool.name,
            "arguments": _arguments(tool.input),
        },
        "result": None if result is None else {
            "is_error": result.is_error,
            "output": output,
            "output_truncated": output_truncated,
            "file_path": result.file_path,
            "num_lines": result.num_lines,
            # What the result actually weighed on disk, which is what the next
            # API call paid to read back — usually far more than ``output``.
            "size_chars": result.size_chars,
            "spawned_agent_id": result.spawned_agent_id,
        },
        "text": text,
        "text_truncated": text_truncated,
        "attributed_tokens": event.attributed_tokens,
        "attributed_cache_read": event.attributed_cache_read,
        "content_tokens": event.content_tokens,
        "tokens": event.tokens.as_dict(),
    }


def block_steps(block: Block) -> list[dict]:
    """Every step of one block, in timeline order."""
    return [step_dict(event, index) for index, event in enumerate(block.events)]


# ----------------------------------------------------------------------
# Rollups the detail page leads with
# ----------------------------------------------------------------------

def step_summary(steps: list[dict]) -> dict:
    """Counts the page shows above the list, so it never has to scan it twice.

    ``failed`` is the one worth leading with: ``ToolResult.is_error`` is
    recorded all the way through the IR and has never been surfaced anywhere in
    the UI, so a block of five steps of which two failed looks exactly like a
    block where everything worked.
    """
    tool_steps = [step for step in steps if step["tool"] is not None]
    failed = [step for step in tool_steps
              if step["result"] and step["result"]["is_error"]]

    tools: dict[str, int] = {}
    for step in tool_steps:
        name = step["tool"]["name"]
        tools[name] = tools.get(name, 0) + 1

    # Distinct files, in first-touched order — "3 files" is more use at a
    # glance than twenty repeated paths.
    files: list[str] = []
    for step in tool_steps:
        path = _file_of(step)
        if path and path not in files:
            files.append(path)

    return {
        "steps": len(steps),
        "tool_calls": len(tool_steps),
        "failed": len(failed),
        "tools": tools,
        "files": files,
        "api_calls": len({step["message_id"] for step in steps
                          if step["message_id"]}),
        "result_chars": sum(step["result"]["size_chars"] for step in steps
                            if step["result"]),
    }


def _file_of(step: dict) -> Optional[str]:
    """The path a step touched, from the result envelope or the call itself."""
    result = step.get("result")
    if result and result.get("file_path"):
        return result["file_path"]
    for argument in step["tool"]["arguments"]:
        if argument["name"] == "file_path":
            return argument["value"]
    return None
