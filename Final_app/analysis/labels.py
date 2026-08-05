"""What a block is called on the bar.

A block's label is read in a tooltip, one line, while scanning a bar of a
hundred of them. It has to answer "what happened here" in a glance, and it is
*not* the place for detail — clicking the block opens the full step list, so
anything that needs a second line belongs there instead.

``read · 8 steps`` failed that test: it named the category and the volume, and
nothing about the work. Everything needed to do better is already parsed and
was being thrown away. Measured on real blocks in this repo's own sessions:

    read  · 8 steps    ->  read · bar.js, blocks.py +2 more · 8 steps
    write · 4 steps    ->  write · bar.js, bar.test.js · 4 steps
    execute · 5 steps  ->  execute · Inspect usage structure +4 · 5 steps

The subject is whichever of these the block actually has, in this order:

  1. **The files it touched.** The most concrete answer, and the only one that
     exposes repetition — ``bar.js ×3`` is three edits to one file, which is
     the signature of iterating on something that is not working. That was
     invisible in ``write · 4 steps``.
  2. **What a Bash call said it was doing.** Claude Code records a
     human-written ``description`` on every Bash invocation ("Confirm new tests
     fail before the fix"). Execute blocks are the largest group here and had
     no subject at all without it.
  3. **The mix of tools.** A last resort that at least distinguishes six reads
     from six greps.

Failures are appended last and unconditionally: ``is_error`` is recorded all
the way through the IR and reached nothing a user could see, so a block where
two of five steps failed looked exactly like one where everything worked.
"""

from __future__ import annotations

import collections
import os
from typing import Optional

from Final_app.analysis.steps import describe_event
from Final_app.ir.models import Event

# ----------------------------------------------------------------------
# Budget
# ----------------------------------------------------------------------
# One tooltip line. The subject is the part that can run away — a single
# absolute path in this repo is already 78 characters — so it is capped on its
# own rather than trusting a trim of the whole label to fall in a sane place.

MAX_SUBJECT = 44
MAX_LABEL = 80

# Naming three files is a list; naming five is a paragraph. Past this the
# label switches to counting instead.
MAX_NAMED_FILES = 2

# Below this many directories, naming files is still the clearer summary.
SCATTERED_DIRS = 3


def _clip(text: str, limit: int) -> str:
    """Trim on a word boundary where there is one, so labels do not end mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut if len(cut) >= limit // 2 else text[:limit]).rstrip(" ,·") + "…"


def clip_label(text: str) -> str:
    """Hold any label to the one line a tooltip gives it."""
    return _clip(text, MAX_LABEL)


# ----------------------------------------------------------------------
# Subjects
# ----------------------------------------------------------------------

def _touched_files(events: list[Event]) -> collections.Counter:
    """Basename -> how many times this block touched it, first-seen order.

    Basenames, not paths: the bar has one line and the detail page has the
    full path. Counting rather than de-duplicating is the whole point — the
    repeat is the finding.
    """
    counts: collections.Counter = collections.Counter()
    for event in events:
        if event.tool is None:
            continue
        path = event.tool.input.get("file_path")
        if not path and event.tool.result is not None:
            path = event.tool.result.file_path
        if path:
            counts[os.path.basename(str(path))] += 1
    return counts


def _directories(events: list[Event]) -> set[str]:
    directories = set()
    for event in events:
        if event.tool is None:
            continue
        path = event.tool.input.get("file_path")
        if not path and event.tool.result is not None:
            path = event.tool.result.file_path
        if path:
            directories.add(os.path.basename(os.path.dirname(str(path))))
    return directories


def _files_subject(events: list[Event]) -> Optional[str]:
    files = _touched_files(events)
    if not files:
        return None

    # Scattered across the tree: naming two of eight files misrepresents the
    # block. One real 8-step read spanned web/, analysis/, ir/ and tests/.
    if len(files) > MAX_NAMED_FILES and len(_directories(events)) >= SCATTERED_DIRS:
        return f"{len(files)} files in {len(_directories(events))} dirs"

    named = []
    for name, count in list(files.items())[:MAX_NAMED_FILES]:
        named.append(f"{name} ×{count}" if count > 1 else name)

    # The overflow hangs off the list without a comma — "a.js, b.js +2 more"
    # reads as a list and a remainder, "a.js, b.js, +2 more" as a broken list.
    subject = ", ".join(named)
    remaining = len(files) - len(named)
    return f"{subject} +{remaining} more" if remaining else subject


def _bash_subject(events: list[Event]) -> Optional[str]:
    """What the first Bash call said it was for, plus how many followed."""
    described = [event.tool.input.get("description") for event in events
                 if event.tool is not None
                 and event.tool.input.get("description")]
    if not described:
        return None

    subject = " ".join(str(described[0]).split())
    others = sum(1 for event in events if event.tool is not None) - 1
    if others <= 0:
        return subject

    # The description covers the *first* call only, so "+4" is what stops the
    # label reading as though one sentence described all five. It is therefore
    # reserved out of the budget before the description is trimmed, rather than
    # appended after and clipped away — which is what happened to
    # "Inspect usage structure in real transcript… · 5 steps".
    suffix = f" +{others}"
    return _clip(subject, MAX_SUBJECT - len(suffix)) + suffix


def _tools_subject(events: list[Event]) -> Optional[str]:
    counts = collections.Counter(event.tool.name for event in events
                                 if event.tool is not None)
    if not counts:
        return None
    return ", ".join(f"{count} {name}" if count > 1 else name
                     for name, count in counts.most_common(2))


def _prose_subject(events: list[Event]) -> Optional[str]:
    """For a block of pure coordination, what it actually said."""
    for event in events:
        if event.tool is None and event.text.strip():
            return " ".join(event.text.split())
    return None


def block_subject(events: list[Event]) -> Optional[str]:
    """The most informative one-liner this block can offer about itself."""
    for candidate in (_files_subject(events), _bash_subject(events),
                      _tools_subject(events), _prose_subject(events)):
        if candidate:
            return _clip(candidate, MAX_SUBJECT)
    return None


# ----------------------------------------------------------------------
# The label
# ----------------------------------------------------------------------

def _failed(events: list[Event]) -> int:
    return sum(1 for event in events
               if event.tool is not None
               and event.tool.result is not None
               and event.tool.result.is_error)


def compose_label(base: str, events: list[Event]) -> str:
    """``kind · subject · N steps · M failed``, trimmed to one line.

    The step count stays even when a subject is present: ``write · bar.js``
    reads as one edit when it was four, which is a worse label than the count
    alone was.
    """
    if not events:
        return base

    if len(events) == 1:
        detail = describe_event(events[0])
        single = f"{base} · {_clip(detail, MAX_SUBJECT)}" if detail else base
        return _clip(single, MAX_LABEL)

    parts = [base]
    subject = block_subject(events)
    if subject:
        parts.append(subject)
    parts.append(f"{len(events)} steps")

    # Last, so it is the thing the eye lands on at the end of the line, and
    # never dropped by the trim — the subject above it is already capped.
    failed = _failed(events)
    if failed:
        parts.append(f"{failed} failed")

    return " · ".join(parts)
