"""Block labels: what a block calls itself on the bar (``analysis.labels``).

The label is read in a one-line tooltip while scanning a hundred blocks, so
these tests are about what survives that line: the subject, the repetition,
the failures, and the cap that keeps them all on it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from Final_app.analysis.labels import MAX_LABEL, block_subject, compose_label
from Final_app.ir.models import (
    EV_ASSISTANT,
    EV_TOOL_USE,
    Event,
    ToolCall,
    ToolResult,
)

BASE_TIME = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)


def tool(name: str, tool_input: dict, second: int = 0,
         is_error: bool = False, result_path: str = "") -> Event:
    return Event(
        uuid=f"u{second}", ts=BASE_TIME + timedelta(seconds=second),
        type=EV_TOOL_USE,
        tool=ToolCall(id=f"t{second}", name=name, input=tool_input,
                      result=ToolResult(is_error=is_error,
                                        file_path=result_path)),
    )


def prose(text: str, second: int = 0) -> Event:
    return Event(uuid=f"p{second}", ts=BASE_TIME + timedelta(seconds=second),
                 type=EV_ASSISTANT, text=text)


# ----------------------------------------------------------------------
# Files — the first choice of subject
# ----------------------------------------------------------------------

def test_a_block_names_the_files_it_touched():
    events = [tool("Read", {"file_path": "/repo/web/bar.js"}, 1),
              tool("Read", {"file_path": "/repo/analysis/blocks.py"}, 2)]

    assert compose_label("read", events) == "read · bar.js, blocks.py · 2 steps"


def test_only_the_basename_is_shown():
    """A single absolute path in this repo is already 78 characters."""
    events = [tool("Edit", {"file_path": "/Users/x/deep/nested/tree/style.css"}, 1),
              tool("Edit", {"file_path": "/Users/x/deep/nested/tree/style.css"}, 2)]

    assert "/Users" not in compose_label("write", events)
    assert "style.css" in compose_label("write", events)


def test_a_repeated_file_is_counted_not_deduplicated():
    """Three edits to one file is iterating on something that is not working.

    ``write · 3 steps`` hid that completely.
    """
    events = [tool("Edit", {"file_path": "/repo/web/bar.js"}, index)
              for index in range(3)]

    assert compose_label("write", events) == "write · bar.js ×3 · 3 steps"


def test_extra_files_beyond_the_first_two_are_counted():
    events = [tool("Read", {"file_path": f"/repo/web/{name}.js"}, index)
              for index, name in enumerate(["a", "b", "c", "d"])]

    assert compose_label("read", events) == "read · a.js, b.js +2 more · 4 steps"


def test_files_scattered_across_the_tree_are_counted_not_named():
    """Naming two of eight misrepresents a block that ranged over the repo.

    A real 8-step read here spanned web/, analysis/, ir/ and tests/.
    """
    events = [
        tool("Read", {"file_path": "/repo/web/bar.js"}, 1),
        tool("Read", {"file_path": "/repo/analysis/blocks.py"}, 2),
        tool("Read", {"file_path": "/repo/ir/models.py"}, 3),
        tool("Read", {"file_path": "/repo/tests/conftest.py"}, 4),
    ]

    assert block_subject(events) == "4 files in 4 dirs"


def test_a_path_found_only_on_the_result_still_counts():
    """Edit records the path on the call; some tools only report it back."""
    events = [tool("NotebookEdit", {}, 1, result_path="/repo/nb.ipynb"),
              tool("NotebookEdit", {}, 2, result_path="/repo/nb.ipynb")]

    assert "nb.ipynb ×2" in compose_label("write", events)


# ----------------------------------------------------------------------
# Bash descriptions — the second choice
# ----------------------------------------------------------------------

def test_an_execute_block_uses_what_bash_said_it_was_doing():
    """Execute blocks touch no files, so they had no subject at all before."""
    events = [
        tool("Bash", {"command": "pytest -q",
                      "description": "Confirm new tests fail before the fix"}, 1),
        tool("Bash", {"command": "pytest -q tests/bar"}, 2),
        tool("Bash", {"command": "node --test"}, 3),
    ]

    label = compose_label("execute", events)

    assert "Confirm new tests fail before the fix +2" in label
    assert label.endswith("3 steps")


def test_a_lone_described_call_does_not_claim_others_followed():
    events = [tool("Bash", {"command": "ls", "description": "List files"}, 1),
              prose("and then", 2)]

    assert compose_label("execute", events) == "execute · List files · 2 steps"


# ----------------------------------------------------------------------
# Fallbacks
# ----------------------------------------------------------------------

def test_a_block_with_no_files_or_descriptions_names_its_tools():
    events = [tool("Grep", {"pattern": "TODO"}, 1),
              tool("Grep", {"pattern": "FIXME"}, 2),
              tool("Glob", {"pattern": "*.py"}, 3)]

    assert compose_label("read", events) == "read · 2 Grep, Glob · 3 steps"


def test_a_block_of_pure_prose_quotes_itself():
    events = [prose("Now the bar rewrite, starting with the layout", 1),
              prose("and then the legend", 2)]

    assert "Now the bar rewrite" in compose_label("coordination", events)


def test_a_single_step_block_keeps_its_one_line_description():
    events = [tool("Read", {"file_path": "/repo/login.py"}, 1)]

    assert compose_label("read", events) == "read · Read /repo/login.py"


def test_an_empty_block_is_just_its_kind():
    assert compose_label("read", []) == "read"


# ----------------------------------------------------------------------
# Failures — appended last, never dropped
# ----------------------------------------------------------------------

def test_failed_steps_are_named_in_the_label():
    """`is_error` reached nothing a user could see before this."""
    events = [tool("Bash", {"command": "pytest"}, 1, is_error=True),
              tool("Bash", {"command": "pytest"}, 2, is_error=True),
              tool("Bash", {"command": "pytest"}, 3)]

    assert compose_label("execute", events).endswith("3 steps · 2 failed")


def test_a_block_with_no_failures_says_nothing_about_them():
    events = [tool("Read", {"file_path": "/a.py"}, 1),
              tool("Read", {"file_path": "/b.py"}, 2)]

    assert "failed" not in compose_label("read", events)


def test_the_failure_count_survives_a_very_long_subject():
    """The subject is capped first, so the trim can never eat the failures."""
    events = [
        tool("Bash", {"command": "x", "description": "word " * 60}, 1, is_error=True),
        tool("Bash", {"command": "y"}, 2, is_error=True),
    ]

    assert compose_label("execute", events).endswith("2 steps · 2 failed")


# ----------------------------------------------------------------------
# The cap
# ----------------------------------------------------------------------

@pytest.mark.parametrize("events", [
    [tool("Bash", {"command": "x", "description": "n " * 90}, index)
     for index in range(4)],
    [tool("Read", {"file_path": "/repo/" + "long-name" * 12 + ".py"}, index)
     for index in range(3)],
    [prose("sentence that runs on and on " * 20, index) for index in range(2)],
])
def test_no_label_outruns_one_tooltip_line(events):
    label = compose_label("read", events)

    assert len(label) <= MAX_LABEL, f"{len(label)} chars: {label}"


def test_a_realistic_chat_message_survives_whole():
    """A regression test for the old 80-char tooltip budget: a normal,
    sentence-length chat message used to be cut around 80 characters even
    though the hover tip is now a floating box with room to wrap it in full."""
    message = ("Can we change the model to gemini luna in the .env file and"
               " make sure the API key still works after that?")
    events = [prose(message, 1)]

    assert compose_label("user_chat", events) == f"user_chat · {message}"


def test_a_clipped_subject_says_it_was_clipped():
    # MAX_SUBJECT is generous now (the hover tip wraps instead of eliding),
    # so this has to actually outrun it to exercise the clip.
    events = [tool("Bash", {"command": "x", "description": "w " * 200}, 1),
              tool("Bash", {"command": "y"}, 2)]

    assert "…" in compose_label("execute", events)


def test_a_clipped_label_does_not_end_mid_word():
    # Long enough to actually outrun MAX_SUBJECT and exercise the clip.
    description = "reticulating splines carefully " + "and then some more words " * 15
    events = [tool("Bash", {"command": "x", "description": description}, 1),
              tool("Bash", {"command": "y"}, 2)]

    subject = block_subject(events)
    assert "…" in subject
    assert not subject.rstrip("…").endswith(" ")
    assert "reticulating" in subject
