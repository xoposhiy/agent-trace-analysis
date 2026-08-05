"""The per-step projection behind the block detail page (``analysis.steps``).

What matters here is that the page can say *what happened* — the command, the
path, the failure — and that nothing it shows is a silent prefix of something
larger.
"""

from __future__ import annotations

from datetime import datetime, timezone

from Final_app.analysis.steps import (
    MAX_ARG_CHARS,
    MAX_RESULT_CHARS,
    block_steps,
    describe_event,
    step_summary,
)
from Final_app.ir.models import (
    EV_ASSISTANT,
    EV_TOOL_USE,
    Block,
    Event,
    ToolCall,
    ToolResult,
    Tokens,
)


def _event(tool_name=None, tool_input=None, result=None, text="",
           message_id="msg1", second=0, attributed=0) -> Event:
    return Event(
        uuid=f"u{second}",
        ts=datetime(2026, 8, 4, 9, 0, second, tzinfo=timezone.utc),
        type=EV_TOOL_USE if tool_name else EV_ASSISTANT,
        message_id=message_id,
        model="claude-opus-5",
        text=text,
        attributed_tokens=attributed,
        tokens=Tokens(output=10),
        tool=None if tool_name is None else ToolCall(
            id=f"t{second}", name=tool_name,
            input=tool_input or {}, result=result),
    )


# ----------------------------------------------------------------------
# What a step says it did
# ----------------------------------------------------------------------

def test_a_bash_step_carries_the_command_it_ran():
    event = _event("Bash", {"command": "pytest -k login", "description": "tests"})
    step = block_steps(Block(kind="execute", events=[event]))[0]

    assert step["tool"]["name"] == "Bash"
    command = [a for a in step["tool"]["arguments"] if a["name"] == "command"]
    assert command[0]["value"] == "pytest -k login"


def test_a_read_step_carries_the_path_and_the_lines_it_returned():
    event = _event("Read", {"file_path": "/repo/login.py"},
                   result=ToolResult(output="def login(): ...",
                                     file_path="/repo/login.py",
                                     num_lines=42, size_chars=8123))
    step = block_steps(Block(kind="read", events=[event]))[0]

    assert step["result"]["num_lines"] == 42
    assert step["result"]["size_chars"] == 8123
    assert step["headline"] == "Read /repo/login.py"


def test_a_failed_step_is_marked_as_failed():
    event = _event("Bash", {"command": "pytest"},
                   result=ToolResult(output="1 failed", is_error=True))
    step = block_steps(Block(kind="execute", events=[event]))[0]

    assert step["result"]["is_error"] is True


def test_prose_between_tool_calls_becomes_a_step_with_no_tool():
    step = block_steps(Block(kind="coordination",
                             events=[_event(text="Now the rewrite:")]))[0]

    assert step["tool"] is None
    assert step["text"] == "Now the rewrite:"


# ----------------------------------------------------------------------
# Nothing is shown as whole when it is a prefix
# ----------------------------------------------------------------------

def test_a_huge_write_argument_is_clipped_and_says_so():
    # A `Write` call's input holds the entire file body; one step can be
    # hundreds of kilobytes if it is passed through untouched.
    body = "x" * (MAX_ARG_CHARS * 4)
    event = _event("Write", {"file_path": "/repo/big.py", "content": body})
    step = block_steps(Block(kind="write", events=[event]))[0]

    content = [a for a in step["tool"]["arguments"] if a["name"] == "content"][0]
    assert len(content["value"]) == MAX_ARG_CHARS
    assert content["truncated"] is True
    assert content["full_chars"] == MAX_ARG_CHARS * 4


def test_a_short_argument_is_not_marked_truncated():
    event = _event("Bash", {"command": "ls"})
    step = block_steps(Block(kind="execute", events=[event]))[0]

    assert step["tool"]["arguments"][0]["truncated"] is False


def test_a_long_result_is_clipped_and_says_so():
    event = _event("Bash", {"command": "ls -R"},
                   result=ToolResult(output="y" * (MAX_RESULT_CHARS * 3)))
    step = block_steps(Block(kind="execute", events=[event]))[0]

    assert len(step["result"]["output"]) == MAX_RESULT_CHARS
    assert step["result"]["output_truncated"] is True


def test_a_non_string_argument_survives_as_text():
    # `replaceAll: false` and `limit: 40` are ordinary in real calls, and a
    # projection that assumed strings would drop or crash on them.
    event = _event("Edit", {"replace_all": False, "limit": 40})
    step = block_steps(Block(kind="write", events=[event]))[0]

    values = {a["name"]: a["value"] for a in step["tool"]["arguments"]}
    assert values == {"replace_all": "False", "limit": "40"}


# ----------------------------------------------------------------------
# The rollup the page leads with
# ----------------------------------------------------------------------

def test_the_summary_counts_failures_tools_and_files():
    events = [
        _event("Read", {"file_path": "/repo/a.py"},
               result=ToolResult(file_path="/repo/a.py", size_chars=100),
               second=1),
        _event("Read", {"file_path": "/repo/b.py"},
               result=ToolResult(file_path="/repo/b.py", size_chars=50),
               second=2),
        _event("Bash", {"command": "pytest"},
               result=ToolResult(is_error=True), second=3),
        _event(text="prose carries no tool", second=4),
    ]
    summary = step_summary(block_steps(Block(kind="read", events=events)))

    assert summary["steps"] == 4
    assert summary["tool_calls"] == 3
    assert summary["failed"] == 1
    assert summary["tools"] == {"Read": 2, "Bash": 1}
    assert summary["files"] == ["/repo/a.py", "/repo/b.py"]
    assert summary["result_chars"] == 150


def test_the_same_file_touched_twice_is_listed_once():
    events = [
        _event("Read", {"file_path": "/repo/a.py"},
               result=ToolResult(file_path="/repo/a.py"), second=1),
        _event("Edit", {"file_path": "/repo/a.py"},
               result=ToolResult(file_path="/repo/a.py"), second=2),
    ]
    summary = step_summary(block_steps(Block(kind="write", events=events)))

    assert summary["files"] == ["/repo/a.py"]


def test_steps_of_one_message_count_as_one_api_call():
    # Several tool calls in one assistant message share `message.id`, and that
    # id — not the step — is what `usage` is reported for.
    events = [_event("Read", {"file_path": f"/repo/{name}.py"},
                     message_id="msg_shared", second=index)
              for index, name in enumerate("abc")]
    summary = step_summary(block_steps(Block(kind="read", events=events)))

    assert summary["steps"] == 3
    assert summary["api_calls"] == 1


def test_an_empty_block_summarises_to_zeroes():
    summary = step_summary(block_steps(Block(kind="read", events=[])))

    assert summary["steps"] == 0
    assert summary["failed"] == 0
    assert summary["files"] == []


# ----------------------------------------------------------------------
# The shared describe helper
# ----------------------------------------------------------------------

def test_describe_falls_back_through_the_input_fields():
    assert describe_event(_event("Grep", {"pattern": "TODO"})) == "Grep TODO"
    assert describe_event(_event("Bash", {"command": "ls -l"})) == "Bash ls -l"
    assert describe_event(_event("Agent", {"description": "Map it"})) == "Agent Map it"
    assert describe_event(_event("Bash", {})) == "Bash"
