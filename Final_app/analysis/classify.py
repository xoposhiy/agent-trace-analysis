"""Classify events into the six block kinds.

Two stages, because the LLM should never be asked what a rule already knows:

  Stage A — rules. Unambiguous tools map straight to a kind. ``Read`` is always
            a read; ``Edit`` is always a write. No call, no cost, no VPN.
  Stage B — LLM judge. Only genuinely ambiguous calls, which in practice means
            ``Bash``: ``cat foo.py`` is a read, ``pytest`` is an execute, and
            only the command text can tell them apart.

Stage B is batched (one request per session, not per call) and cached on disk by
``hash(tool_name + input)``, so a given command is judged once ever — across
sessions, not just within one.

Everything degrades: with no key, no network, or no VPN, Stage B is skipped and
the ambiguous calls fall back to a conservative rule with ``confidence = None``,
which the UI renders as provisional. The bar always draws.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from Final_app.config import CACHE_DIR, JUDGE_MODEL, LLM_BASE_URL, LLM_TIMEOUT_S
from Final_app.ir.models import (
    COORDINATION,
    EV_ASSISTANT,
    EV_COMPACTION,
    EV_TOOL_USE,
    EV_USER,
    EXECUTE,
    READ,
    SUBAGENT,
    USER_CHAT,
    WRITE,
    Event,
)

# ----------------------------------------------------------------------
# Stage A — rules
# ----------------------------------------------------------------------

# Exact tool names, matched case-insensitively. Prefer exact names over
# substring matching: substring rules mislabel silently (a "TodoWrite" caught by
# a "write" rule becomes an edit, which it very much is not).
RULE_BY_TOOL: dict[str, str] = {
    # read — gathering information, changing nothing
    "read": READ,
    "grep": READ,
    "glob": READ,
    "notebookread": READ,
    "webfetch": READ,
    "websearch": READ,
    "toolsearch": READ,
    "ls": READ,

    # write — changing the working tree
    "edit": WRITE,
    "write": WRITE,
    "multiedit": WRITE,
    "notebookedit": WRITE,
    "applypatch": WRITE,

    # coordination — planning, bookkeeping, talking to the harness
    "todowrite": COORDINATION,
    "todoread": COORDINATION,
    "exitplanmode": COORDINATION,
    "enterplanmode": COORDINATION,
    "askuserquestion": COORDINATION,
    "task_create": COORDINATION,
    "taskcreate": COORDINATION,
    "taskupdate": COORDINATION,
    "skill": COORDINATION,

    # subagent — delegation. Claude Code renamed Task -> Agent; both appear on
    # disk, and keying on either alone silently finds nothing.
    "agent": SUBAGENT,
    "task": SUBAGENT,
    "workflow": SUBAGENT,
}

# Tools whose kind depends on their arguments rather than their name.
AMBIGUOUS_TOOLS = {"bash", "bashoutput", "shell", "run", "execute"}

# Fallback for a Bash command when the judge is unavailable. A first word in
# this set reads as exploration; anything else is treated as execution, which
# is the safer default (a build wrongly called a read hides real work, whereas
# a `cat` wrongly called an execute merely over-reports activity).
READ_ONLY_COMMANDS = {
    "cat", "grep", "rg", "find", "ls", "head", "tail", "wc", "less", "more",
    "tree", "pwd", "which", "file", "diff", "stat", "du", "df", "echo",
    "printenv", "env", "whoami", "date", "type", "realpath", "basename",
    "dirname", "readlink", "sort", "uniq", "cut", "awk", "sed", "jq",
}


def rule_kind(event: Event) -> Optional[str]:
    """The kind a rule is certain of, or None when the judge should decide."""
    if event.type == EV_USER:
        # Only a real user prompt is "chatting with user". These are the thin,
        # rare markers on the bar — the moments the human intervened.
        #
        # Claude Code puts a lot into the user role that the human never typed:
        # a skill's body, IDE context, slash-command echoes, local command
        # output. Those are the harness setting the agent up to work, so they
        # read as coordination — marking them as the human speaking put skill
        # activations on the bar as if the user had said them.
        return USER_CHAT if event.is_human_prompt else COORDINATION
    if event.type == EV_ASSISTANT:
        # Assistant prose ("Let me check whether…") is coordination.
        #
        # It is emphatically not user_chat: prose sits between almost every
        # tool call, so as its own kind nothing ever merges (150 blocks from a
        # 227-event session). As coordination the sketch's absorption rule
        # applies — prose between two reads is swallowed by the read run — and
        # prose that separates two *different* kinds stays visible as the
        # boundary it marks.
        return COORDINATION
    if event.type == EV_COMPACTION:
        # A structural break the bar draws as a divider, not a work block.
        return COORDINATION
    if event.type != EV_TOOL_USE or event.tool is None:
        return COORDINATION

    name = event.tool.name.strip().lower()
    if name in RULE_BY_TOOL:
        return RULE_BY_TOOL[name]
    if name in AMBIGUOUS_TOOLS:
        return None
    # An unknown tool is coordination rather than a guess — new tools appear
    # regularly and mislabelling them as work distorts the bar.
    return COORDINATION


def heuristic_shell_kind(command: str) -> str:
    """Best-effort kind for a shell command with no judge available.

    Deliberately crude — it exists so the bar renders offline, not to replace
    the judge. Anything with a pipe or chain is treated as execution, because a
    read-only first word tells you nothing about what follows it.
    """
    text = (command or "").strip()
    if not text:
        return EXECUTE
    if any(sep in text for sep in ("|", "&&", ";", ">", "$(")):
        return EXECUTE
    first = text.split()[0].rsplit("/", 1)[-1]
    return READ if first in READ_ONLY_COMMANDS else EXECUTE


# ----------------------------------------------------------------------
# Stage B — the judge
# ----------------------------------------------------------------------

_CACHE_FILE = CACHE_DIR / "tool_kinds.json"
_cache: Optional[dict] = None

# Long outputs are irrelevant to the classification and expensive to send.
MAX_INPUT_CHARS = 300
JUDGE_KINDS = (READ, WRITE, EXECUTE, COORDINATION)


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(_cache, dict):
            _cache = {}
    except (OSError, json.JSONDecodeError):
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp = _CACHE_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(_load_cache(), indent=0), encoding="utf-8")
        temp.replace(_CACHE_FILE)
    except OSError:
        pass


def tool_cache_key(tool_name: str, tool_input: dict) -> str:
    """Content-addressed, so an identical command is judged once ever."""
    payload = json.dumps(
        [tool_name.lower(), tool_input, JUDGE_MODEL],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def describe_call(event: Event) -> str:
    """The one-line rendering of a tool call the judge sees."""
    tool = event.tool
    command = tool.input.get("command") or ""
    description = tool.input.get("description") or ""
    if not command:
        command = json.dumps(tool.input, ensure_ascii=False)
    command = " ".join(str(command).split())[:MAX_INPUT_CHARS]
    return f"{tool.name}: {command}" + (f"   # {description}" if description else "")


def build_classify_prompt(calls: list[Event]) -> str:
    """Ask for one kind per call, as a JSON array. Pure string; no API call."""
    numbered = "\n".join(f"{i + 1}. {describe_call(e)}" for i, e in enumerate(calls))
    return (
        "You classify shell commands run by a coding agent into activity types.\n\n"
        "Types:\n"
        f"- {READ}: gathers information without changing anything "
        "(cat, grep, ls, git log, git diff, git status).\n"
        f"- {WRITE}: modifies files or the repository "
        "(sed -i, mv, rm, mkdir, git commit, git apply, writing a file).\n"
        f"- {EXECUTE}: runs or builds something and produces a result "
        "(tests, build, lint, type check, install, running a script, servers).\n"
        f"- {COORDINATION}: environment or session bookkeeping with no direct "
        "effect on the work (cd, export, echo of a status message, sleep).\n\n"
        "When a command chains several actions, classify it by its main effect: "
        "a command that edits then runs tests is execute.\n\n"
        f"Commands:\n{numbered}\n\n"
        "Answer with ONLY a JSON array of objects, no other text, one per "
        'command, in order:\n'
        '[{"n": 1, "kind": "read", "confidence": 0.9}, ...]'
    )


def llm_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def judge_calls(calls: list[Event]) -> dict[str, tuple[str, float]]:
    """Classify ambiguous calls in one request. Returns ``{key: (kind, conf)}``.

    Returns ``{}`` on any failure — the caller falls back to the heuristic.
    """
    if not calls or not llm_available():
        return {}

    try:
        from openai import OpenAI

        client = (OpenAI(base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_S)
                  if LLM_BASE_URL else OpenAI(timeout=LLM_TIMEOUT_S))
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=min(4000, 60 * len(calls) + 200),
            temperature=0,
            messages=[{"role": "user", "content": build_classify_prompt(calls)}],
        )
        text = (response.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        verdicts = json.loads(text)
    except Exception:
        return {}

    if not isinstance(verdicts, list):
        return {}

    out: dict[str, tuple[str, float]] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        try:
            index = int(verdict.get("n", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(calls):
            continue
        kind = str(verdict.get("kind", "")).strip().lower()
        if kind not in JUDGE_KINDS:
            continue
        try:
            confidence = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        event = calls[index]
        out[tool_cache_key(event.tool.name, event.tool.input)] = (kind, confidence)
    return out


# ----------------------------------------------------------------------
# The public entry point
# ----------------------------------------------------------------------

def classify_events(
    events: list[Event], use_judge: bool = True
) -> list[tuple[str, Optional[float]]]:
    """Kind and confidence for every event, in order.

    ``confidence`` is None for rule-decided events (a rule is not a guess) and
    for heuristic fallbacks (which the UI marks provisional); it is the judge's
    own number when the judge decided.
    """
    kinds: list[Optional[str]] = []
    confidences: list[Optional[float]] = []
    ambiguous: list[Event] = []

    # Stage A.
    for event in events:
        kind = rule_kind(event)
        kinds.append(kind)
        confidences.append(None)
        if kind is None:
            ambiguous.append(event)

    # Stage B — only what a rule could not settle, and only what is not cached.
    cache = _load_cache()
    if ambiguous:
        uncached, seen = [], set()
        for event in ambiguous:
            key = tool_cache_key(event.tool.name, event.tool.input)
            if key in cache or key in seen:
                continue
            seen.add(key)
            uncached.append(event)

        if uncached and use_judge:
            fresh = judge_calls(uncached)
            if fresh:
                cache.update({k: list(v) for k, v in fresh.items()})
                _save_cache()

    # Resolve every ambiguous slot: cache, then heuristic.
    for index, event in enumerate(events):
        if kinds[index] is not None:
            continue
        key = tool_cache_key(event.tool.name, event.tool.input)
        hit = cache.get(key)
        if hit:
            kinds[index] = hit[0]
            confidences[index] = hit[1]
        else:
            kinds[index] = heuristic_shell_kind(event.tool.input.get("command", ""))
            confidences[index] = None

    return list(zip(kinds, confidences))
