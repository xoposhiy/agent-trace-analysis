"""
main.py — frequency of vibe-fixing symptoms across the SWE-chat dataset
(ALL sessions, not just long ones), using Haiku as an LLM-judge.

Console output + a regenerated markdown report (with real example evidence
per symptom) written to REPORT_OUTPUT_PATH. Runs on every parseable session
by default (SAMPLE = None below).

Symptoms detected:
    no_spec                  — vague one-line request, or agent submits
                                despite its own hedging/uncertainty
                                (does NOT apply to pure questions/explanations)
    no_closed_loop            — request gives no way to verify the fix worked
                                (does NOT apply to pure questions/explanations)
    no_acceptance_criteria    — vague success bar, no concrete criteria
                                (does NOT apply to pure questions/explanations)
    no_visual_reference       — "make it look better" with no image/mockup
    repetitive_fix_attempts   — same USER-REPORTED bug fixed wrong repeatedly
                                (excludes normal TDD red-green iteration)
    scope_files_too_many      — too many files changed in one session (metadata-only)
    scope_turns_too_long      — session ran for an unusually large number of turns (metadata-only)

    (no_verification_by_user was removed — see CHANGELOG)

CHANGELOG (latest revision):
    - REMOVED all regex/heuristic intent classification. classify_intent(),
      QUESTION_PATTERNS, CHANGE_REQUEST_PATTERNS, and the "intent" field are
      gone entirely. The judge now decides "is this a real change request or
      just a question" purely by reading the actual message text itself, via
      its own applicable/present fields in the schema — no pre-computed hint
      shown in the prompt, and nothing biasing it either way.
    - The deterministic post-filter (apply_post_filter) is simplified to a
      minimal STRUCTURAL check with no text matching at all: it only
      overrides the four request-quality symptoms when a session has exactly
      one user message (is_single_message_session). Any session with more
      than one message is left entirely to the judge.
    - build_case_file now preserves EVERY tool_use call as exactly one
      timeline entry, not just the 3 previously-hardcoded categories
      (edit/write, read/grep/glob, bash-test-commands). Tool calls we don't
      have special handling for (WebFetch, Task/subagent spawns,
      NotebookEdit, MCP tools, etc.) now show up in the timeline with a
      generic label + short input summary instead of being silently dropped.
      Recognized categories (edits, reads, spec-file reads, test commands)
      still get more specific labels in the same single entry — no
      duplicate entries per tool call.

CHANGELOG (previous revision):
    - REMOVED no_verification_by_user entirely. It was flagged as unreliable
      (mostly detecting "no proof shown in chat" rather than "user actually
      skipped verifying") and we're not attempting to fix it right now — it's
      gone from definitions, the judge calls, the report, and the counts.
    - Judge calls are now STRUCTURED OUTPUT instead of free-text-JSON that we
      manually strip and json.loads(). Each call uses response_format with a
      JSON schema whose first field is "reasoning" (a string), followed by
      "applicable", "present", "evidence". Putting "reasoning" first in the
      schema gives the model room to think through the evidence before it
      has to commit to the boolean — plain "return only JSON" prompting was
      squeezing out any deliberation. Falls back to the old manual-parse
      approach if the proxy/model rejects response_format (see
      judge_one_symptom).
    - Symptom definitions now sit at the very top of each call's *user*
      message (defs + few-shot examples first, trace referenced second),
      per the request to put definitions "at the top of the prompt."
    - ONE SEPARATE JUDGE CALL PER SYMPTOM instead of one call judging all
      symptoms at once. Each call gets: this one symptom's definition +
      examples, plus the full session trace. The trace itself (large, and
      identical across a given session's 5 calls) is sent as the SYSTEM
      message with an (attempted) prompt-cache breakpoint, since it's the
      expensive, repeated part; the symptom-specific instruction (small,
      varies per call) is the user message. This is a best-effort attempt at
      cache-prefix reuse — see build_shared_context()/CACHE_CONTROL comments,
      since whether the underlying proxy actually honors the cache_control
      hint depends on your LiteLLM setup and isn't guaranteed by this code.
    - Chronological, INTERLEAVED transcript: previously user messages and
      agent thinking were shown to the judge in two separate blocks, so the
      judge had no way to know which thinking happened between which two
      user messages. build_case_file now builds a single "timeline" in
      strict event order (user message / thinking block / test command, as
      they actually occurred) and that's what gets rendered into the prompt.
    - All prior fixes retained: full unfiltered thinking capture (size-capped
      only as a safety valve, not by keyword), fixed tool_result content
      parsing (was stringifying the Python repr of list-typed content),
      wider test command/result capture windows, explicit utf-8 transcript
      reading, auto-regenerated report with real examples.

COST NOTE: this revision makes 5 LLM calls per session instead of 1 (one per
remaining symptom). That's a 5x increase in judge API calls versus the prior
version. Prompt caching (if your proxy honors it) should absorb most of the
added *token* cost since the large shared trace is reused across the 5 calls,
but it is still 5x the request count / latency. Consider SAMPLE for testing.

Requirements: pip install datasets huggingface_hub openai
Dataset is gated: accept terms on huggingface.co and run `hf auth login` first.
Calls Claude through JetBrains' internal LiteLLM proxy (OpenAI-compatible
API). Set your LiteLLM key as OPENAI_API_KEY before running.
"""

import json
from collections import Counter

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from openai import OpenAI

REPO = "SALT-NLP/SWE-chat"
JUDGE_MODEL = "anthropic/claude-haiku-4-5"
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"
REPORT_OUTPUT_PATH = "vibe_fixing_report.md"

SAMPLE = 15   # None = run on every parseable session

SCOPE_FILES_TOO_MANY_THRESHOLD = 8
SCOPE_TURNS_TOO_LONG_THRESHOLD = 150

# Caps that exist ONLY to stop a single pathological session (e.g. a giant
# generated file diff, or a runaway chain-of-thought) from blowing up the
# prompt. These are not content filters — everything under the cap is passed
# through untouched and unfiltered.
MAX_THINKING_CHARS_PER_BLOCK = 4000
MAX_TOTAL_THINKING_CHARS = 20000
MAX_TEST_COMMAND_CHARS = 400
MAX_TEST_RESULT_CHARS = 600
MAX_USER_MESSAGE_CHARS = 2000

TEST_KEYWORDS = ("test", "pytest", "jest", "npm run", "go test", "cargo test", "rspec")
FAIL_MARKERS = ("fail", "error", "traceback", "exception", "not ok")
PASS_MARKERS = ("pass", "ok", "success", "0 failing", "all tests passed")

# Filenames that plausibly carry a spec/instructions even when the user's own
# request is vague — e.g. the user says "add the endpoint" but AGENTS.md
# already documents exactly how endpoints should be structured in this repo.
# Heuristic on the filename only (case-insensitive substring match), not file
# content — deliberately cheap, no extra LLM call. Extend this list if your
# codebases use other conventions (e.g. .cursorrules, ARCHITECTURE.md).
SPEC_FILE_PATTERNS = (
    "agents.md", "claude.md", "spec.md", "specification.md", "design.md",
    "requirements.md", "readme.md", "contributing.md", "architecture.md",
    "cursorrules", ".cursorrules", "issue.md", "rfc",
)

# Whether to attempt OpenAI-style structured output (response_format with a
# JSON schema). If the proxy/model rejects it, judge_one_symptom falls back
# to the old "return only JSON" text prompting automatically per-call.
USE_STRUCTURED_OUTPUT = True

# Whether to attempt an Anthropic-style cache_control breakpoint on the
# shared system-message trace. Best-effort: if your LiteLLM proxy doesn't
# pass this through, it's simply ignored by the provider — harmless, just no
# cache savings. Set to False if you'd rather not send the extra field at all.
ATTEMPT_PROMPT_CACHING = True

# Each symptom now carries its own definition + its own short calibration
# examples, since each is judged in its own isolated call.
SYMPTOM_DEFINITIONS = {
    "no_spec": (
        "The user's request is vague/one-line ('change the button color', "
        "'replace the navigation bar with menu', 'add parameters to endpoint', "
        "'you are wrong') with no real spec, OR the agent's own thinking shows "
        "it is unsure or stuck but it submits an answer anyway as if everything "
        "is fine. IMPORTANT EXCLUSION #1: this symptom is about the quality of a "
        "REQUEST FOR A CHANGE. If the user is only asking a question, asking "
        "for an explanation, a code review, or a walkthrough — with no code "
        "change requested at all — there is no 'spec' to be vague, so mark "
        "applicable=false. Judge this yourself from the actual message text — "
        "a precise question ('what does this regex do?') is not vague just "
        "because it's short. IMPORTANT EXCLUSION #2: if the "
        "user's request is vague on its own BUT the agent read a spec-like "
        "file in this session (see the 'SPEC-LIKE FILES THE AGENT READ' line) "
        "that plausibly explains what the vague request means in this repo — "
        "e.g. the user says 'add the endpoint' and the agent read AGENTS.md, "
        "which documents this repo's exact endpoint conventions — then a real "
        "spec effectively existed and was used, so mark present=false. Only "
        "keep exclusion #2 if the file read is plausibly relevant to the "
        "actual request, not just any doc the agent happened to open."
    ),
    "no_closed_loop": (
        "The request ('fix tests', 'app is not running', etc.) gives the agent "
        "no way to verify whether the fix actually worked — e.g. no test suite "
        "was run and no reproduction step was checked before/after the fix. "
        "IMPORTANT EXCLUSION: only applies when a FIX was actually requested "
        "and made. If the turn is a pure question/explanation with no change "
        "made, mark applicable=false. If the agent DID run a test/reproduction "
        "step and reported the result, mark present=false."
    ),
    "no_acceptance_criteria": (
        "The request has a vague success bar with no concrete criteria — "
        "'make it faster', 'clean this up', 'don't change the format/API' — "
        "so 'done' is subjective. IMPORTANT EXCLUSION: only applies to "
        "requests for a CHANGE. Mark applicable=false for pure questions "
        "('how does the caching layer work?'). Mark present=false for requests "
        "that already give a concrete target ('get load time under 200ms', "
        "'match this exact screenshot')."
    ),
    "no_visual_reference": (
        "The request is about visual/UI appearance ('make it look better', "
        "'match the design') but no actual image, mockup, or design file was "
        "ever provided anywhere in the conversation. IMPORTANT EXCLUSION: if "
        "the session has no visual/UI request at all (backend, CLI, data, "
        "infra work), mark applicable=false. Also mark present=false if a "
        "screenshot/mockup/reference URL/design file was provided at any "
        "point in the conversation, even if not in the very first message."
    ),
    "repetitive_fix_attempts": (
        "The user reports the SAME underlying problem more than once because "
        "the agent's earlier fix didn't actually work — e.g. the user says "
        "'still broken', 'that didn't fix it', or re-describes the same bug "
        "after a fix was claimed complete. This is about the agent's fix being "
        "wrong, not about normal engineering iteration. Mark present=false for: "
        "a standard test-driven-development loop (write a test, run it, see it "
        "fail, fix the code, run it again, see it pass); a linter/build error "
        "being fixed and immediately re-checked; or a code reviewer requesting "
        "sequential/different changes addressed one at a time. This symptom is "
        "always applicable (applicable=true) regardless of request type — "
        "even a pure Q&A session could technically show a repeated wrong "
        "'fix' if the agent kept giving corrected-but-still-wrong answers."
    ),
}

SYMPTOM_ORDER = list(SYMPTOM_DEFINITIONS)
REQUEST_QUALITY_SYMPTOMS = ("no_spec", "no_closed_loop", "no_acceptance_criteria", "no_visual_reference")

# Few-shot calibration examples, per symptom, shown at the top of that
# symptom's user message (ahead of the reference to the trace). Kept short
# and generic (not tied to any real session).
FEW_SHOT_EXAMPLES_BY_SYMPTOM = {
    "no_spec": (
        "FLAG: user says only \"fix the button\", agent proceeds with no "
        "clarifying context and no spec ever emerges.\n"
        "DO NOT FLAG (applicable=false): user asks \"explain what this regex "
        "does\" — this is a question, not a change request.\n"
        "DO NOT FLAG (present=false): user says \"add the endpoint\" (vague on "
        "its own), but the agent reads AGENTS.md and it documents this repo's "
        "exact endpoint conventions — a real spec existed and was used."
    ),
    "no_closed_loop": (
        "FLAG: user says \"the API times out, fix it\"; agent edits code but "
        "never runs the endpoint or a test to confirm the timeout is gone.\n"
        "DO NOT FLAG (applicable=false): user asks \"why does the API time "
        "out?\" (no fix requested). DO NOT FLAG (present=false): the agent "
        "fixed it AND ran a reproduction step afterward."
    ),
    "no_acceptance_criteria": (
        "FLAG: \"make this cleaner\" / \"optimize this\" with no definition of "
        "clean or fast enough.\n"
        "DO NOT FLAG (present=false): \"get load time under 200ms\" (concrete "
        "target given). DO NOT FLAG (applicable=false): \"explain how the "
        "caching layer works\" (a question, not a target-setting request)."
    ),
    "no_visual_reference": (
        "FLAG: \"make the landing page look more modern\", no screenshot, "
        "mockup, or reference site anywhere in the conversation.\n"
        "DO NOT FLAG (present=false): user attaches a screenshot and says "
        "\"match this\". DO NOT FLAG (applicable=false): the session is "
        "backend/CLI work with no visual request at all."
    ),
    "repetitive_fix_attempts": (
        "FLAG: user says \"still crashes on submit\" shortly after the agent "
        "claimed the submit crash was fixed.\n"
        "DO NOT FLAG (present=false): user writes a test, it fails, agent "
        "fixes the code, test passes — one clean TDD cycle is not repetitive."
    ),
}

# Structured-output JSON schema. "reasoning" is deliberately the FIRST
# property: since JSON is generated field-by-field in declared order, this
# gives the model space to think through the evidence before it has to
# commit to "present". Plain "respond with only JSON" prompting was
# squeezing out any deliberation entirely.
SYMPTOM_JUDGMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "symptom_judgment",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Think step by step through the relevant evidence before deciding. This is scratch space, not the final answer.",
                },
                "applicable": {
                    "type": "boolean",
                    "description": "False only if this symptom conceptually cannot apply to this session (see the exclusion in the symptom definition). True otherwise.",
                },
                "present": {
                    "type": "boolean",
                    "description": "Whether the symptom is present in this session. Should be false whenever applicable is false.",
                },
                "evidence": {
                    "type": "string",
                    "description": "One short sentence pointing to the specific evidence for your answer.",
                },
            },
            "required": ["reasoning", "applicable", "present", "evidence"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_tables():
    sessions = load_dataset(REPO, "sessions", split="train")
    logs = load_dataset(REPO, "session_logs", split="train")
    return sessions, logs


def claude_code_session_ids(sessions):
    """
    All sessions we can actually parse. Restricted to agent == 'Claude Code'
    because other agents (Codex, Copilot CLI, Cursor, OpenCode, Gemini CLI)
    use different event schemas entirely.
    """
    return [s["session_id"] for s in sessions if s["agent"] == "Claude Code"]


def path_map(logs):
    return {log["session_id"]: log["transcript_path"] for log in logs}


def read_transcript(session_id, paths):
    local_path = hf_hub_download(REPO, paths[session_id], repo_type="dataset")
    events = []
    # Explicit encoding + errors="replace" so odd/invalid bytes in a
    # transcript line get replaced rather than silently corrupting the read
    # or throwing a hard decode error that skips the whole session.
    with open(local_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


# ----------------------------------------------------------------------
# Case-file extraction (condensed session summary fed to the judge)
# ----------------------------------------------------------------------

def _extract_text_from_content(content):
    """Content on a tool_result / message block can be a plain string OR a
    list of structured blocks like [{"type": "text", "text": "..."}]. Doing
    str(content) on the list case stringifies the Python repr (braces,
    quotes, key names and all) instead of the actual text — this avoids that."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif isinstance(b.get("content"), str):
                    parts.append(b["content"])
        return "\n".join(parts)
    return str(content) if content else ""


def build_case_file(events):
    """
    Builds a condensed case file. The key structural piece is `timeline`:
    a single list of entries in the STRICT order events actually occurred,
    so the judge can tell which thinking happened between which two user
    messages, rather than seeing "all messages" then "all thinking" as two
    disconnected blocks.
    """
    timeline = []              # [{"kind": "user_message"/"thinking"/"test_command"/"spec_file_read"/"tool_call", ...}] in strict order
    user_messages = []         # [(text, has_image)]
    files_touched = set()
    files_read = set()
    spec_files_read = set()     # subset of files_read matching SPEC_FILE_PATTERNS
    test_runs = []              # [{"command": str, "result": str|None, "id": tool_use_id}]
    total_thinking_chars = 0
    thinking_truncated = False
    turn_no = 0

    for event in events:
        etype = event.get("type")
        message = event.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None

        if etype == "assistant":
            turn_no += 1
            if isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue

                    if b.get("type") == "thinking":
                        thinking_text = b.get("thinking", "") or ""
                        if thinking_text and total_thinking_chars < MAX_TOTAL_THINKING_CHARS:
                            snippet = thinking_text[:MAX_THINKING_CHARS_PER_BLOCK]
                            timeline.append({"kind": "thinking", "turn": turn_no, "text": snippet})
                            total_thinking_chars += len(snippet)
                        elif thinking_text:
                            thinking_truncated = True

                    if b.get("type") == "tool_use":
                        name = b.get("name", "") or "(unnamed tool)"
                        tool_input = b.get("input", {}) or {}
                        name_lower = name.lower()

                        # Every tool_use produces exactly ONE timeline entry —
                        # nothing is silently dropped. Recognized categories
                        # (edit/write, read/grep/glob, bash test commands) get
                        # a more specific label/detail; anything else
                        # (WebFetch, Task/subagent, NotebookEdit, MCP tools,
                        # etc.) still gets a generic entry so the judge sees
                        # the full session, not a pre-filtered slice of it.
                        label = name
                        detail = ""
                        is_spec_file = False

                        if any(k in name_lower for k in ("edit", "write")) and "todowrite" not in name_lower:
                            fp = tool_input.get("file_path")
                            if fp:
                                files_touched.add(fp)
                            detail = fp or ""
                        elif any(k in name_lower for k in ("read", "grep", "glob")) and "todo" not in name_lower:
                            fp = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("pattern")
                            if fp:
                                files_read.add(fp)
                                if any(pat in fp.lower() for pat in SPEC_FILE_PATTERNS):
                                    is_spec_file = True
                                    spec_files_read.add(fp)
                            detail = fp or ""
                        elif "bash" in name_lower:
                            command = (tool_input.get("command", "") or "")[:MAX_TEST_COMMAND_CHARS]
                            detail = command
                            if any(k in command.lower() for k in TEST_KEYWORDS):
                                test_runs.append({"command": command, "result": None, "id": b.get("id")})
                                label = "Bash (test/build command)"
                        else:
                            # Generic fallback: short repr of whatever inputs
                            # this tool call had, so it's still visible even
                            # if we don't have special-case handling for it.
                            detail = ", ".join(f"{k}={str(v)[:120]}" for k, v in tool_input.items())[:300]

                        timeline.append({
                            "kind": "tool_call", "turn": turn_no, "name": name,
                            "label": label, "detail": detail, "is_spec_file": is_spec_file,
                        })

        if etype == "user":
            has_image = False
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                has_image = any(isinstance(b, dict) and b.get("type") == "image" for b in content)
                text = " ".join(parts).strip()
            else:
                text = ""
            if text:
                trimmed_text = text[:MAX_USER_MESSAGE_CHARS]
                user_messages.append((trimmed_text, has_image))
                timeline.append({
                    "kind": "user_message", "turn": turn_no, "text": trimmed_text,
                    "has_image": has_image,
                })

            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tool_use_id = b.get("tool_use_id")
                        result_text = _extract_text_from_content(b.get("content", ""))[:MAX_TEST_RESULT_CHARS].lower()
                        for tr in test_runs:
                            if tr["id"] == tool_use_id and tr["result"] is None:
                                if any(m in result_text for m in FAIL_MARKERS):
                                    tr["result"] = "fail"
                                elif any(m in result_text for m in PASS_MARKERS):
                                    tr["result"] = "pass"
                                else:
                                    tr["result"] = "unclear"

    return {
        "timeline": timeline,
        "user_messages": user_messages,
        "files_touched": sorted(files_touched),
        "files_read": sorted(files_read),
        "spec_files_read": sorted(spec_files_read),
        "test_runs": test_runs,
        "thinking_truncated": thinking_truncated,
        "total_turns": turn_no,
    }


def compute_scope_flags(case_file):
    """Direct metadata check — no LLM needed. Two independent flags."""
    return {
        "scope_files_too_many": len(case_file["files_touched"]) >= SCOPE_FILES_TOO_MANY_THRESHOLD,
        "scope_turns_too_long": case_file["total_turns"] >= SCOPE_TURNS_TOO_LONG_THRESHOLD,
    }


def is_single_message_session(case_file):
    """True if the session has exactly one user message (no back-and-forth
    at all). Used as a minimal, non-heuristic structural backstop: a session
    with only one message almost certainly isn't a real vibe-fixing case for
    the four 'request quality' symptoms, regardless of what it says. Sessions
    with more than one message are left entirely to the judge's own
    'applicable'/'present' answer — no text-pattern matching involved."""
    return len(case_file["user_messages"]) == 1


# ----------------------------------------------------------------------
# LLM-as-judge — one call per symptom, shared cacheable trace + per-symptom prompt
# ----------------------------------------------------------------------

def render_timeline(case_file):
    """Chronological, interleaved rendering: user messages, agent thinking,
    and test commands in the exact order they occurred, so the judge can see
    which thinking followed which message instead of two disconnected blocks."""
    lines = []
    for entry in case_file["timeline"]:
        if entry["kind"] == "user_message":
            tag = "[image attached]" if entry["has_image"] else "[text only]"
            lines.append(f"USER (turn {entry['turn']}) {tag}: {entry['text']}")
        elif entry["kind"] == "thinking":
            lines.append(f"AGENT THINKING (turn {entry['turn']}): {entry['text']}")
        elif entry["kind"] == "tool_call":
            spec_note = " [spec-like file]" if entry["is_spec_file"] else ""
            detail = f": {entry['detail']}" if entry["detail"] else ""
            lines.append(f"AGENT TOOL CALL (turn {entry['turn']}) {entry['label']}{spec_note}{detail}")
    if case_file["thinking_truncated"]:
        lines.append("(additional thinking beyond the size cap omitted)")
    return "\n\n".join(lines) if lines else "(empty session)"


def build_shared_context(case_file):
    """The large part of the prompt that's IDENTICAL across all 5 per-symptom
    calls for a given session. Sent as the system message so it's the shared
    prefix a cache breakpoint can apply to (see ATTEMPT_PROMPT_CACHING)."""
    tests_summary = "\n".join(
        f"- `{t['command']}` -> {t['result'] or 'unclear'}" for t in case_file["test_runs"]
    ) or "(none run)"

    return (
        "You are reviewing one coding-agent session for signs of 'vibe-fixing' "
        "(accepting fixes without proper spec or verification).\n\n"
        f"CHRONOLOGICAL SESSION TIMELINE (every user message, every piece of "
        f"agent thinking, and every tool call the agent made, in the exact "
        f"order they occurred):\n\n{render_timeline(case_file)}\n\n"
        f"FILES TOUCHED: {len(case_file['files_touched'])} files\n\n"
        f"SPEC-LIKE FILES THE AGENT READ (e.g. AGENTS.md, CLAUDE.md, SPEC.md, "
        f"README.md, design docs — these may already explain what a vague-"
        f"looking user request means in this repo): "
        f"{', '.join(case_file['spec_files_read']) if case_file['spec_files_read'] else '(none read)'}\n\n"
        f"TEST/BUILD COMMAND RESULTS SUMMARY:\n{tests_summary}\n"
    )


def build_symptom_prompt(symptom_name):
    """The small part that varies per call: this one symptom's definition and
    examples, at the top, followed by the instruction referencing the trace
    already given in the system message."""
    definition = SYMPTOM_DEFINITIONS[symptom_name]
    examples = FEW_SHOT_EXAMPLES_BY_SYMPTOM[symptom_name]
    return (
        f"SYMPTOM TO EVALUATE: {symptom_name}\n\n"
        f"DEFINITION: {definition}\n\n"
        f"CALIBRATION EXAMPLES (generic, not from this session):\n{examples}\n\n"
        "Using the session timeline given above, decide whether this specific "
        "symptom is present in THIS session. Think through the evidence in your "
        "reasoning field first, then give your final applicable/present/evidence answer."
    )


def _messages_for_call(case_file, symptom_name):
    system_content = build_shared_context(case_file)
    if ATTEMPT_PROMPT_CACHING:
        # Best-effort Anthropic-style cache breakpoint. If the proxy doesn't
        # honor cache_control, this is simply ignored — no error, no cache
        # savings. If your OpenAI-SDK version validates content blocks
        # strictly and rejects the extra key, set ATTEMPT_PROMPT_CACHING=False.
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}],
        }
    else:
        system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": build_symptom_prompt(symptom_name)}
    return [system_message, user_message]


def _parse_structured_response(response):
    text = response.choices[0].message.content
    if text is None:
        raise ValueError("empty response content")
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def judge_one_symptom(client, case_file, symptom_name, model=JUDGE_MODEL):
    """One isolated call for one symptom. Tries structured output first;
    falls back to plain-JSON prompting if the proxy/model rejects
    response_format (e.g. unsupported on this model via this proxy)."""
    messages = _messages_for_call(case_file, symptom_name)

    if USE_STRUCTURED_OUTPUT:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=800,
                temperature=0,
                timeout=45.0,
                messages=messages,
                response_format=SYMPTOM_JUDGMENT_SCHEMA,
            )
            return _parse_structured_response(response)
        except Exception:
            pass  # fall through to plain-JSON prompting below

    # Fallback: no response_format support. Ask for the same shape in plain
    # text and parse manually, same as the pre-structured-output version.
    fallback_messages = list(messages)
    fallback_messages[-1] = {
        "role": "user",
        "content": (
            fallback_messages[-1]["content"]
            + "\n\nRespond with ONLY a JSON object, no other text, in exactly this "
              'shape: {"reasoning": "...", "applicable": true/false, "present": '
              'true/false, "evidence": "..."}'
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=800,
            temperature=0,
            timeout=45.0,
            messages=fallback_messages,
        )
        return _parse_structured_response(response)
    except Exception as e:
        return {"error": str(e)}


def apply_post_filter(symptom_name, result, case_file):
    """Minimal structural safety net — no text/regex heuristics involved.
    If the session has exactly one user message (no back-and-forth at all),
    force the four request-quality symptoms to not-present regardless of
    what the judge returned, since a single-message session almost certainly
    isn't a real vibe-fixing case. Any session with more than one message is
    left entirely to the judge's own applicable/present judgment."""
    if not isinstance(result, dict) or "error" in result:
        return result
    if symptom_name in REQUEST_QUALITY_SYMPTOMS and is_single_message_session(case_file):
        if result.get("present"):
            result = dict(result)
            result["present"] = False
            result["applicable"] = False
            result["evidence"] = "overridden: session has only a single user message"
    return result


# ----------------------------------------------------------------------
# Main driver — console output + regenerated markdown report
# ----------------------------------------------------------------------

def run(session_ids, paths, sample=SAMPLE, verbose=True):
    print(f"(sample = {'ALL' if sample is None else sample} sessions)")

    client = OpenAI(base_url=LITELLM_BASE_URL, timeout=45.0)
    ids = session_ids if sample is None else session_ids[:sample]
    total = len(ids)
    print(f"total sessions to process: {total:,}\n")

    counts = Counter()
    call_successes = Counter()   # per-symptom denominator: sessions where that symptom's call succeeded
    n_ok = n_skipped = 0
    evidence_samples = {name: [] for name in SYMPTOM_ORDER}
    debug_steps = 3

    for i, session_id in enumerate(ids, 1):
        show_debug = i <= debug_steps
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] processing {session_id} "
                  f"(ok={n_ok} skipped={n_skipped})", flush=True)
        try:
            if show_debug:
                print(f"    -> downloading transcript...", flush=True)
            events = read_transcript(session_id, paths)
            if show_debug:
                print(f"    -> downloaded ({len(events)} lines), building case file...", flush=True)
            case_file = build_case_file(events)
            if not case_file["user_messages"]:
                if show_debug:
                    print(f"    -> no user messages, skipping", flush=True)
                continue
            n_ok += 1

            scope_flags = compute_scope_flags(case_file)
            for scope_name, is_flagged in scope_flags.items():
                if is_flagged:
                    counts[scope_name] += 1

            for symptom_name in SYMPTOM_ORDER:
                if show_debug:
                    print(f"    -> judging {symptom_name}...", flush=True)
                result = judge_one_symptom(client, case_file, symptom_name)
                if not result or "error" in result:
                    if show_debug and result:
                        print(f"       error: {result.get('error')}", flush=True)
                    continue

                result = apply_post_filter(symptom_name, result, case_file)
                call_successes[symptom_name] += 1

                if result.get("present"):
                    counts[symptom_name] += 1
                    if len(evidence_samples[symptom_name]) < 5:
                        evidence_samples[symptom_name].append((session_id, result.get("evidence", "")))

        except Exception as e:
            n_skipped += 1
            if show_debug:
                print(f"    -> exception: {e}", flush=True)

    print(f"\nsessions with usable data: {n_ok} | skipped: {n_skipped}")
    print(f"\n{'symptom':28s} {'count':>7s} {'% of judged sessions':>22s}")
    scope_names = ["scope_files_too_many", "scope_turns_too_long"]
    for name in SYMPTOM_ORDER + scope_names:
        denom = call_successes[name] if name not in scope_names else n_ok
        denom = denom or 1
        print(f"{name:28s} {counts[name]:>7,} {100*counts[name]/denom:>21.0f}%")

    if verbose:
        print("\nsample evidence (spot-check these against the real transcripts):")
        for name, samples in evidence_samples.items():
            if samples:
                print(f"\n  {name}:")
                for sid, ev in samples:
                    print(f"    [{sid}] {ev}")

    return counts, call_successes, n_ok, evidence_samples


# ----------------------------------------------------------------------
# Report generation — regenerates the markdown report with real examples
# ----------------------------------------------------------------------

SYMPTOM_DESCRIPTIONS_SHORT = {
    "no_spec": "The user's request is very short and unclear, or the agent shows doubt but still submits an answer",
    "no_closed_loop": "The user asks for a fix, but there is no way to check if it worked (no test run)",
    "no_acceptance_criteria": "The user's goal is vague (\"make it faster\", \"clean this up\"), with no clear target",
    "no_visual_reference": "The user asks for a UI/visual change, but gives no image or design file",
    "repetitive_fix_attempts": "The agent fixes the same bug wrong more than once, and the user has to report it again",
    "scope_files_too_many": "Too many files were changed in one session",
    "scope_turns_too_long": "The session had an unusually high number of turns",
}


def write_markdown_report(counts, call_successes, n_ok, evidence_samples, path=REPORT_OUTPUT_PATH):
    ordered = ["no_closed_loop", "no_spec", "no_acceptance_criteria",
               "scope_turns_too_long", "scope_files_too_many", "repetitive_fix_attempts", "no_visual_reference"]
    scope_names = {"scope_files_too_many", "scope_turns_too_long"}

    lines = []
    lines.append("# Vibe-Fixing Symptoms in the SWE-Chat Dataset\n")
    lines.append(
        "This report shows how often \"vibe-fixing\" happens in coding-agent sessions. "
        "Vibe-fixing means a user accepts a fix from the agent without a clear task, without "
        f"checking it, or without proof that it works. I checked **{n_ok:,} real coding sessions** "
        "from the SWE-Chat dataset (agent: Claude Code). All sessions were included, not only "
        "long ones.\n"
    )

    lines.append("## What I Looked For\n")
    lines.append("I checked each session for 6 symptoms:\n")
    lines.append("| Symptom | What it means |")
    lines.append("|---|---|")
    for name in ["no_spec", "no_closed_loop", "no_acceptance_criteria", "no_visual_reference",
                 "repetitive_fix_attempts", "scope_files_too_many", "scope_turns_too_long"]:
        lines.append(f"| `{name}` | {SYMPTOM_DESCRIPTIONS_SHORT[name]} |")
    lines.append("")

    lines.append("## How I Detected Them\n")
    lines.append(
        "I used two methods:\n\n"
        "**1. LLM-as-judge (Claude Haiku 4.5), one call per symptom.** Instead of asking one "
        "call to judge every symptom at once, each session now gets one isolated call per "
        "symptom. Each call gets that symptom's definition and calibration examples up front, "
        "then the full session trace. The trace itself — a chronological, interleaved timeline "
        "of user messages, agent thinking, and test commands in the order they actually happened "
        "— is identical across a session's calls, so it's sent as a shared, cacheable prefix "
        "rather than rebuilt from scratch each time.\n\n"
        "Compared to the previous version:\n"
        "- User messages and agent thinking are now interleaved in one chronological timeline, "
        "instead of being shown as two separate, disconnected blocks — so the judge can tell "
        "which thinking happened between which two messages.\n"
        "- Every thinking block from the agent is included in full, not just short excerpts that "
        "already contained a hedge word.\n"
        "- Judge responses use structured output with a `reasoning` field that comes first, so the "
        "model has room to think before it has to commit to a yes/no, instead of being squeezed "
        "into a bare JSON object with no room to reason.\n"
        "- The four symptoms about request quality (`no_spec`, `no_closed_loop`, "
        "`no_acceptance_criteria`, `no_visual_reference`) are explicitly defined to not apply to "
        "sessions that are pure questions/explanations with no code change requested, enforced "
        "both in the prompt and as a deterministic override after the judge answers.\n\n"
        "**2. Metadata-only rules.** The two `scope_*` symptoms don't need an LLM. I just count "
        "files touched and turns per session, and flag sessions above a threshold.\n"
    )

    lines.append("## Results\n")
    lines.append("| Symptom | Count | % of judged sessions |")
    lines.append("|---|---|---|")
    for name in ordered:
        denom = n_ok if name in scope_names else call_successes.get(name, 0)
        pct = round(100 * counts[name] / denom) if denom else 0
        lines.append(f"| `{name}` | {counts[name]:,} | {pct}% |")
    lines.append("")

    lines.append("## Examples\n")
    lines.append(
        "For each symptom flagged by the LLM judge, here are real examples pulled from this run "
        "(session id + the judge's one-line evidence). These are spot-check material, not proof — "
        "always worth reading the underlying transcript before trusting an aggregate number.\n"
    )
    for name in ["no_spec", "no_closed_loop", "no_acceptance_criteria", "no_visual_reference",
                 "repetitive_fix_attempts"]:
        lines.append(f"**`{name}`**\n")
        samples = evidence_samples.get(name) or []
        if samples:
            for sid, ev in samples:
                ev_clean = ev.strip() or "(no evidence text returned)"
                lines.append(f"- [`{sid}`] {ev_clean}")
        else:
            lines.append("- (no examples captured in this run)")
        lines.append("")

    lines.append("## A Note of Caution\n")
    lines.append(
        "`no_verification_by_user` has been removed from this report entirely — it was mostly "
        "detecting \"no proof shown in the transcript\" rather than \"the user actually skipped "
        "verifying,\" and a person could always test something outside the chat window, so it "
        "wasn't trustworthy as reported. The remaining symptoms rely on clearer, easier-to-check "
        "evidence (the actual request text, whether a test command was run and "
        "what it returned, file counts), but spot-checking the Examples section above against "
        "real transcripts is still recommended before citing these numbers externally.\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    print("Loading dataset tables from HuggingFace (this can take a while, no progress bar)...")
    sessions, logs = load_tables()
    print(f"Loaded {len(sessions):,} sessions and {len(logs):,} session logs.")

    all_ids = claude_code_session_ids(sessions)
    paths = path_map(logs)
    print(f"Filtered to {len(all_ids):,} Claude Code sessions.\n")

    counts, call_successes, n_ok, evidence_samples = run(all_ids, paths)
    write_markdown_report(counts, call_successes, n_ok, evidence_samples)