"""
classify.py — symptom/category definitions, prompt construction, and the
LLM-judge calls. Imports case_file.py for the timeline/rendering.

Runnable standalone:

    python classify.py test <session_id> [call_name]
    python classify.py dump-examples [n] [out_dir]

DESIGN NOTES (this revision):
    - The old 5 separate symptoms (no_spec, no_closed_loop,
      no_acceptance_criteria, no_visual_reference, repetitive_fix_attempts)
      are replaced by TWO grouped categories plus one unchanged single
      symptom:
        * not_enough_verification (ONE LLM call) — subcategories:
          not-tested, self-report, ask-for-manual-testing,
          repetitive-bug-fixes
        * not_enough_specification (ONE LLM call) — subcategories:
          no-spec-detected, repetitive-requirements-fixes, self-report
        * no_visual_reference (ONE LLM call, unchanged definition) — kept
          exactly as it was.
      scope_files_too_many / scope_turns_too_long remain metadata-only
      checks in case_file.py, untouched.
    - Each category call returns a "findings" LIST, not a single
      present/absent boolean — a session can show the same or different
      subcategories more than once, and we now capture every occurrence,
      not just the first. Each finding carries its own subcategory,
      location (message number(s)), and evidence.
    - "reasoning" is still the first schema property, before "findings",
      so the model has room to think before committing to specific findings.
    - "location" cites MESSAGE number(s) now (not "turn"), matching
      case_file.py's message-based numbering.
    - Structured output via response_format, with an automatic fallback to
      plain-JSON-in-text prompting if the model/proxy rejects it. The
      fallback resends the messages with cache_control stripped, so a proxy
      that chokes on that field can't take both attempts down with it.
    - Every failure and fallback is recorded in DIAGNOSTICS (and the first
      of each kind is printed), so a run can tell you WHY calls failed
      instead of silently degrading.
    - Findings coming back from the model are validated against the
      declared subcategory set before being returned — the fallback path is
      plain-JSON prompted with no enum enforcement, so a hallucinated
      subcategory must not reach the aggregation layer.
    - TEMPERATURE is configurable and can be None — some models (e.g. the
      one currently configured below) reject a custom temperature entirely
      and only support their default.
"""

import json
import time
from collections import Counter

from openai import OpenAI

import case_file as cf

JUDGE_MODEL = "openai/gpt-5.6-luna"
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"

# Some models reject a custom temperature outright (only support their
# default). Set to a number (e.g. 0) only if your model actually supports it.
TEMPERATURE = None

USE_STRUCTURED_OUTPUT = True
ATTEMPT_PROMPT_CACHING = True   # best-effort, unverified whether your proxy honors it

# Reasoning models spend part of this budget on hidden reasoning tokens before
# emitting a single character of JSON, so a tight cap truncates the answer and
# shows up as an unparseable-response "failure". The response itself is small;
# be generous.
MAX_COMPLETION_TOKENS = 6000

# "<stage>: <detail>" -> how many times it happened. run_experiment.py dumps
# this at the end of a run and stores it in results.json.
DIAGNOSTICS = Counter()
_DIAGNOSTICS_REPORTED = set()


def _note(stage, detail, verbose=True):
    """Record a failure/fallback reason. Prints only the first occurrence of
    each distinct kind so a systematic problem (bad auth, response_format
    rejected, responses truncating) is visible immediately without spamming
    one line per session."""
    key = f"{stage}: {detail}"
    DIAGNOSTICS[key] += 1
    if verbose and key not in _DIAGNOSTICS_REPORTED:
        _DIAGNOSTICS_REPORTED.add(key)
        print(f"    [diagnostic] {key}", flush=True)


# ----------------------------------------------------------------------
# Category / symptom definitions
# ----------------------------------------------------------------------

CATEGORY_DEFINITIONS = {
    "not_enough_verification": {
        "intro": (
            "This category is about whether the implementation was actually checked "
            "before being treated as finished. Look at the whole session timeline — "
            "tool calls, their results, and what the agent and user say — and decide "
            "which (if any) of the subcategories below occurred, and how many times."
        ),
        "subcategories": {
            "not-tested": (
                "The agent treats the task as finished/complete, but there is no "
                "evidence anywhere afterward of real verification — no dedicated test "
                "was written or run, and no manual check (running the app, hitting an "
                "endpoint, reproducing the original issue, etc.) appears in the tool "
                "calls. Only applies when a real implementation/fix was actually "
                "made — do not flag pure Q&A turns with no code change."
            ),
            "self-report": (
                "The agent itself explicitly says or clearly implies that some "
                "important part of the work was not tested — e.g. 'I haven't run "
                "this myself', 'this part is untested', 'you may want to double "
                "check X'. This is about the agent's own words admitting a gap, not "
                "your inference from silence."
            ),
            "ask-for-manual-testing": (
                "The agent explicitly asks the user to test or verify something "
                "manually themselves — e.g. 'please check that this works on your "
                "end', 'can you confirm the button behaves correctly now?'. This is "
                "about the agent directly handing verification to the human, rather "
                "than checking it itself."
            ),
            "repetitive-bug-fixes": (
                "After the agent considers an implementation finished, the user does "
                "their own manual testing and reports one or more bugs, requiring "
                "further fixes. Flag one occurrence per distinct finished-then-broken "
                "cycle. This is about a genuine regression report after the agent "
                "believed the work was done — normal iteration during initial, "
                "still-in-progress development is NOT this (e.g. writing code, "
                "running a test, fixing based on that test's result, is healthy "
                "iteration, not this subcategory)."
            ),
        },
        "few_shot": (
            "FLAG not-tested: agent says \"Done! The login flow should work now\" "
            "and ends the session — no test run, no manual check, nothing.\n"
            "FLAG self-report: agent says \"I've implemented the API endpoint but "
            "haven't tested the error-handling path.\"\n"
            "FLAG ask-for-manual-testing: agent says \"I can't run the UI myself — "
            "can you click through the signup flow and confirm it works?\"\n"
            "FLAG repetitive-bug-fixes: agent says the feature is done; user replies "
            "\"still crashes when I submit an empty form\"; agent fixes it; user "
            "later reports a second, different bug in the same feature — that's two "
            "separate occurrences.\n"
            "DO NOT FLAG: agent writes a function, writes a test for it, runs the "
            "test, sees it pass — that's verification working correctly.\n"
            "DO NOT FLAG: the whole session is the user asking \"what does this "
            "function do?\" with no change made."
        ),
    },
    "not_enough_specification": {
        "intro": (
            "This category is about whether the user's request was clear enough to "
            "act on. Look at the whole session timeline and decide which (if any) of "
            "the subcategories below occurred, and how many times."
        ),
        "subcategories": {
            "no-spec-detected": (
                "The user asks for an implementation or change, but the request is "
                "vague/underspecified ('add validation', 'make it better', 'handle "
                "errors properly') with no concrete detail on what's actually "
                "wanted. IMPORTANT EXCLUSION: does not apply to pure questions/"
                "explanations with no change requested. IMPORTANT EXCLUSION: if the "
                "timeline shows the agent reading a file (a Read/Grep/Glob tool "
                "call) whose name or content plausibly explains what the vague "
                "request means in this repo (e.g. AGENTS.md, a design doc, an issue "
                "description), do not flag this — a real spec effectively existed "
                "and was used. Use your own judgment on whether a given file read "
                "is actually plausibly relevant, don't assume every file read counts."
            ),
            "repetitive-requirements-fixes": (
                "The agent implements something, but the user pushes back because "
                "it doesn't match what they actually wanted — a requirements/"
                "understanding mismatch, NOT a technical bug — and this correction "
                "repeats across multiple turns. Flag one occurrence per distinct "
                "requirement-mismatch correction cycle. IMPORTANT: if the repeated "
                "correction is about something technically broken (a bug) rather "
                "than a misunderstanding of what was wanted, that belongs under the "
                "separate not_enough_verification category instead — do not flag it "
                "here."
            ),
            "self-report": (
                "The agent itself explicitly says or clearly implies it doesn't "
                "have enough information/specification to proceed confidently — "
                "e.g. 'I'm not entirely sure what you want here, I'll assume X', "
                "'this is ambiguous, going with the simplest interpretation'. This "
                "is about the agent's own words admitting a specification gap."
            ),
        },
        "few_shot": (
            "FLAG no-spec-detected: user says \"add validation\" with no detail on "
            "what should be validated or how; agent proceeds with no clarifying "
            "context and no spec ever emerges.\n"
            "DO NOT FLAG no-spec-detected: user asks \"explain what this validation "
            "function does\" — a question, not a change request.\n"
            "DO NOT FLAG no-spec-detected: user says \"add the endpoint\" (vague on "
            "its own), but the timeline shows the agent reading AGENTS.md, which "
            "documents this repo's exact endpoint conventions.\n"
            "FLAG repetitive-requirements-fixes: agent builds a search feature that "
            "searches only titles; user says \"I wanted it to search descriptions "
            "too\"; agent adjusts; user says \"actually I meant search everything "
            "including comments\" — a real requirements misunderstanding repeating, "
            "two occurrences.\n"
            "DO NOT FLAG repetitive-requirements-fixes: user reports the search "
            "feature throws an error on empty input — that's a bug, belongs under "
            "not_enough_verification instead.\n"
            "FLAG self-report: agent says \"You didn't specify the date format, so "
            "I'm assuming ISO 8601 — let me know if you meant something else.\""
        ),
    },
}

SINGLE_SYMPTOM_DEFINITIONS = {
    "no_visual_reference": {
        "definition": (
            "The request is about visual/UI appearance ('make it look better', "
            "'match the design') but no actual image, mockup, or design file was "
            "ever provided anywhere in the conversation. IMPORTANT EXCLUSION: if "
            "the session has no visual/UI request at all (backend, CLI, data, "
            "infra work), this does not apply. Also does not apply if a "
            "screenshot/mockup/reference URL/design file was provided at any "
            "point in the conversation, even if not in the very first message."
        ),
        "few_shot": (
            "FLAG: \"make the landing page look more modern\", no screenshot, "
            "mockup, or reference site anywhere in the conversation.\n"
            "DO NOT FLAG: user attaches a screenshot and says \"match this\". "
            "DO NOT FLAG: the session is backend/CLI work with no visual request "
            "at all."
        ),
    },
}

# Unified registry of every LLM call we make. "kind" distinguishes a
# grouped category (multiple subcategories, one call) from a single symptom
# (no subcategories, but same findings-list shape for uniform handling
# downstream).
CALLS = {
    "not_enough_verification": {"kind": "category", **CATEGORY_DEFINITIONS["not_enough_verification"]},
    "not_enough_specification": {"kind": "category", **CATEGORY_DEFINITIONS["not_enough_specification"]},
    "no_visual_reference": {"kind": "single", **SINGLE_SYMPTOM_DEFINITIONS["no_visual_reference"]},
}
CALL_ORDER = list(CALLS)

# Which calls get the deterministic single-message-session override. This
# mirrors the old REQUEST_QUALITY_SYMPTOMS idea: a session with no real
# back-and-forth almost certainly isn't a genuine specification/verification
# case. no_visual_reference is left out, same as before.
POST_FILTER_CALLS = ("not_enough_verification", "not_enough_specification")


def all_result_keys():
    """Every distinct (call, subcategory) key this file can produce, in a
    stable order. Used by run_experiment.py and report.py so both always
    agree on the full set of keys, even ones that end up with zero hits."""
    keys = []
    for call_name, info in CALLS.items():
        if info["kind"] == "category":
            keys.extend(f"{call_name}:{subcat}" for subcat in info["subcategories"])
        else:
            keys.append(call_name)
    return keys


def valid_subcategories(call_name):
    """The only subcategory values a finding for this call may carry. For a
    single symptom that's the call name itself, which keeps the finding shape
    identical across both kinds."""
    info = CALLS[call_name]
    if info["kind"] == "category":
        return set(info["subcategories"])
    return {call_name}


# ----------------------------------------------------------------------
# Structured-output schemas
# ----------------------------------------------------------------------

def _build_category_schema(subcategory_keys):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "category_findings",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Think step by step through the whole session before deciding. Scratch space, not the final answer.",
                    },
                    "findings": {
                        "type": "array",
                        "description": "One entry per distinct occurrence of any subcategory. Empty array if none occurred. The same subcategory can appear more than once if it happened multiple times.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subcategory": {"type": "string", "enum": list(subcategory_keys)},
                                "location": {"type": "string", "description": "Which message(s) this occurrence's evidence comes from, e.g. 'message 4' or 'messages 4-6'."},
                                "evidence": {"type": "string", "description": "One short sentence pointing to the specific evidence."},
                            },
                            "required": ["subcategory", "location", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["reasoning", "findings"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _build_single_symptom_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "symptom_findings",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Think step by step through the whole session before deciding.",
                    },
                    "findings": {
                        "type": "array",
                        "description": "One entry per distinct occurrence. Empty array if it never occurred.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "Which message(s) this occurrence's evidence comes from, e.g. 'message 4' or 'messages 4-6'."},
                                "evidence": {"type": "string", "description": "One short sentence pointing to the specific evidence."},
                            },
                            "required": ["location", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["reasoning", "findings"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _schema_for(call_name):
    info = CALLS[call_name]
    if info["kind"] == "category":
        return _build_category_schema(list(info["subcategories"]))
    return _build_single_symptom_schema()


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

def build_shared_context(case_file):
    """The large part of the prompt that's IDENTICAL across all 3 calls for
    a given session. Sent as the system message so it's the shared prefix a
    cache breakpoint can apply to (see ATTEMPT_PROMPT_CACHING)."""
    return (
        "You are reviewing one coding-agent session for signs of 'vibe-fixing' "
        "(accepting fixes without proper specification or verification).\n\n"
        "CHRONOLOGICAL SESSION TIMELINE (every user message, every piece of "
        "agent thinking, and every tool call the agent made — including its "
        "raw result — in the exact order they occurred, each tagged with the "
        f"message number it happened at):\n\n{cf.render_timeline(case_file)}\n\n"
        f"FILES TOUCHED (edited/written): {len(case_file['files_touched'])} files\n"
    )


def build_symptom_prompt(call_name):
    """The small part that varies per call: this category's (or symptom's)
    definition and examples, at the top, followed by the instruction
    referencing the trace already given in the system message."""
    info = CALLS[call_name]
    if info["kind"] == "category":
        sub_defs = "\n\n".join(f"- {k}: {v}" for k, v in info["subcategories"].items())
        return (
            f"CATEGORY TO EVALUATE: {call_name}\n\n"
            f"{info['intro']}\n\n"
            f"SUBCATEGORIES:\n{sub_defs}\n\n"
            f"CALIBRATION EXAMPLES (generic, not from this session):\n{info['few_shot']}\n\n"
            "Using the session timeline given above, find EVERY occurrence of any "
            "subcategory above — not just the first. Think through the evidence in "
            "your reasoning field first, then list each occurrence in 'findings' "
            "with its subcategory, location (message number(s)), and evidence. "
            "Return an empty findings list if none occurred."
        )
    return (
        f"SYMPTOM TO EVALUATE: {call_name}\n\n"
        f"DEFINITION: {info['definition']}\n\n"
        f"CALIBRATION EXAMPLES (generic, not from this session):\n{info['few_shot']}\n\n"
        "Using the session timeline given above, find EVERY occurrence of this "
        "symptom — not just the first. Think through the evidence in your "
        "reasoning field first, then list each occurrence in 'findings' with its "
        "location (message number(s)) and evidence. Return an empty findings list "
        "if it never occurred."
    )


def _messages_for_call(case_file, call_name):
    system_content = build_shared_context(case_file)
    if ATTEMPT_PROMPT_CACHING:
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}],
        }
    else:
        system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": build_symptom_prompt(call_name)}
    return [system_message, user_message]


def _message_text_length(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


def _parse_structured_response(response):
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise ValueError(
            "response truncated (finish_reason=length) — raise MAX_COMPLETION_TOKENS"
        )
    text = choice.message.content
    if text is None or not text.strip():
        raise ValueError("empty response content")
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def _plain_messages(messages):
    """The same messages with any structured content blocks flattened back to
    plain strings. The fallback attempt must NOT resend cache_control: if that
    field is what the proxy rejected, resending it verbatim would make the
    fallback fail for the exact same reason as the first attempt."""
    plain = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        plain.append({"role": message["role"], "content": content})
    return plain


def _validate_findings(call_name, result, verbose=True):
    """Normalise and validate what the model returned. The fallback path has no
    schema/enum enforcement, so an unknown subcategory can arrive; if it reached
    the aggregation layer it would create a result key nobody downstream knows
    about. Drop those (recording why) rather than propagating them."""
    if not isinstance(result, dict):
        return {"error": f"judge returned {type(result).__name__}, expected an object"}

    findings = result.get("findings")
    if findings is None:
        findings = []
    if not isinstance(findings, list):
        _note("validate", f"{call_name}: 'findings' was {type(findings).__name__}, not a list", verbose)
        findings = []

    allowed = valid_subcategories(call_name)
    is_single = CALLS[call_name]["kind"] == "single"

    clean = []
    for finding in findings:
        if not isinstance(finding, dict):
            _note("validate", f"{call_name}: finding was {type(finding).__name__}, not an object", verbose)
            continue
        # Single symptoms carry no subcategory of their own — inject the call
        # name so downstream code can treat every finding uniformly.
        subcategory = call_name if is_single else finding.get("subcategory")
        if subcategory not in allowed:
            _note("validate", f"{call_name}: unknown subcategory {subcategory!r}", verbose)
            continue
        clean.append({
            "subcategory": subcategory,
            "location": str(finding.get("location") or "n/a"),
            "evidence": str(finding.get("evidence") or ""),
        })

    result = dict(result)
    result["findings"] = clean
    return result


def _extra_kwargs():
    kwargs = {}
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE
    return kwargs


def judge_one_call(client, case_file, call_name, model=JUDGE_MODEL, verbose=True):
    """One isolated call for one category/symptom. Returns (result, meta).
    result = {"reasoning": str, "findings": [{"subcategory", "location",
    "evidence"}, ...]} — "subcategory" is always present and always a declared
    value, for both kinds, so downstream code can treat all calls uniformly.
    On failure, result is {"error": "..."} and meta is still populated."""
    info = CALLS[call_name]
    messages = _messages_for_call(case_file, call_name)
    prompt_chars = sum(_message_text_length(m) for m in messages)
    schema = _schema_for(call_name)
    start = time.perf_counter()

    result = None
    if USE_STRUCTURED_OUTPUT:
        try:
            response = client.chat.completions.create(
                model=model, max_tokens=MAX_COMPLETION_TOKENS, timeout=60.0,
                messages=messages, response_format=schema, **_extra_kwargs(),
            )
            result = _parse_structured_response(response)
        except Exception as e:
            _note("structured_output", f"{type(e).__name__}: {e}"[:200], verbose)
            result = None

    if result is None:
        # cache_control stripped here on purpose — see _plain_messages.
        fallback_messages = _plain_messages(messages)
        shape_hint = (
            '{"reasoning": "...", "findings": [{"subcategory": "...", "location": "...", "evidence": "..."}]}'
            if info["kind"] == "category" else
            '{"reasoning": "...", "findings": [{"location": "...", "evidence": "..."}]}'
        )
        fallback_messages[-1] = {
            "role": "user",
            "content": (
                fallback_messages[-1]["content"]
                + f"\n\nRespond with ONLY a JSON object, no other text, in exactly this shape: {shape_hint}"
            ),
        }
        try:
            response = client.chat.completions.create(
                model=model, max_tokens=MAX_COMPLETION_TOKENS, timeout=60.0,
                messages=fallback_messages, **_extra_kwargs(),
            )
            result = _parse_structured_response(response)
        except Exception as e:
            _note("fallback", f"{type(e).__name__}: {e}"[:200], verbose)
            elapsed = time.perf_counter() - start
            return (
                {"error": f"{type(e).__name__}: {e}"},
                {"elapsed_seconds": elapsed, "prompt_chars": prompt_chars},
            )

    result = _validate_findings(call_name, result, verbose)

    elapsed = time.perf_counter() - start
    meta = {"elapsed_seconds": elapsed, "prompt_chars": prompt_chars}
    return result, meta


def apply_post_filter(call_name, result, case_file):
    """Minimal structural safety net — no text/regex heuristics. If the
    session has exactly one user message, clear all findings for the two
    specification/verification calls, regardless of what the judge
    returned. no_visual_reference is untouched (never had this override)."""
    if not isinstance(result, dict) or "error" in result:
        return result
    if call_name in POST_FILTER_CALLS and cf.is_single_message_session(case_file):
        if result.get("findings"):
            result = dict(result)
            result["findings"] = []
    return result


# ----------------------------------------------------------------------
# Standalone modes: test one session, or dump prompts for N sessions
# ----------------------------------------------------------------------

def dump_examples(n=20, out_dir="examples"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset tables from HuggingFace...")
    sessions, logs = cf.load_tables()
    ids = cf.claude_code_session_ids(sessions)[:n]
    paths = cf.path_map(logs)

    stats = []
    for session_id in ids:
        case_file, download_s, parse_s = cf.build_case_file_with_timing(session_id, paths)
        shared = build_shared_context(case_file)
        parts = [f"=== SHARED SESSION CONTEXT (system message) — {len(shared):,} chars ===\n{shared}\n"]
        call_total = 0
        for call_name in CALL_ORDER:
            prompt = build_symptom_prompt(call_name)
            call_total += len(prompt)
            parts.append(f"=== PROMPT: {call_name} (user message) — {len(prompt):,} chars ===\n{prompt}\n")

        out_path = os.path.join(out_dir, f"{session_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        total_chars = len(shared) + call_total
        stats.append((session_id, len(shared), total_chars, download_s, parse_s))
        print(f"  wrote {out_path}  (shared: {len(shared):,} chars, "
              f"total across {len(CALL_ORDER)} calls: {total_chars:,} chars)")

    if stats:
        avg_shared = sum(s[1] for s in stats) / len(stats)
        avg_total = sum(s[2] for s in stats) / len(stats)
        avg_dl = sum(s[3] for s in stats) / len(stats)
        avg_parse = sum(s[4] for s in stats) / len(stats)
        print(f"\n{len(stats)} sessions dumped to '{out_dir}/'")
        print(f"avg shared-context size: {avg_shared:,.0f} chars")
        print(f"avg total prompt chars across {len(CALL_ORDER)} calls: {avg_total:,.0f} chars")
        print(f"avg download time: {avg_dl:.2f}s | avg parse time: {avg_parse:.2f}s")


def _test_one_session(session_id, call_name=None):
    print("Loading dataset tables from HuggingFace...")
    sessions, logs = cf.load_tables()
    paths = cf.path_map(logs)

    case_file, download_s, parse_s = cf.build_case_file_with_timing(session_id, paths)
    print(f"download: {download_s:.2f}s | parse: {parse_s:.2f}s | "
          f"user messages: {len(case_file['user_messages'])}\n")

    client = OpenAI(base_url=LITELLM_BASE_URL, timeout=60.0)
    calls = [call_name] if call_name else CALL_ORDER
    for name in calls:
        result, meta = judge_one_call(client, case_file, name)
        result = apply_post_filter(name, result, case_file)
        print(f"--- {name} ---")
        print(f"  elapsed: {meta['elapsed_seconds']:.2f}s | prompt size: {meta['prompt_chars']:,} chars")
        print(f"  {result}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    mode = sys.argv[1]
    if mode == "test":
        if len(sys.argv) < 3:
            print("usage: python classify.py test <session_id> [call_name]")
            sys.exit(1)
        sid = sys.argv[2]
        call = sys.argv[3] if len(sys.argv) > 3 else None
        _test_one_session(sid, call)
    elif mode == "dump-examples":
        n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        out_dir_arg = sys.argv[3] if len(sys.argv) > 3 else "examples"
        dump_examples(n_arg, out_dir_arg)
    else:
        print(__doc__)