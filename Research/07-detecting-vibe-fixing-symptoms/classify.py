"""
classify.py — symptom definitions, prompt construction, and the LLM-judge
call itself. Imports case_file.py for the timeline/rendering, but has no
knowledge of the overall experiment loop (that's run_experiment.py) or the
report format (that's report.py) — this file's only job is "given one
session's case file and one symptom name, decide present/not-present."

Runnable standalone in two modes:

    python classify.py test <session_id> [symptom_name]
        Judges one real session against one symptom (or all 5 if omitted),
        printing the full result (reasoning, applicable, present, location,
        evidence) plus how long the call took and how big the prompt was.

    python classify.py dump-examples [n] [out_dir]
        Builds (but does NOT send to the API) the prompts for the first n
        sessions (default 20, out_dir default "examples/") and writes one
        file per session containing the shared context + all 5 symptom
        prompts, with character counts. For manually inspecting how big
        traces actually are before spending API budget on them.

DESIGN NOTES (this revision):
    - No regex, no keyword lists. TEST_KEYWORDS/FAIL_MARKERS/PASS_MARKERS
      and SPEC_FILE_PATTERNS are both gone entirely. Every tool call and its
      raw result are already visible in the rendered timeline (case_file.py)
      — the judge decides for itself whether a given tool call was a test,
      whether it passed, and whether any file read plausibly served as a
      spec, purely from reading the raw content itself.
    - Structured output schema now includes "location": the judge cites
      which turn(s) its evidence comes from (e.g. "turn 4" or "turns 4-6"),
      using the turn numbers already assigned in case_file.py — no separate
      segmentation pass, no new numbering scheme. This is what a downstream
      application would use to highlight/jump to the relevant part of a
      session.
    - "reasoning" is still the first schema property, so the model has room
      to think before it has to commit to present/absent.
    - judge_one_symptom returns (result, meta) where meta carries
      elapsed_seconds and prompt_chars, so callers can track where time and
      size are actually going instead of guessing.
"""

import json
import time

from openai import OpenAI

import case_file as cf

JUDGE_MODEL = "openai/gpt-5.6-luna"
TEMPERATURE = None
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"

# Whether to attempt OpenAI-style structured output (response_format with a
# JSON schema). If the proxy/model rejects it, judge_one_symptom falls back
# to the old "return only JSON" text prompting automatically per-call.
USE_STRUCTURED_OUTPUT = True

# Whether to attempt an Anthropic-style cache_control breakpoint on the
# shared system-message trace. Best-effort and UNVERIFIED on your proxy —
# if it's not honored, this is simply ignored (no error, no cache savings).
ATTEMPT_PROMPT_CACHING = True

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
        "because it's short. IMPORTANT EXCLUSION #2: look at the tool calls in "
        "the timeline above — if the request is vague on its own BUT the agent "
        "read a file during this session (a Read/Grep/Glob tool call) whose "
        "name or content plausibly explains what the vague request means in "
        "this repo (e.g. AGENTS.md, a design doc, a spec file, an issue "
        "description), mark present=false — a real spec effectively existed "
        "and was used. Use your own judgment on which files are plausibly "
        "relevant based on what's shown in the timeline; don't assume every "
        "file read counts, only ones that plausibly cover this specific request."
    ),
    "no_closed_loop": (
        "The request ('fix tests', 'app is not running', etc.) gives the agent "
        "no way to verify whether the fix actually worked. Look at the tool "
        "calls in the timeline above — bash commands and their raw results are "
        "shown in full. Decide for yourself whether any of them constitute a "
        "test or reproduction check, and whether the result indicates success "
        "or failure — do not expect us to have pre-labeled this for you. "
        "IMPORTANT EXCLUSION: only applies when a FIX was actually requested "
        "and made. If the turn is a pure question/explanation with no change "
        "made, mark applicable=false. If the agent DID run something that "
        "plausibly verifies the fix and the result looks like a real check, "
        "mark present=false."
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
        "always applicable (applicable=true) regardless of request type."
    ),
}

SYMPTOM_ORDER = list(SYMPTOM_DEFINITIONS)
REQUEST_QUALITY_SYMPTOMS = ("no_spec", "no_closed_loop", "no_acceptance_criteria", "no_visual_reference")

FEW_SHOT_EXAMPLES_BY_SYMPTOM = {
    "no_spec": (
        "FLAG: user says only \"fix the button\", agent proceeds with no "
        "clarifying context and no spec ever emerges.\n"
        "DO NOT FLAG (applicable=false): user asks \"explain what this regex "
        "does\" — this is a question, not a change request.\n"
        "DO NOT FLAG (present=false): user says \"add the endpoint\" (vague on "
        "its own), but the timeline shows the agent reading AGENTS.md, which "
        "documents this repo's exact endpoint conventions — a real spec "
        "existed and was used."
    ),
    "no_closed_loop": (
        "FLAG: user says \"the API times out, fix it\"; the timeline shows "
        "edits but no bash command that looks like a test or reproduction "
        "step anywhere afterward.\n"
        "DO NOT FLAG (applicable=false): user asks \"why does the API time "
        "out?\" (no fix requested). DO NOT FLAG (present=false): a bash tool "
        "call afterward whose result plausibly shows the issue is resolved, "
        "even if the command itself isn't literally named 'test'."
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
# property so the model has room to think before committing to "present".
# "location" cites the turn number(s) the evidence comes from, using the
# turn numbers already assigned in case_file.py's timeline — this is how a
# downstream application knows WHERE in the session to highlight/jump to.
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
                "location": {
                    "type": "string",
                    "description": "Which turn(s) in the timeline this symptom's evidence comes from, e.g. 'turn 4' or 'turns 4-6'. Use 'n/a' if present is false.",
                },
                "evidence": {
                    "type": "string",
                    "description": "One short sentence pointing to the specific evidence for your answer.",
                },
            },
            "required": ["reasoning", "applicable", "present", "location", "evidence"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

def build_shared_context(case_file):
    """The large part of the prompt that's IDENTICAL across all 5 per-symptom
    calls for a given session. Sent as the system message so it's the shared
    prefix a cache breakpoint can apply to (see ATTEMPT_PROMPT_CACHING)."""
    return (
        "You are reviewing one coding-agent session for signs of 'vibe-fixing' "
        "(accepting fixes without proper spec or verification).\n\n"
        "CHRONOLOGICAL SESSION TIMELINE (every user message, every piece of "
        "agent thinking, and every tool call the agent made — including its "
        "raw result — in the exact order they occurred, each tagged with the "
        f"turn number it happened at):\n\n{cf.render_timeline(case_file)}\n\n"
        f"FILES TOUCHED (edited/written): {len(case_file['files_touched'])} files\n"
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
        "reasoning field first, then give your final applicable/present/"
        "location/evidence answer. For location, cite the turn number(s) from "
        "the timeline where your evidence comes from."
    )


def _messages_for_call(case_file, symptom_name):
    system_content = build_shared_context(case_file)
    if ATTEMPT_PROMPT_CACHING:
        # Best-effort Anthropic-style cache breakpoint — UNVERIFIED whether
        # your LiteLLM proxy actually honors this. If not, it's simply
        # ignored: no error, no cache savings.
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}],
        }
    else:
        system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": build_symptom_prompt(symptom_name)}
    return [system_message, user_message]


def _message_text_length(message):
    """Char length of a message's content, whether it's a plain string or a
    list of content blocks (the cache_control case)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


def _parse_structured_response(response):
    text = response.choices[0].message.content
    if text is None:
        raise ValueError("empty response content")
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def judge_one_symptom(client, case_file, symptom_name, model=JUDGE_MODEL):
    """One isolated call for one symptom. Tries structured output first;
    falls back to plain-JSON prompting if the proxy/model rejects
    response_format. Returns (result, meta) — meta carries elapsed_seconds
    and prompt_chars so callers can track timing/size without guessing."""
    messages = _messages_for_call(case_file, symptom_name)
    prompt_chars = sum(_message_text_length(m) for m in messages)
    start = time.perf_counter()

    result = None
    if USE_STRUCTURED_OUTPUT:
        try:
            kwargs = {"model": model, "max_tokens": 800, "timeout": 45.0,
                    "messages": messages, "response_format": SYMPTOM_JUDGMENT_SCHEMA}
            if TEMPERATURE is not None:
                kwargs["temperature"] = TEMPERATURE
            response = client.chat.completions.create(**kwargs)
            result = _parse_structured_response(response)
        except Exception:
            result = None  # fall through to plain-JSON prompting below

    if result is None:
        fallback_messages = list(messages)
        fallback_messages[-1] = {
            "role": "user",
            "content": (
                fallback_messages[-1]["content"]
                + "\n\nRespond with ONLY a JSON object, no other text, in exactly this "
                  'shape: {"reasoning": "...", "applicable": true/false, "present": '
                  'true/false, "location": "...", "evidence": "..."}'
            ),
        }
        try:
            kwargs = {"model": model, "max_tokens": 800, "timeout": 45.0,
                    "messages": fallback_messages}
            if TEMPERATURE is not None:
                kwargs["temperature"] = TEMPERATURE
            response = client.chat.completions.create(**kwargs)
            result = _parse_structured_response(response)
        except Exception as e:
            result = {"error": str(e)}

    elapsed = time.perf_counter() - start
    meta = {"elapsed_seconds": elapsed, "prompt_chars": prompt_chars}
    return result, meta


def apply_post_filter(symptom_name, result, case_file):
    """Minimal structural safety net — no text/regex heuristics. If the
    session has exactly one user message (no back-and-forth at all), force
    the four request-quality symptoms to not-present regardless of what the
    judge returned. Any session with more than one message is left entirely
    to the judge's own applicable/present judgment."""
    if not isinstance(result, dict) or "error" in result:
        return result
    if symptom_name in REQUEST_QUALITY_SYMPTOMS and cf.is_single_message_session(case_file):
        if result.get("present"):
            result = dict(result)
            result["present"] = False
            result["applicable"] = False
            result["location"] = "n/a"
            result["evidence"] = "overridden: session has only a single user message"
    return result


# ----------------------------------------------------------------------
# Standalone modes: test one session, or dump prompts for N sessions
# ----------------------------------------------------------------------

def dump_examples(n=20, out_dir="examples"):
    """Builds the prompts for the first n sessions WITHOUT calling the API,
    and writes one file per session (shared context + all 5 symptom
    prompts, each with a character count) for manual inspection of trace
    size/content."""
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
        symptom_total = 0
        for symptom_name in SYMPTOM_ORDER:
            prompt = build_symptom_prompt(symptom_name)
            symptom_total += len(prompt)
            parts.append(f"=== SYMPTOM PROMPT: {symptom_name} (user message) — {len(prompt):,} chars ===\n{prompt}\n")

        out_path = os.path.join(out_dir, f"{session_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        total_chars = len(shared) + symptom_total
        stats.append((session_id, len(shared), total_chars, download_s, parse_s))
        print(f"  wrote {out_path}  (shared: {len(shared):,} chars, "
              f"total across 5 calls: {total_chars:,} chars)")

    if stats:
        avg_shared = sum(s[1] for s in stats) / len(stats)
        avg_total = sum(s[2] for s in stats) / len(stats)
        avg_dl = sum(s[3] for s in stats) / len(stats)
        avg_parse = sum(s[4] for s in stats) / len(stats)
        print(f"\n{len(stats)} sessions dumped to '{out_dir}/'")
        print(f"avg shared-context size: {avg_shared:,.0f} chars")
        print(f"avg total prompt chars across 5 calls: {avg_total:,.0f} chars")
        print(f"avg download time: {avg_dl:.2f}s | avg parse time: {avg_parse:.2f}s")


def _test_one_session(session_id, symptom_name=None):
    print("Loading dataset tables from HuggingFace...")
    sessions, logs = cf.load_tables()
    paths = cf.path_map(logs)

    case_file, download_s, parse_s = cf.build_case_file_with_timing(session_id, paths)
    print(f"download: {download_s:.2f}s | parse: {parse_s:.2f}s | "
          f"user messages: {len(case_file['user_messages'])}\n")

    client = OpenAI(base_url=LITELLM_BASE_URL, timeout=45.0)
    symptoms = [symptom_name] if symptom_name else SYMPTOM_ORDER
    for name in symptoms:
        result, meta = judge_one_symptom(client, case_file, name)
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
            print("usage: python classify.py test <session_id> [symptom_name]")
            sys.exit(1)
        sid = sys.argv[2]
        symptom = sys.argv[3] if len(sys.argv) > 3 else None
        _test_one_session(sid, symptom)
    elif mode == "dump-examples":
        n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        out_dir_arg = sys.argv[3] if len(sys.argv) > 3 else "examples"
        dump_examples(n_arg, out_dir_arg)
    else:
        print(__doc__)