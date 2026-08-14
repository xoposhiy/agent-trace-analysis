"""
classify.py — symptom/category definitions, prompt construction, and the
LLM-judge calls. Imports case_file.py for the timeline/rendering.

Runnable standalone:

    python classify.py test <session_id> [call_name]
    python classify.py dump-examples [n] [out_dir]

Key decisions:
    - Three LLM calls per session: two grouped categories plus one single
      symptom.
        * not_enough_verification — not-tested, self-report,
          ask-for-manual-testing, repetitive-bug-fixes
        * not_enough_specification — no-spec-detected,
          repetitive-requirements-fixes, self-report
        * no_visual_reference — no subcategories
      scope_files_too_many / scope_turns_too_long stay metadata-only checks
      in case_file.py.
    - Each call returns a "findings" LIST, not a present/absent boolean: a
      session can show the same subcategory more than once, and every
      occurrence is captured.
    - A finding carries a `cause_prompt` AND a list of `evidence` blocks,
      because one location field had two incompatible consumers. The
      statistics need attribution (one problem, one place, or a rate is not a
      rate); an auditor needs verifiability (every place it can be seen). For
      the repetitive-* subcategories those are provably different blocks: the
      cause is the earlier prompt called finished, the only observable symptom
      is the user's later complaint.
    - Evidence entries are OBJECTS ({prompt, step, note}) — a coordinate with
      no text cannot be checked, and being checkable is their whole point.
    - Every finding also carries a `confidence` in 0..1: how sure the judge is
      that the finding MATCHES THE DEFINITION it is filed under. It is asked
      for after the evidence, so the number is given once the judge has already
      committed to what it can point at. Nothing is filtered by it here — it is
      recorded so a later analysis can cut the same run at any threshold.
    - Coordinates are TWO integers, matching the `P.S` header case_file.py
      prints. Never a string: "4.2" in a text field is the failure mode the
      split exists to prevent, and the validator rejects it rather than guess.
    - "reasoning" is the first schema property, before "findings", so the
      model has room to think before committing.
    - Structured output via response_format, with an automatic fallback to
      plain-JSON prompting. The fallback strips cache_control, so a proxy that
      chokes on that field can't take both attempts down.
    - Every failure and fallback lands in DIAGNOSTICS (first of each kind is
      printed), so a run can say WHY calls failed instead of degrading
      silently.
    - Coordinates are RECORDED, NEVER REPAIRED. A miscited coordinate is
      measured behaviour of the judge, and this pipeline's output is
      measurements of that judge — correcting it destroys the only trace it
      happened. Nothing is renumbered and no finding is dropped for pointing
      somewhere impossible; instead each cause gets a `cause_kind` marker
      computed here from the timeline: "real" / "system" / "out_of_range".
    - The one exception is an unknown `subcategory`, which is dropped: it
      invents a result key no consumer knows about, corrupting the whole
      aggregation rather than just its own row.
    - Only cause_kind == "real" feeds the numerator of "% of prompts" — the
      rest have no real prompt to be a fraction of. Excluded from the
      numerator is not excluded from the results: they still count in
      `problems`, appear in Examples marked, and are reported as an
      attribution-miss rate.
    - TEMPERATURE can be None — some models reject a custom temperature.
"""

import json
import time
from collections import Counter

from openai import OpenAI

import case_file as cf

JUDGE_MODEL = "openai/gpt-5.6-luna"
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"

# OPENAI_API_KEY (the LiteLLM key) is read from the environment; importing
# case_file above already loaded the repo-root .env.

# Some models reject a custom temperature and only support their default.
TEMPERATURE = None

USE_STRUCTURED_OUTPUT = True
ATTEMPT_PROMPT_CACHING = True   # best-effort, unverified whether the proxy honors it

# Reasoning models spend part of this budget on hidden reasoning tokens before
# emitting any JSON, so a tight cap truncates the answer and surfaces as an
# unparseable-response failure. Be generous; the response itself is small.
MAX_COMPLETION_TOKENS = 6000

# "<stage>: <detail>" -> how many times it happened. run_experiment.py dumps
# this at the end of a run and stores it in results.json.
DIAGNOSTICS = Counter()
_DIAGNOSTICS_REPORTED = set()


def _note(stage, detail, verbose=True):
    """Record a failure/fallback reason. Prints only the first occurrence of
    each kind, so a systematic problem is visible immediately without one
    line per session."""
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

# Every LLM call we make. "kind" distinguishes a grouped category from a
# single symptom; both return the same findings-list shape.
CALLS = {
    "not_enough_verification": {"kind": "category", **CATEGORY_DEFINITIONS["not_enough_verification"]},
    "not_enough_specification": {"kind": "category", **CATEGORY_DEFINITIONS["not_enough_specification"]},
    "no_visual_reference": {"kind": "single", **SINGLE_SYMPTOM_DEFINITIONS["no_visual_reference"]},
}
CALL_ORDER = list(CALLS)


def all_result_keys():
    """Every (call, subcategory) key this file can produce, in a stable
    order, so run_experiment.py and report.py agree on the full set even for
    keys with zero hits."""
    keys = []
    for call_name, info in CALLS.items():
        if info["kind"] == "category":
            keys.extend(f"{call_name}:{subcat}" for subcat in info["subcategories"])
        else:
            keys.append(call_name)
    return keys


def valid_subcategories(call_name):
    """The only subcategory values a finding for this call may carry. For a
    single symptom that's the call name itself, keeping the finding shape
    identical across both kinds."""
    info = CALLS[call_name]
    if info["kind"] == "category":
        return set(info["subcategories"])
    return {call_name}


# ----------------------------------------------------------------------
# Structured-output schemas
# ----------------------------------------------------------------------

# The judge reads a joined coordinate ("4.2") off the timeline and is asked
# for two integers, so these descriptions spell out the split and show the
# parse. Without the worked example the compact form invites returning the
# string "4.2" — the failure mode the two-integer shape exists to prevent.
_CAUSE_PROMPT_DESC = (
    "The number P of the user prompt whose work led to this problem — a single "
    "integer, e.g. 4, read off a '[USER P.1]' header. Prompt number only: no step, "
    "no range, no list, no text."
)
_EVIDENCE_ARRAY_DESC = (
    "Every block where this problem can actually be observed, at least one. Each "
    "entry is one block of the timeline. These are the places a human will open to "
    "check you, so an entry must point at a block that visibly shows something."
)
_EVIDENCE_PROMPT_DESC = (
    "The prompt number P of the cited block — the FIRST of the two numbers in its "
    "header. From a header '[thinking 4.2]': prompt = 4. An integer, never a string, "
    "never '4.2'."
)
_EVIDENCE_STEP_DESC = (
    "The step number S of the cited block — the SECOND of the two numbers in its "
    "header. From a header '[thinking 4.2]': step = 2. An integer, never a string, "
    "never '4.2'."
)
_EVIDENCE_NOTE_DESC = (
    "One short sentence, in English, saying what is visible at that specific block."
)
_CONFIDENCE_DESC = (
    "How sure you are that this finding matches the definition it is filed under — "
    "a number from 0 to 1. Rate the DEFINITION MATCH only: not how serious the "
    "problem is, not how bad the session looks overall. Use above 0.9 when the "
    "timeline shows it beyond argument, around 0.5 when the case is genuinely "
    "borderline, and below 0.3 when you are mostly guessing. Still report a finding "
    "you are unsure about — a low number here is how you say that; leaving it out is "
    "not."
)
_FINDINGS_ARRAY_DESC = (
    "One entry per distinct problem. Empty array if none occurred. The same "
    "subcategory can appear more than once if it happened more than once; do not "
    "merge separate problems and do not split one problem across entries."
)


def _evidence_schema():
    """The evidence list, identical for both call kinds."""
    return {
        "type": "array",
        "description": _EVIDENCE_ARRAY_DESC,
        "items": {
            "type": "object",
            "properties": {
                "prompt": {"type": "integer", "description": _EVIDENCE_PROMPT_DESC},
                "step": {"type": "integer", "description": _EVIDENCE_STEP_DESC},
                "note": {"type": "string", "description": _EVIDENCE_NOTE_DESC},
            },
            "required": ["prompt", "step", "note"],
            "additionalProperties": False,
        },
    }


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
                        "description": _FINDINGS_ARRAY_DESC,
                        "items": {
                            "type": "object",
                            "properties": {
                                "subcategory": {"type": "string", "enum": list(subcategory_keys)},
                                "cause_prompt": {"type": "integer", "description": _CAUSE_PROMPT_DESC},
                                "evidence": _evidence_schema(),
                                # Last on purpose: the model fills the fields in
                                # order, so it rates its own certainty after it
                                # has written down what it can point at.
                                "confidence": {"type": "number", "description": _CONFIDENCE_DESC},
                            },
                            "required": ["subcategory", "cause_prompt", "evidence", "confidence"],
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
                        "description": _FINDINGS_ARRAY_DESC,
                        "items": {
                            "type": "object",
                            "properties": {
                                "cause_prompt": {"type": "integer", "description": _CAUSE_PROMPT_DESC},
                                "evidence": _evidence_schema(),
                                # See the note in _build_category_schema.
                                "confidence": {"type": "number", "description": _CONFIDENCE_DESC},
                            },
                            "required": ["cause_prompt", "evidence", "confidence"],
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
    """The large part of the prompt, IDENTICAL across all 3 calls for a
    session. Sent as the system message so it is the shared prefix a cache
    breakpoint can apply to (see ATTEMPT_PROMPT_CACHING).

    The numbering block explains what the P.S labels mean; the schema field
    descriptions explain how to split one into two integers. Both are needed
    — nothing connects the labels here to the answer fields there.

    Two counts go in, not one: numbering can run to 40 in a session with 24
    real prompts, and a judge told only "24" would treat every citation above
    it as impossible."""
    max_number = case_file["max_prompt_number"]
    n_real = len(case_file["user_messages"])
    return (
        "You are reviewing one coding-agent session for signs of 'vibe-fixing' "
        "(accepting fixes without proper specification or verification).\n\n"
        "CHRONOLOGICAL SESSION TIMELINE (every user message, every reply the "
        "agent wrote back to the user, every piece of agent thinking, and "
        "every tool call the agent made — including its raw result — in the "
        f"exact order they occurred):\n\n{cf.render_timeline(case_file)}\n\n"
        "HOW THE TIMELINE IS NUMBERED. Every block's header carries a two-part "
        "coordinate `P.S` — for example `[thinking 4.2]` means prompt 4, step 2.\n"
        "`P` is the user-prompt number: a `[USER P.1]` line opens prompt P, and "
        "every block after it carries the same P until the next `[USER ...]` or "
        "`[SYSTEM ...]` line.\n"
        "`S` is the block's position inside that prompt, starting at 1.\n"
        "`[SYSTEM P.1]` lines are NOT user prompts. They are wrappers emitted by "
        "the tooling — slash commands, context-compaction summaries, interruptions "
        "— and contain no request from the human, but they still consume a number, "
        "so the numbering has gaps between real prompts. You may cite them as "
        "evidence; never treat them as a request.\n"
        "Read both numbers off the label — never count or compute them yourself.\n"
        f"This session's numbering runs up to {max_number}, of which {n_real} are "
        "real user prompts.\n\n"
        "Note the difference between block types: '[reply-to-user P.S]' is what the "
        "agent actually told the human, while '[thinking P.S]' is private reasoning "
        "the human never saw. A claim the agent only made in its own thinking was "
        "never said to the user.\n\n"
        f"FILES TOUCHED (edited/written): {len(case_file['files_touched'])} files\n"
    )


# Appended to every call's instructions. Each paragraph fixes a specific
# failure mode:
#   - cause vs. evidence: for the repetitive-* subcategories the cause prompt
#     is one where the symptom is by definition not yet observable.
#   - "usually the same": a model asked to fill two location fields will hunt
#     for two DIFFERENT places and find something. For four of the seven
#     subcategories they coincide, so that has to be said out loud.
#   - not-tested: its evidence is an absence, which has no coordinate —
#     redirected to the block where completion is claimed, which has one.
#   - no collapsing: merging findings that share a cause is an exact set
#     operation done in run_experiment.py, not a job for the model.
#   - the language rule: a Japanese session produced Japanese evidence in an
#     English report.
LOCATION_RULE = (
    "FOR EACH FINDING you must give TWO different things, and they answer two "
    "different questions.\n\n"
    "'cause_prompt' — WHOSE WORK caused this. The number of the user prompt the "
    "agent was WORKING ON when the problem was created, not the prompt where the "
    "consequence surfaced. For repetitive-bug-fixes and repetitive-requirements-"
    "fixes that means the earlier prompt whose work was treated as finished and "
    "later turned out to be wrong, NOT the prompt in which the user complained "
    "about it. A single integer, and never the number of a '[SYSTEM ...]' line: "
    "those are not prompts and nobody's work belongs to them.\n\n"
    "'evidence' — WHERE IT IS VISIBLE. One entry per block that actually shows the "
    "problem, each with its own prompt and step numbers plus a one-sentence note. "
    "For repetitive-bug-fixes and repetitive-requirements-fixes this is precisely "
    "the user's later complaints — the blocks cause_prompt deliberately does not "
    "point at. Citing a '[SYSTEM ...]' block as evidence is allowed and sometimes "
    "correct: '[Request interrupted by user]' is evidence of something.\n\n"
    "For self-report, ask-for-manual-testing, no-spec-detected and "
    "no_visual_reference the cause and the evidence are normally the SAME prompt. "
    "That is expected and correct — do not hunt for a different block just to make "
    "the two fields differ.\n\n"
    "For not-tested, the evidence is the block where the work is DECLARED finished "
    "(usually a '[reply-to-user P.S]' saying it is done). You cannot cite the "
    "absence of a test — it has no location — so cite the claim that was made "
    "without one.\n\n"
    "Return one finding per distinct problem. Do not merge two problems that happen "
    "to share a cause_prompt into one finding, and do not split one problem into "
    "several. Grouping is done afterwards, outside your answer.\n\n"
    "Write 'reasoning' and every evidence 'note' in ENGLISH, even when the session "
    "itself is in another language."
)

# Kept separate from LOCATION_RULE because it fixes a different failure mode: a
# judge with no way to express doubt either drops the borderline cases or
# reports them as firmly as the obvious ones, and the aggregate cannot tell the
# two apart afterwards.
CONFIDENCE_RULE = (
    "FOR EACH FINDING also give a 'confidence' from 0 to 1: how sure you are that "
    "the finding MATCHES THE DEFINITION you are filing it under.\n\n"
    "This is about the definition match and nothing else. Do not lower it because "
    "the problem looks minor, and do not raise it because the session looks careless "
    "overall. Above 0.9 means the timeline shows it beyond argument; about 0.5 means "
    "the case is genuinely borderline; below 0.3 means you are mostly guessing.\n\n"
    "Report the borderline cases too, with a low number, instead of leaving them out. "
    "A finding at 0.4 and a missing finding are not the same answer, and only the "
    "first one can be reviewed."
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
            "with its subcategory, its cause_prompt, its evidence blocks, and its "
            "confidence. "
            f"Return an empty findings list if none occurred.\n\n{LOCATION_RULE}"
            f"\n\n{CONFIDENCE_RULE}"
        )
    return (
        f"SYMPTOM TO EVALUATE: {call_name}\n\n"
        f"DEFINITION: {info['definition']}\n\n"
        f"CALIBRATION EXAMPLES (generic, not from this session):\n{info['few_shot']}\n\n"
        "Using the session timeline given above, find EVERY occurrence of this "
        "symptom — not just the first. Think through the evidence in your "
        "reasoning field first, then list each occurrence in 'findings' with its "
        "cause_prompt, its evidence blocks, and its confidence. Return an empty "
        f"findings list if it never occurred.\n\n{LOCATION_RULE}"
        f"\n\n{CONFIDENCE_RULE}"
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
    """The same messages with structured content blocks flattened to plain
    strings. The fallback must NOT resend cache_control: if that field is what
    the proxy rejected, the fallback would fail for the same reason."""
    plain = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        plain.append({"role": message["role"], "content": content})
    return plain


def _coerce_int(raw):
    """Turn whatever came back into a plain int, or None if it isn't one.

    The fallback path has no schema, so an integer can arrive as 4, "4" or
    "prompt 4" — the same answer typed differently, all accepted.

    "4.2" is NOT accepted: stripping non-digits would yield 42, a number the
    session may well contain, pointing somewhere unrelated, with nothing left
    to say it was invented. So exactly one run of digits is a number; two or
    more ("4.2", "4-6", "3, 7") is a composite and is rejected, not guessed."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    runs = []
    current = ""
    for ch in str(raw):
        if ch.isdigit():
            current += ch
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    if len(runs) != 1:
        return None
    return int(runs[0])


def _coerce_confidence(raw):
    """Read the judge's confidence as a float in 0..1, or None if it is not one.

    Notation is normalised ("0.8" and 0.8 are the same answer), the VALUE is
    not: 80 does not become 0.8 and 1.4 is not clamped to 1.0, because either
    would turn a judge that ignored the scale into one that followed it. An
    unreadable answer becomes None and the raw text is kept alongside it, so a
    missing number stays visible instead of looking like low confidence."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def _keep_verbatim(raw):
    """What we store when no number could be read: the answer itself,
    unchanged. Only guards hashability and JSON-serialisability — the cause is
    used as part of a set key in run_experiment.py and ends up in results.json
    — so a dict or list becomes its repr rather than something countable."""
    if isinstance(raw, (str, int, float, bool)) or raw is None:
        return raw
    return repr(raw)


def _classify_prompt_number(value, prompts, max_number):
    """The `cause_kind` marker, computed from the timeline. Only "real" is a
    number that exists AND belongs to something a human asked for, so only
    "real" can enter the numerator of a rate over real prompts.

    Also used on evidence prompts, where "system" is not a miss — citing
    '[Request interrupted by user]' is legitimate — and only "out_of_range"
    signals a problem."""
    if not isinstance(value, int) or isinstance(value, bool):
        return "out_of_range"
    if not 1 <= value <= max_number:
        return "out_of_range"
    return "system" if prompts[value]["kind"] == "system" else "real"


def _validate_evidence(call_name, raw_evidence, prompts, max_number, verbose):
    """Normalise one finding's evidence list WITHOUT repairing it. Returns
    [{prompt, step, note, prompt_kind}, ...], possibly empty.

    Nothing is dropped, clamped or zeroed: an out-of-range prompt is kept and
    marked, a step past the end of its prompt is kept as written. Repairing
    either leaves nothing in the output to distinguish a judge that cited
    precisely from one that guessed.

    An empty list stays empty, never backfilled from the cause: a finding the
    judge could not point at is a different observation from one it pointed at
    its own cause."""
    if not isinstance(raw_evidence, list):
        if raw_evidence:
            _note("validate",
                  f"{call_name}: evidence was {type(raw_evidence).__name__}, not a list",
                  verbose)
        raw_evidence = []

    clean = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            _note("validate",
                  f"{call_name}: evidence entry was {type(item).__name__}, not an object",
                  verbose)
            continue
        note = str(item.get("note") or "").strip()

        prompt = _coerce_int(item.get("prompt"))
        if prompt is None:
            prompt = _keep_verbatim(item.get("prompt"))
        prompt_kind = _classify_prompt_number(prompt, prompts, max_number)
        if prompt_kind == "out_of_range":
            _note("validate",
                  f"{call_name}: evidence prompt {prompt!r} is not in 1..{max_number} "
                  "(kept, marked)", verbose)

        step = _coerce_int(item.get("step"))
        if step is None:
            step = _keep_verbatim(item.get("step"))
        elif prompt_kind != "out_of_range":
            max_step = prompts[prompt]["steps"]
            if not 1 <= step <= max_step:
                _note("validate",
                      f"{call_name}: evidence step {step!r} is not in 1..{max_step} "
                      f"for prompt {prompt} (kept as-is)", verbose)

        clean.append({
            "prompt": prompt, "step": step, "note": note,
            "prompt_kind": prompt_kind,
        })
    return clean


def _validate_findings(call_name, result, case_file, verbose=True):
    """Normalise what the model returned and CLASSIFY its coordinates. The
    fallback path has no schema enforcement, so an unknown subcategory or a
    malformed coordinate can arrive.

    Exactly one thing is rejected: an unknown subcategory, which invents a
    result key no consumer knows about and corrupts the whole aggregation.
    Everything else — a cause on a harness wrapper, a cause outside the
    session, a step past the end of its prompt — is kept verbatim and
    labelled. A pipeline that quietly fixes those reports a judge more
    accurate than the one it ran.

    Classification range is 1..max_prompt_number, over BOTH kinds of block,
    not the count of real prompts: numbering runs straight through the
    wrappers, so in a session with 24 real prompts a valid citation can read
    37."""
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
    prompts = case_file["prompts"]
    max_number = case_file["max_prompt_number"]

    clean = []
    for finding in findings:
        if not isinstance(finding, dict):
            _note("validate", f"{call_name}: finding was {type(finding).__name__}, not an object", verbose)
            continue
        # Single symptoms carry no subcategory — inject the call name so
        # downstream code can treat every finding uniformly.
        subcategory = call_name if is_single else finding.get("subcategory")
        if subcategory not in allowed:
            _note("validate", f"{call_name}: unknown subcategory {subcategory!r}", verbose)
            continue

        # Normalising notation ("4", "prompt 4") is not correcting an answer.
        # If no number can be read, store it exactly as it arrived — guessing
        # at "4.2" would invent a citation.
        raw_cause = finding.get("cause_prompt")
        cause = _coerce_int(raw_cause)
        if cause is None:
            cause = _keep_verbatim(raw_cause)
        cause_kind = _classify_prompt_number(cause, prompts, max_number)
        if cause_kind == "system":
            _note("validate",
                  f"{call_name}: cause_prompt {cause} is a [SYSTEM] block, not a user "
                  "prompt (kept, marked system)", verbose)
        elif cause_kind == "out_of_range":
            _note("validate",
                  f"{call_name}: cause_prompt {cause!r} is not a prompt in "
                  f"1..{max_number} (kept, marked out_of_range)", verbose)

        raw_confidence = finding.get("confidence")
        confidence = _coerce_confidence(raw_confidence)
        if confidence is None:
            _note("validate",
                  f"{call_name}: confidence {raw_confidence!r} is not a number in "
                  "0..1 (kept as null)", verbose)

        entry = {
            "subcategory": subcategory,
            "cause_prompt": cause,
            # Ours, not the model's: the judge is never asked whether its own
            # citation landed.
            "cause_kind": cause_kind,
            "evidence": _validate_evidence(
                call_name, finding.get("evidence"), prompts, max_number, verbose),
            # The judge's own certainty that this matches the definition. Never
            # used to drop a finding here; recorded for later analysis.
            "confidence": confidence,
        }
        # Only when the number could not be read, so the common case stays
        # small and the odd case stays auditable.
        if confidence is None and raw_confidence is not None:
            entry["confidence_raw"] = _keep_verbatim(raw_confidence)
        clean.append(entry)

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
    result = {"reasoning": str, "findings": [{"subcategory", "cause_prompt",
    "cause_kind", "evidence": [{"prompt", "step", "note", "prompt_kind"}, ...],
    "confidence"}, ...]}. "subcategory" is always present and always a declared
    value for both kinds; "cause_kind" and "prompt_kind" are computed here, not
    returned by the model; "evidence" may be empty; "confidence" is the model's
    own 0..1 rating, or None when it returned something outside that scale (the
    original then also appears as "confidence_raw"). On failure, result is
    {"error": "..."} and meta is still populated."""
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
        # Coordinates shown as bare integers, never "4" or "4.2". With no
        # schema behind this path, this string is the only thing defining the
        # shape, and a quoted example invites a string back.
        shape_hint = (
            '{"reasoning": "...", "findings": [{"subcategory": "...", "cause_prompt": 4, '
            '"evidence": [{"prompt": 9, "step": 3, "note": "..."}], "confidence": 0.8}]}'
            if info["kind"] == "category" else
            '{"reasoning": "...", "findings": [{"cause_prompt": 4, '
            '"evidence": [{"prompt": 9, "step": 3, "note": "..."}], "confidence": 0.8}]}'
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

    result = _validate_findings(call_name, result, case_file, verbose)

    elapsed = time.perf_counter() - start
    meta = {"elapsed_seconds": elapsed, "prompt_chars": prompt_chars}
    return result, meta


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
          f"real prompts: {len(case_file['user_messages'])} | "
          f"system events: {case_file['system_events']} | "
          f"numbering up to: {case_file['max_prompt_number']}\n")

    client = OpenAI(base_url=LITELLM_BASE_URL, timeout=60.0)
    calls = [call_name] if call_name else CALL_ORDER
    for name in calls:
        result, meta = judge_one_call(client, case_file, name)
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