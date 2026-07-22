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
    no_verification_by_user   — positive evidence the user did NOT verify the fix
                                (not just "transcript doesn't show verification")

CHANGELOG (this revision):
    - All symptom definitions rewritten with explicit exclusions + few-shot
      examples (see FEW_SHOT_EXAMPLES) to reduce false positives, especially
      "explain this" / pure Q&A turns getting flagged as no_spec /
      no_acceptance_criteria / no_closed_loop.
    - Added a cheap heuristic pre-classifier (classify_intent) that tags each
      user message as change_request / question / ambiguous. This tag is shown
      to the judge, AND used as a deterministic post-filter: if a session has
      zero change-request turns, the four "request quality" symptoms are
      forced to False regardless of what the judge says (defense in depth).
    - Thinking capture no longer pre-filters by hedge keywords or truncates to
      150 chars / 3 snippets. ALL thinking blocks are now passed to the judge
      in full (see MAX_THINKING_CHARS_PER_BLOCK / MAX_TOTAL_THINKING_CHARS for
      the only caps that remain, which exist purely to keep a single
      pathological session from blowing up the prompt size — they are far
      above normal thinking length and are NOT keyword-based).
    - Fixed tool_result content parsing: previously did str(list-of-dicts)
      on structured content blocks, which stringified the Python repr instead
      of the actual text. Now properly extracts "text" fields from list-typed
      content before checking for pass/fail markers.
    - Test command / result capture windows widened (120->400, 300->600 chars)
      since real test output/commands were getting cut before the pass/fail
      markers appeared.
    - Transcript files are now read with explicit encoding="utf-8",
      errors="replace" so odd bytes don't silently drop or crash a line.
    - After each run, a markdown report is regenerated with a per-symptom
      "Examples" section populated from real evidence_samples collected
      during that run (not placeholder text).

Requirements: pip install datasets huggingface_hub openai
Dataset is gated: accept terms on huggingface.co and run `hf auth login` first.
Calls Claude through JetBrains' internal LiteLLM proxy (OpenAI-compatible
API). Set your LiteLLM key as OPENAI_API_KEY before running.
"""

import re
import json
from collections import Counter

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from openai import OpenAI

REPO = "SALT-NLP/SWE-chat"
JUDGE_MODEL = "anthropic/claude-haiku-4-5"
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"
REPORT_OUTPUT_PATH = "vibe_fixing_report.md"

SAMPLE = None   # None = run on every parseable session

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

# Heuristic-only, used for the intent tag shown to the judge + the
# deterministic post-filter. Not used to silently drop data — only to label it.
QUESTION_PATTERNS = (
    r"^\s*(what|why|how|when|where|which|who)\b",
    r"^\s*(explain|describe|walk me through|can you explain|could you explain)\b",
    r"^\s*(is there|are there|does|do|did|is it|can it)\b",
    r"\bwhat does\b", r"\bwhy does\b", r"\bhow does\b",
    r"^\s*(review|summarize|summarise)\b.*\?$",
)
CHANGE_REQUEST_PATTERNS = (
    r"\b(fix|add|remove|delete|change|update|refactor|implement|create|build|"
    r"replace|rewrite|optimi[sz]e|clean\s?up|make it|improve|migrate|upgrade|"
    r"convert|rename|move|split|merge|revert)\b",
)


def classify_intent(text):
    """Cheap heuristic tag: 'question', 'change_request', or 'ambiguous'.
    Deliberately conservative — only tags 'question' when the message reads
    as pure Q&A with no request-a-change language at all."""
    t = text.strip().lower()
    is_question = any(re.search(p, t) for p in QUESTION_PATTERNS) or t.endswith("?")
    is_change = any(re.search(p, t) for p in CHANGE_REQUEST_PATTERNS)
    if is_question and not is_change:
        return "question"
    if is_change:
        return "change_request"
    return "ambiguous"


SYMPTOM_DEFINITIONS = {
    "no_spec": (
        "The user's request is vague/one-line ('change the button color', "
        "'replace the navigation bar with menu', 'add parameters to endpoint', "
        "'you are wrong') with no real spec, OR the agent's own thinking shows "
        "it is unsure or stuck but it submits an answer anyway as if everything "
        "is fine. IMPORTANT EXCLUSION: this symptom is about the quality of a "
        "REQUEST FOR A CHANGE. If the user is only asking a question, asking "
        "for an explanation, a code review, or a walkthrough — with no code "
        "change requested at all — there is no 'spec' to be vague, so do NOT "
        "flag this. A precise question ('what does this regex do?') is not "
        "vague just because it's short."
    ),
    "no_closed_loop": (
        "The request ('fix tests', 'app is not running', etc.) gives the agent "
        "no way to verify whether the fix actually worked — e.g. no test suite "
        "was run and no reproduction step was checked before/after the fix. "
        "IMPORTANT EXCLUSION: only applies when a FIX was actually requested "
        "and made. If the turn is a pure question/explanation with no change "
        "made, or if the agent DID run a test/reproduction step and reported "
        "the result, do NOT flag this."
    ),
    "no_acceptance_criteria": (
        "The request has a vague success bar with no concrete criteria — "
        "'make it faster', 'clean this up', 'don't change the format/API' — "
        "so 'done' is subjective. IMPORTANT EXCLUSION: only applies to "
        "requests for a CHANGE. Do not flag pure questions ('how does the "
        "caching layer work?'), and do not flag requests that already give a "
        "concrete target ('get load time under 200ms', 'match this exact "
        "screenshot')."
    ),
    "no_visual_reference": (
        "The request is about visual/UI appearance ('make it look better', "
        "'match the design') but no actual image, mockup, or design file was "
        "ever provided anywhere in the conversation. IMPORTANT EXCLUSION: if "
        "the session has no visual/UI request at all (backend, CLI, data, "
        "infra work), this symptom simply does not apply — answer false, not "
        "true. Also do not flag if a screenshot/mockup/reference URL/design "
        "file was provided at any point in the conversation, even if not in "
        "the very first message."
    ),
    "repetitive_fix_attempts": (
        "The user reports the SAME underlying problem more than once because "
        "the agent's earlier fix didn't actually work — e.g. the user says "
        "'still broken', 'that didn't fix it', or re-describes the same bug "
        "after a fix was claimed complete. This is about the agent's fix being "
        "wrong, not about normal engineering iteration. Do NOT flag: a "
        "standard test-driven-development loop (write a test, run it, see it "
        "fail, fix the code, run it again, see it pass); a linter/build error "
        "being fixed and immediately re-checked; or a code reviewer requesting "
        "sequential/different changes that get addressed one at a time. Those "
        "are healthy iteration, not repeated wrong fixes on the SAME bug."
    ),
    "no_verification_by_user": (
        "There is POSITIVE evidence the user did not verify the fix — e.g. the "
        "user's very next message reports the SAME bug still happening after "
        "being told it was fixed, or the user explicitly says they didn't test "
        "it, or accepts a claim of success with visible skepticism/haste that "
        "contradicts having actually checked ('ok whatever, ship it', 'sure, "
        "I'll trust you'). Do NOT flag a session just because the transcript "
        "itself doesn't show a test command or a screenshot — the user may "
        "well have tested it outside the chat, and the absence of proof in "
        "the transcript is not evidence of absence of verification. Only flag "
        "when something in the conversation ACTIVELY suggests the user "
        "skipped verifying, not merely that verification isn't visible."
    ),
}

# Few-shot calibration examples shown directly in the judge prompt. Kept
# short and generic (not tied to any real session) — purpose is to anchor
# the model's threshold, especially for the pure-Q&A exclusion.
FEW_SHOT_EXAMPLES = """
CALIBRATION EXAMPLES (generic, not from this session — use only to calibrate
your judgment, do not reference these in your evidence text):

- no_spec:
    FLAG: user says only "fix the button", agent proceeds with no clarifying
    context and no spec ever emerges.
    DO NOT FLAG: user asks "explain what this regex does" — this is a
    question, not a change request, so vagueness-of-spec doesn't apply.

- no_closed_loop:
    FLAG: user says "the API times out, fix it"; agent edits code but never
    runs the endpoint or a test to confirm the timeout is gone.
    DO NOT FLAG: user asks "why does the API time out?" (no fix requested,
    nothing to close the loop on) — or the agent fixed it AND ran a
    reproduction step afterward.

- no_acceptance_criteria:
    FLAG: "make this cleaner" / "optimize this" with no definition of clean
    or fast enough.
    DO NOT FLAG: "get load time under 200ms" (concrete target given), or
    "explain how the caching layer works" (a question, not a target-setting
    request).

- no_visual_reference:
    FLAG: "make the landing page look more modern", no screenshot, mockup,
    or reference site anywhere in the conversation.
    DO NOT FLAG: user attaches a screenshot and says "match this" — or the
    session is backend/CLI work with no visual request at all.

- repetitive_fix_attempts:
    FLAG: user says "still crashes on submit" shortly after the agent claimed
    the submit crash was fixed.
    DO NOT FLAG: user writes a test, it fails, agent fixes the code, test
    passes — one clean TDD cycle is not repetitive.

- no_verification_by_user:
    FLAG: user's next message says "didn't try it but sounds right, ship it",
    or the same bug resurfaces right after being declared fixed.
    DO NOT FLAG: the transcript just ends after the fix with no test shown —
    that alone is not evidence the user skipped checking.
"""


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
    list of structured blocks like [{"type": "text", "text": "..."}]. The
    previous version did str(content) on the list case, which stringified
    the Python repr (braces, quotes, key names and all) instead of the
    actual text — this fixes that."""
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
    user_messages = []       # [(turn_no, text, has_image, intent)]
    files_touched = set()
    test_runs = []            # [{"command": str, "result": str|None, "id": tool_use_id}]
    thinking_blocks = []      # [{"turn": int, "text": str}] — ALL thinking, unfiltered
    turn_no = 0
    total_thinking_chars = 0
    thinking_truncated = False

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
                            thinking_blocks.append({"turn": turn_no, "text": snippet})
                            total_thinking_chars += len(snippet)
                        elif thinking_text:
                            thinking_truncated = True

                    if b.get("type") == "tool_use":
                        name = b.get("name", "")
                        tool_input = b.get("input", {}) or {}
                        if any(k in name.lower() for k in ("edit", "write")) and "todowrite" not in name.lower():
                            fp = tool_input.get("file_path")
                            if fp:
                                files_touched.add(fp)
                        if "bash" in name.lower():
                            command = tool_input.get("command", "") or ""
                            if any(k in command.lower() for k in TEST_KEYWORDS):
                                test_runs.append({
                                    "command": command[:MAX_TEST_COMMAND_CHARS],
                                    "result": None,
                                    "id": b.get("id"),
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
                intent = classify_intent(text)
                user_messages.append((turn_no, text[:MAX_USER_MESSAGE_CHARS], has_image, intent))

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
        "user_messages": user_messages,
        "files_touched": sorted(files_touched),
        "test_runs": test_runs,
        "thinking_blocks": thinking_blocks,
        "thinking_truncated": thinking_truncated,
        "total_turns": turn_no,
    }


def compute_scope_flags(case_file):
    """Direct metadata check — no LLM needed. Two independent flags."""
    return {
        "scope_files_too_many": len(case_file["files_touched"]) >= SCOPE_FILES_TOO_MANY_THRESHOLD,
        "scope_turns_too_long": case_file["total_turns"] >= SCOPE_TURNS_TOO_LONG_THRESHOLD,
    }


def session_has_change_request(case_file):
    """True if at least one user message reads as an actual request for a
    code change (not pure Q&A). Used as a deterministic post-filter safety
    net for the four 'request quality' symptoms."""
    return any(intent == "change_request" for _, _, _, intent in case_file["user_messages"])


# ----------------------------------------------------------------------
# LLM-as-judge
# ----------------------------------------------------------------------

REQUEST_QUALITY_SYMPTOMS = ("no_spec", "no_closed_loop", "no_acceptance_criteria", "no_visual_reference")


def build_symptom_judge_prompt(case_file):
    msgs = "\n".join(
        f"{i+1}. [{intent} | {'image attached' if img else 'text only'}] {text}"
        for i, (_, text, img, intent) in enumerate(case_file["user_messages"])
    ) or "(no user messages)"

    tests = "\n".join(
        f"- `{t['command']}` -> {t['result'] or 'unclear'}" for t in case_file["test_runs"]
    ) or "(none run)"

    if case_file["thinking_blocks"]:
        thinking = "\n\n".join(
            f"[turn {tb['turn']}] {tb['text']}" for tb in case_file["thinking_blocks"]
        )
        if case_file["thinking_truncated"]:
            thinking += "\n\n(additional thinking beyond the size cap omitted)"
    else:
        thinking = "(no thinking blocks captured)"

    defs = "\n".join(f"- {name}: {desc}" for name, desc in SYMPTOM_DEFINITIONS.items())

    return (
        "You are reviewing one coding-agent session for signs of 'vibe-fixing' "
        "(accepting fixes without proper spec or verification). Here is the "
        "session, condensed:\n\n"
        f"USER MESSAGES (in order, each tagged with a heuristic intent label "
        f"'question' / 'change_request' / 'ambiguous' — treat this label as a "
        f"hint, not ground truth, and re-read the message yourself):\n{msgs}\n\n"
        f"FILES TOUCHED: {len(case_file['files_touched'])} files\n\n"
        f"TEST/BUILD COMMANDS RUN:\n{tests}\n\n"
        f"FULL ASSISTANT THINKING (every thinking block in this session, in "
        f"order, not just ones that look hedgy — read all of it and decide "
        f"for yourself whether it shows doubt, confusion, or a hedge-then-"
        f"submit pattern):\n{thinking}\n\n"
        f"SYMPTOM DEFINITIONS:\n{defs}\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "For each symptom, decide if it is present in THIS session. Answer with "
        "ONLY a JSON object, no other text, in exactly this shape:\n"
        "{\n"
        '  "session_is_pure_qa": true or false,  // true only if EVERY user message in this session is a question/explanation request with no code change ever requested\n'
        '  "no_spec": {"present": true or false, "evidence": "<one short sentence>"},\n'
        '  "no_closed_loop": {"present": true or false, "evidence": "..."},\n'
        '  "no_acceptance_criteria": {"present": true or false, "evidence": "..."},\n'
        '  "no_visual_reference": {"present": true or false, "evidence": "..."},\n'
        '  "repetitive_fix_attempts": {"present": true or false, "evidence": "..."},\n'
        '  "no_verification_by_user": {"present": true or false, "evidence": "..."}\n'
        "}"
    )


def judge_symptoms(client, case_file, model=JUDGE_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1200,   # bumped: prompt now includes full thinking + few-shot, more room for evidence text
            temperature=0,
            timeout=45.0,      # bumped alongside larger prompt so we don't time out mid-response
            messages=[{"role": "user", "content": build_symptom_judge_prompt(case_file)}],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


def apply_post_filters(result, case_file):
    """Deterministic safety net on top of the judge's own answer: if the
    session has no actual change-request turn (or the judge itself says the
    whole session is pure Q&A), force the four request-quality symptoms to
    False regardless of what the judge returned. This directly targets the
    'explain me' false-positive complaint — belt and suspenders alongside the
    prompt-level exclusion, since judges do occasionally ignore instructions."""
    if not isinstance(result, dict) or "error" in result:
        return result
    pure_qa = bool(result.get("session_is_pure_qa")) or not session_has_change_request(case_file)
    if pure_qa:
        for name in REQUEST_QUALITY_SYMPTOMS:
            entry = result.get(name)
            if isinstance(entry, dict) and entry.get("present"):
                result[name] = {
                    "present": False,
                    "evidence": "overridden: session has no actual change-request turn (pure Q&A)",
                }
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
    n_ok = n_skipped = n_judged = 0
    evidence_samples = {name: [] for name in SYMPTOM_DEFINITIONS}
    debug_steps = 3

    for i, session_id in enumerate(ids, 1):
        show_debug = i <= debug_steps
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] processing {session_id} "
                  f"(ok={n_ok} judged={n_judged} skipped={n_skipped})", flush=True)
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

            if show_debug:
                print(f"    -> calling Haiku judge...", flush=True)
            result = judge_symptoms(client, case_file)
            if show_debug:
                print(f"    -> judge call finished", flush=True)
            if not result or "error" in result:
                n_skipped += 1
                if show_debug and result:
                    print(f"    -> judge error: {result.get('error')}", flush=True)
                continue

            result = apply_post_filters(result, case_file)
            n_judged += 1

            for name in SYMPTOM_DEFINITIONS:
                entry = result.get(name, {})
                if isinstance(entry, dict) and entry.get("present"):
                    counts[name] += 1
                    if len(evidence_samples[name]) < 5:
                        evidence_samples[name].append((session_id, entry.get("evidence", "")))

        except Exception as e:
            n_skipped += 1
            if show_debug:
                print(f"    -> exception: {e}", flush=True)

    print(f"\nsessions with usable data: {n_ok} | skipped: {n_skipped} | judged: {n_judged}")
    print(f"\n{'symptom':28s} {'count':>7s} {'% of relevant sessions':>24s}")
    scope_names = ["scope_files_too_many", "scope_turns_too_long"]
    for name in list(SYMPTOM_DEFINITIONS) + scope_names:
        denom = (n_judged or 1) if name not in scope_names else (n_ok or 1)
        print(f"{name:28s} {counts[name]:>7,} {100*counts[name]/denom:>23.0f}%")

    if verbose:
        print("\nsample evidence (spot-check these against the real transcripts):")
        for name, samples in evidence_samples.items():
            if samples:
                print(f"\n  {name}:")
                for sid, ev in samples:
                    print(f"    [{sid}] {ev}")

    return counts, n_judged, n_ok, evidence_samples


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
    "no_verification_by_user": "There is a real sign the user did not check the fix (not just \"no proof shown\")",
}


def write_markdown_report(counts, n_judged, n_ok, evidence_samples, path=REPORT_OUTPUT_PATH):
    ordered = ["no_closed_loop", "no_verification_by_user", "no_spec", "no_acceptance_criteria",
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
    lines.append("I checked each session for 7 symptoms:\n")
    lines.append("| Symptom | What it means |")
    lines.append("|---|---|")
    for name in ["no_spec", "no_closed_loop", "no_acceptance_criteria", "no_visual_reference",
                 "repetitive_fix_attempts", "scope_files_too_many", "scope_turns_too_long",
                 "no_verification_by_user"]:
        lines.append(f"| `{name}` | {SYMPTOM_DESCRIPTIONS_SHORT[name]} |")
    lines.append("")

    lines.append("## How I Detected Them\n")
    lines.append(
        "I used two methods:\n\n"
        "**1. LLM-as-judge (Claude Haiku 4.5).** For 6 of the symptoms, I cannot use simple rules "
        "— I need to read the conversation. So each session was sent to Haiku, one API call per "
        "session. Compared to the first version of this report, the case file now includes:\n"
        "- all user messages, in order, each tagged with a heuristic intent label "
        "(question / change request / ambiguous), so the judge can tell a request for a fix apart "
        "from a plain question\n"
        "- the number of files touched\n"
        "- any test/build commands run, and whether they passed or failed, with a wider capture "
        "window and correct parsing of structured tool results (an earlier version was "
        "accidentally reading the Python object representation instead of the actual text in some "
        "cases)\n"
        "- **every thinking block from the agent, in full** — not just short excerpts that already "
        "contained a hedge word. Pre-filtering by keyword before the judge ever saw the text meant "
        "we were partly deciding the answer ourselves; the judge now reads all of the agent's "
        "reasoning and decides for itself whether it shows doubt\n\n"
        "Haiku received this case file plus a clear definition of each symptom, a short set of "
        "calibration examples (flag / do-not-flag pairs) for each symptom, and returned a yes/no "
        "answer with a short reason for each one. To directly address the \"explain me\" false-"
        "positive problem: the four symptoms that describe the quality of a *request* "
        "(`no_spec`, `no_closed_loop`, `no_acceptance_criteria`, `no_visual_reference`) are now "
        "explicitly defined to not apply to sessions that are pure questions/explanations with no "
        "code change requested, and this is enforced twice — once in the prompt itself, and again "
        "as a deterministic override after the judge answers, in case the judge doesn't follow the "
        "instruction.\n\n"
        "**2. Metadata-only rules.** The two `scope_*` symptoms don't need an LLM. I just count "
        "files touched and turns per session, and flag sessions above a threshold.\n"
    )

    lines.append(f"## Results ({n_judged:,} Sessions Judged)\n")
    lines.append("| Symptom | Count | % of sessions |")
    lines.append("|---|---|---|")
    for name in ordered:
        denom = n_ok if name in scope_names else n_judged
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
                 "repetitive_fix_attempts", "no_verification_by_user"]:
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
        "`no_verification_by_user` should still be treated carefully even after tightening its "
        "prompt and adding calibration examples. A person could always test something outside the "
        "chat window, so this number likely remains an upper bound on true skipped-verification "
        "rather than an exact count. The other symptoms rely on clearer, easier-to-check evidence "
        "(a request's wording plus its intent tag, whether a test command was run and what it "
        "returned, file counts) and the pure-Q&A override directly removes a known false-positive "
        "source, so they should be more trustworthy as reported — but spot-checking the Examples "
        "section above against real transcripts is still recommended before citing these numbers "
        "externally.\n"
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

    counts, n_judged, n_ok, evidence_samples = run(all_ids, paths)
    write_markdown_report(counts, n_judged, n_ok, evidence_samples)