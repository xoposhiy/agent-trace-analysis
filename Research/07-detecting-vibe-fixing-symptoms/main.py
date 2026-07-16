"""
main.py — frequency of vibe-fixing symptoms across the SWE-chat dataset
(ALL sessions, not just long ones), using Haiku as an LLM-judge.

Console output only — no files are written. Runs on every parseable
session by default (SAMPLE = None below).

Symptoms detected:
    no_spec                  — vague one-line request, or agent submits
                                despite its own hedging/uncertainty
    no_closed_loop            — request gives no way to verify the fix worked
    no_acceptance_criteria    — vague success bar, no concrete criteria
    no_visual_reference       — "make it look better" with no image/mockup
    repetitive_fix_attempts   — same USER-REPORTED bug fixed wrong repeatedly
                                (excludes normal TDD red-green iteration)
    scope_files_too_many      — too many files changed in one session (metadata-only)
    scope_turns_too_long      — session ran for an unusually large number of turns (metadata-only)
    no_verification_by_user   — positive evidence the user did NOT verify the fix
                                (not just "transcript doesn't show verification")

Requirements: pip install datasets huggingface_hub openai
Dataset is gated: accept terms on huggingface.co and run `hf auth login` first.
Calls Claude through JetBrains' internal LiteLLM proxy (OpenAI-compatible
API). Set your LiteLLM key as OPENAI_API_KEY before running.
"""

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import Counter
import json
from openai import OpenAI

REPO = "SALT-NLP/SWE-chat"
JUDGE_MODEL = "anthropic/claude-haiku-4-5"
LITELLM_BASE_URL = "https://litellm.labs.jb.gg"

SAMPLE = None   # None = run on every parseable session

SCOPE_FILES_TOO_MANY_THRESHOLD = 8
SCOPE_TURNS_TOO_LONG_THRESHOLD = 150

TEST_KEYWORDS = ("test", "pytest", "jest", "npm run", "go test", "cargo test", "rspec")
FAIL_MARKERS = ("fail", "error", "traceback", "exception", "not ok")
PASS_MARKERS = ("pass", "ok", "success", "0 failing", "all tests passed")
HEDGE_PHRASES = (
    "not sure", "unclear", "i think", "might not", "not certain",
    "let me guess", "hopefully", "should work", "not 100% sure", "unsure",
)

SYMPTOM_DEFINITIONS = {
    "no_spec": (
        "The user's request is vague/one-line (e.g. 'change the button color', "
        "'replace the navigation bar with menu', 'add parameters to endpoint', "
        "'you are wrong') with no real spec, OR the agent's own thinking shows "
        "it is unsure or stuck but it submits an answer anyway as if everything "
        "is fine."
    ),
    "no_closed_loop": (
        "The request ('fix tests', 'app is not running', etc.) gives the agent "
        "no way to verify whether the fix actually worked — e.g. no test suite "
        "was run and no reproduction step was checked before/after the fix."
    ),
    "no_acceptance_criteria": (
        "The request has a vague success bar with no concrete criteria — "
        "'make it faster', 'clean this up', 'don't change the format/API' — "
        "so 'done' is subjective."
    ),
    "no_visual_reference": (
        "The request is about visual/UI appearance ('make it look better', "
        "'match the design') but no actual image, mockup, or design file was "
        "ever provided anywhere in the conversation."
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
        "sequential changes that get addressed one at a time. Those are healthy "
        "iteration, not repeated wrong fixes."
    ),
    "no_verification_by_user": (
        "There is POSITIVE evidence the user did not verify the fix — e.g. the "
        "user's very next message reports the SAME bug still happening after "
        "being told it was fixed, or the user explicitly says they didn't test "
        "it, or accepts a claim of success with visible skepticism/haste that "
        "contradicts having actually checked. Do NOT flag a session just "
        "because the transcript itself doesn't show a test command or a "
        "screenshot — the user may well have tested it outside the chat, and "
        "the absence of proof in the transcript is not evidence of absence of "
        "verification. Only flag when something in the conversation actively "
        "suggests the user skipped verifying, not merely that verification "
        "isn't visible."
    ),
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
    with open(local_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


# ----------------------------------------------------------------------
# Case-file extraction (condensed session summary fed to the judge)
# ----------------------------------------------------------------------

def build_case_file(events):
    user_messages = []       # [(turn_no, text, has_image)]
    files_touched = set()
    test_runs = []            # [{"command": str, "result": str|None, "id": tool_use_id}]
    hedge_snippets = []       # up to 3 short excerpts of hedging thinking-then-submit
    turn_no = 0

    for event in events:
        etype = event.get("type")
        message = event.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None

        if etype == "assistant":
            turn_no += 1
            if isinstance(content, list):
                hedge_text = None
                saw_tool_use = False
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "thinking" and any(
                        p in b.get("thinking", "").lower() for p in HEDGE_PHRASES
                    ):
                        hedge_text = b.get("thinking", "")
                    if b.get("type") == "tool_use":
                        saw_tool_use = True
                if hedge_text and saw_tool_use and len(hedge_snippets) < 3:
                    hedge_snippets.append(hedge_text[:150])

                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        name = b.get("name", "")
                        tool_input = b.get("input", {}) or {}
                        if any(k in name.lower() for k in ("edit", "write")) and "todowrite" not in name.lower():
                            fp = tool_input.get("file_path")
                            if fp:
                                files_touched.add(fp)
                        if "bash" in name.lower():
                            command = tool_input.get("command", "") or ""
                            if any(k in command.lower() for k in TEST_KEYWORDS):
                                test_runs.append({"command": command[:120], "result": None, "id": b.get("id")})

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
                user_messages.append((turn_no, text, has_image))

            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tool_use_id = b.get("tool_use_id")
                        result_text = str(b.get("content", ""))[:300].lower()
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
        "hedge_snippets": hedge_snippets,
        "total_turns": turn_no,
    }


def compute_scope_flags(case_file):
    """Direct metadata check — no LLM needed. Two independent flags."""
    return {
        "scope_files_too_many": len(case_file["files_touched"]) >= SCOPE_FILES_TOO_MANY_THRESHOLD,
        "scope_turns_too_long": case_file["total_turns"] >= SCOPE_TURNS_TOO_LONG_THRESHOLD,
    }


# ----------------------------------------------------------------------
# LLM-as-judge
# ----------------------------------------------------------------------

def build_symptom_judge_prompt(case_file):
    msgs = "\n".join(
        f"{i+1}. [{'image attached' if img else 'text only'}] {text}"
        for i, (_, text, img) in enumerate(case_file["user_messages"])
    ) or "(no user messages)"
    tests = "\n".join(
        f"- `{t['command']}` -> {t['result'] or 'unclear'}" for t in case_file["test_runs"]
    ) or "(none run)"
    hedges = "\n".join(f'- "{h}"' for h in case_file["hedge_snippets"]) or "(none found)"
    defs = "\n".join(f"- {name}: {desc}" for name, desc in SYMPTOM_DEFINITIONS.items())

    return (
        "You are reviewing one coding-agent session for signs of 'vibe-fixing' "
        "(accepting fixes without proper spec or verification). Here is the "
        "session, condensed:\n\n"
        f"USER MESSAGES (in order):\n{msgs}\n\n"
        f"FILES TOUCHED: {len(case_file['files_touched'])} files\n\n"
        f"TEST/BUILD COMMANDS RUN:\n{tests}\n\n"
        f"ASSISTANT HEDGING (thinking showed doubt, but it still submitted a tool call):\n{hedges}\n\n"
        f"SYMPTOM DEFINITIONS:\n{defs}\n\n"
        "For each symptom, decide if it is present in THIS session. Answer with "
        "ONLY a JSON object, no other text, in exactly this shape:\n"
        "{\n"
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
            max_tokens=800,   # bumped slightly for a more verbose model (was 600 for Haiku)
            temperature=0,
            timeout=30.0,   # fail fast instead of hanging indefinitely on a stalled connection
            messages=[{"role": "user", "content": build_symptom_judge_prompt(case_file)}],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# Main driver — console output only, no files written
# ----------------------------------------------------------------------

def run(session_ids, paths, sample=SAMPLE, verbose=True):
    print(f"(sample = {'ALL' if sample is None else sample} sessions)")

    # timeout=30 is a backstop in case a request never triggers judge_symptoms's
    # own per-call timeout (e.g. hangs during connection setup); without this,
    # a stalled network call can block the whole script forever with no error.
    client = OpenAI(base_url=LITELLM_BASE_URL, timeout=30.0)
    ids = session_ids if sample is None else session_ids[:sample]
    total = len(ids)
    print(f"total sessions to process: {total:,}\n")

    counts = Counter()
    n_ok = n_skipped = n_judged = 0
    evidence_samples = {name: [] for name in SYMPTOM_DEFINITIONS}
    debug_steps = 3   # print fine-grained per-step timing for the first few sessions only

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

    return counts, n_judged, n_ok


if __name__ == "__main__":
    print("Loading dataset tables from HuggingFace (this can take a while, no progress bar)...")
    sessions, logs = load_tables()
    print(f"Loaded {len(sessions):,} sessions and {len(logs):,} session logs.")

    all_ids = claude_code_session_ids(sessions)
    paths = path_map(logs)
    print(f"Filtered to {len(all_ids):,} Claude Code sessions.\n")

    run(all_ids, paths)