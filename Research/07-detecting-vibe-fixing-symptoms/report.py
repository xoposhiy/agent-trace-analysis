"""
report.py — reads a results.json (produced by run_experiment.py) and writes
the markdown report. Has NO dependency on the dataset or the API — it only
reads the saved JSON, so it can be rerun any time to regenerate the report
(e.g. after tweaking report wording) without touching HuggingFace or
spending any LLM budget again.

Run directly:

    python report.py                      # reads results.json, writes vibe_fixing_report.md
    python report.py my_results.json out.md
"""

import json
import sys

SYMPTOM_DESCRIPTIONS_SHORT = {
    "no_spec": "The user's request is very short and unclear, or the agent shows doubt but still submits an answer",
    "no_closed_loop": "The user asks for a fix, but there is no way to check if it worked (no test run)",
    "no_acceptance_criteria": "The user's goal is vague (\"make it faster\", \"clean this up\"), with no clear target",
    "no_visual_reference": "The user asks for a UI/visual change, but gives no image or design file",
    "repetitive_fix_attempts": "The agent fixes the same bug wrong more than once, and the user has to report it again",
    "scope_files_too_many": "Too many files were changed in one session",
    "scope_turns_too_long": "The session had an unusually high number of turns",
}

LLM_SYMPTOM_ORDER = ["no_spec", "no_closed_loop", "no_acceptance_criteria",
                     "no_visual_reference", "repetitive_fix_attempts"]
SCOPE_NAMES = ["scope_files_too_many", "scope_turns_too_long"]


def load_results(path="results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_markdown_report(results, path="vibe_fixing_report.md"):
    counts = results["counts"]
    call_successes = results["call_successes"]
    n_ok = results["n_ok"]
    evidence_samples = results["evidence_samples"]
    timing = results.get("timing", {})

    ordered = ["no_closed_loop", "no_spec", "no_acceptance_criteria",
               "scope_turns_too_long", "scope_files_too_many", "repetitive_fix_attempts", "no_visual_reference"]

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
    lines.append("I checked each session for 7 checks (5 judged by an LLM, 2 by simple thresholds):\n")
    lines.append("| Symptom | What it means |")
    lines.append("|---|---|")
    for name in LLM_SYMPTOM_ORDER + SCOPE_NAMES:
        lines.append(f"| `{name}` | {SYMPTOM_DESCRIPTIONS_SHORT[name]} |")
    lines.append("")

    lines.append("## How I Detected Them\n")
    lines.append(
        "I used two methods:\n\n"
        "**1. LLM-as-judge (Claude Haiku 4.5), one isolated call per symptom.** Each session's "
        "raw transcript is rendered as a single chronological, typed-block timeline — every user "
        "message, every piece of agent thinking, and every tool call together with its raw result, "
        "in the exact order they happened, each tagged with the turn number it occurred at (this "
        "rendering approach is inspired by "
        "[VCC](https://github.com/lllyasviel/VCC), a compiler for agent conversation logs). That "
        "same timeline is reused as a shared prefix across a session's 5 symptom calls. Each call "
        "also cites which turn(s) its evidence came from, so findings can be traced back to a "
        "specific point in the session rather than just a session-wide yes/no.\n\n"
        "Notably, we do NOT pre-label which tool calls are \"tests\" or which files are \"specs\" "
        "using keyword lists or filename patterns — every tool call and its raw output are shown "
        "to the judge as-is, and it decides for itself (e.g. whether a bash command was a "
        "meaningful verification step, or whether a file the agent read plausibly explains an "
        "otherwise-vague request).\n\n"
        "**2. Metadata-only rules.** The two `scope_*` symptoms don't need an LLM. I just count "
        "files touched and turns per session, and flag sessions above a threshold.\n"
    )

    lines.append("## Results\n")
    lines.append("| Symptom | Count | % of judged sessions |")
    lines.append("|---|---|---|")
    for name in ordered:
        denom = n_ok if name in SCOPE_NAMES else call_successes.get(name, 0)
        count = counts.get(name, 0)
        pct = round(100 * count / denom) if denom else 0
        lines.append(f"| `{name}` | {count:,} | {pct}% |")
    lines.append("")

    lines.append("## Examples\n")
    lines.append(
        "For each symptom flagged by the LLM judge, here are real examples pulled from this run "
        "(session id, the turn(s) the evidence came from, and the judge's one-line reason). These "
        "are spot-check material, not proof — always worth reading the underlying transcript "
        "before trusting an aggregate number.\n"
    )
    for name in LLM_SYMPTOM_ORDER:
        lines.append(f"**`{name}`**\n")
        samples = evidence_samples.get(name) or []
        if samples:
            for s in samples:
                ev_clean = (s.get("evidence") or "").strip() or "(no evidence text returned)"
                loc = s.get("location", "n/a")
                lines.append(f"- [`{s['session_id']}`] ({loc}) {ev_clean}")
        else:
            lines.append("- (no examples captured in this run)")
        lines.append("")

    if timing:
        lines.append("## Performance Notes\n")
        n_for_avg = max(n_ok, 1)
        dl = timing.get("download_total_s", 0)
        parse = timing.get("parse_total_s", 0)
        lines.append(
            f"Total wall-clock time for this run: {timing.get('total_wall_clock_s', 0):.0f}s. "
            f"Average download time per session: {dl / n_for_avg:.2f}s. "
            f"Average parse time per session: {parse / n_for_avg:.2f}s.\n"
        )
        lines.append("| Symptom | Avg call time | Avg prompt size |")
        lines.append("|---|---|---|")
        for name in LLM_SYMPTOM_ORDER:
            c = timing.get("call_count_by_symptom", {}).get(name, 0) or 1
            avg_time = timing.get("call_total_s_by_symptom", {}).get(name, 0) / c
            avg_chars = timing.get("prompt_chars_total_by_symptom", {}).get(name, 0) / c
            lines.append(f"| `{name}` | {avg_time:.2f}s | {avg_chars:,.0f} chars |")
        lines.append("")

    lines.append("## A Note of Caution\n")
    lines.append(
        "`no_verification_by_user` has been removed from this report entirely — it was mostly "
        "detecting \"no proof shown in the transcript\" rather than \"the user actually skipped "
        "verifying,\" and a person could always test something outside the chat window, so it "
        "wasn't trustworthy as reported. The remaining symptoms rely on clearer, easier-to-check "
        "evidence (the actual request text, the raw tool calls and their results, file counts), "
        "but spot-checking the Examples section above against real transcripts is still "
        "recommended before citing these numbers externally.\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    report_path = sys.argv[2] if len(sys.argv) > 2 else "vibe_fixing_report.md"
    results = load_results(results_path)
    write_markdown_report(results, report_path)