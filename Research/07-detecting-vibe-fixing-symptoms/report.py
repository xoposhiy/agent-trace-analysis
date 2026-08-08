"""
report.py — reads a results.json (produced by run_experiment.py) and writes
the markdown report. No dependency on the dataset or the API.

Everything the report claims about the run — how many sessions, whether that
was a sample or the whole dataset, which rendering caps the judge saw, how many
calls failed — is read out of results.json rather than hardcoded here, so the
prose can't drift from what run_experiment.py actually did. Older results files
missing those keys still render; the affected sections degrade to what they can
actually support.

Run directly:

    python report.py
    python report.py my_results.json out.md
"""

import json
import sys

CATEGORY_DESCRIPTIONS = {
    "not_enough_verification": "The implementation wasn't actually checked before being treated as finished.",
    "not_enough_specification": "The user's request wasn't clear enough to act on.",
}

SUBCATEGORY_DESCRIPTIONS = {
    "not_enough_verification": {
        "not-tested": "Agent claims the task is finished but never verified it (no test, no manual check).",
        "self-report": "Agent itself says some important part wasn't tested.",
        "ask-for-manual-testing": "Agent asks the human to test something manually.",
        "repetitive-bug-fixes": "After the agent called it done, the user tested manually and reported bugs.",
    },
    "not_enough_specification": {
        "no-spec-detected": "User asked for an implementation without a detailed enough spec.",
        "repetitive-requirements-fixes": "Agent fixed it the wrong way and the user pushed back, repeatedly.",
        "self-report": "Agent itself says it doesn't have enough specification.",
    },
}

SINGLE_SYMPTOM_DESCRIPTIONS = {
    "no_visual_reference": "The user asks for a UI/visual change, but gives no image or design file.",
}

SCOPE_DESCRIPTIONS = {
    "scope_files_too_many": "Too many files were changed in one session",
    "scope_turns_too_long": "The session had an unusually high number of turns",
}

SCOPE_NAMES = ["scope_files_too_many", "scope_turns_too_long"]


def load_results(path="results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _scope_sentence(results):
    """Describe what this run actually covered, from the run's own metadata.
    Never hardcode it: SAMPLE in run_experiment.py is a knob, and a report that
    claims "all sessions" after a 15-session sample is simply wrong."""
    n_ok = results["n_ok"]
    n_candidates = results.get("n_candidates")

    if "sample_requested" not in results:
        # Pre-dating the metadata; say only what we can actually support.
        return (
            f"I checked **{n_ok:,} coding sessions** from the SWE-Chat dataset "
            "(agent: Claude Code). This results file does not record whether that "
            "was the whole dataset or a sample.\n"
        )
    if results["sample_requested"] is None:
        return (
            f"I checked **every parseable session** — {n_ok:,} in total — from the "
            "SWE-Chat dataset (agent: Claude Code). Short sessions were included "
            "alongside long ones.\n"
        )
    drawn_from = f" drawn from the {n_candidates:,} parseable Claude Code sessions" if n_candidates else ""
    return (
        f"I checked a **sample of {n_ok:,} sessions**{drawn_from} in the SWE-Chat "
        "dataset. The sample is the first N sessions in dataset order, **not a random "
        "sample**, so it is a spot check rather than a dataset-wide estimate. Short "
        "sessions were included alongside long ones.\n"
    )


def write_markdown_report(results, path="vibe_fixing_report.md"):
    counts = results["counts"]
    occurrence_counts = results.get("occurrence_counts", {})
    category_counts = results.get("category_counts", {})
    call_successes = results["call_successes"]
    n_ok = results["n_ok"]
    evidence_samples = results["evidence_samples"]
    timing = results.get("timing", {})

    lines = []
    lines.append("# Vibe-Fixing Symptoms in the SWE-Chat Dataset\n")
    lines.append(
        "This report shows how often \"vibe-fixing\" happens in coding-agent sessions — "
        "the agent's work moving forward without enough specification or enough "
        "verification. " + _scope_sentence(results)
    )

    lines.append("## What I Looked For\n")
    lines.append(
        "Two main categories, each broken into subcategories, plus one standalone "
        "symptom and two metadata-only checks:\n"
    )
    lines.append("| Category / Symptom | What it means |")
    lines.append("|---|---|")
    for cat_name, desc in CATEGORY_DESCRIPTIONS.items():
        lines.append(f"| **`{cat_name}`** | {desc} |")
        for subcat, sub_desc in SUBCATEGORY_DESCRIPTIONS[cat_name].items():
            lines.append(f"| &nbsp;&nbsp;`{subcat}` | {sub_desc} |")
    for name, desc in SINGLE_SYMPTOM_DESCRIPTIONS.items():
        lines.append(f"| `{name}` | {desc} |")
    for name, desc in SCOPE_DESCRIPTIONS.items():
        lines.append(f"| `{name}` | {desc} |")
    lines.append("")

    caps = results.get("rendering_caps", {})
    thinking_cap = caps.get("thinking_chars_per_block")
    tool_cap = caps.get("tool_result_chars")
    user_cap = caps.get("user_message_chars")

    lines.append("## How I Detected Them\n")
    lines.append(
        "**LLM-as-judge, one call per category** (3 calls per session total: "
        "`not_enough_verification`, `not_enough_specification`, `no_visual_reference`). "
        "Each session's raw transcript is rendered as a single chronological, "
        "typed-block timeline — every user message, every piece of agent thinking, "
        "and every tool call together with its raw result, in the exact order they "
        "happened, each tagged with the message number it occurred at (rendering "
        "approach inspired by [VCC](https://github.com/lllyasviel/VCC)). That same "
        "timeline is reused across a session's 3 calls.\n\n"
        "Each call returns **every occurrence** it finds, not just the first — a "
        "session can show the same subcategory multiple times (e.g. the agent asks "
        "for manual testing twice, or gets pushed back on requirements three times), "
        "and each one is recorded with its own message location and evidence.\n\n"
        "We do NOT pre-label which tool calls are \"tests\" or which files are "
        "\"specs\" using keyword lists or filename patterns — every tool call and "
        "its raw output are shown to the judge as-is, and it decides for itself.\n\n"
        "**Metadata-only rules.** `scope_files_too_many` and `scope_turns_too_long` "
        "don't need an LLM — just a count of files touched and assistant turns per "
        "session, flagged above a threshold.\n"
    )

    if caps:
        lines.append(
            "**What the judge does and doesn't see.** The timeline is condensed, not "
            "verbatim, and the caps matter when reading the numbers below — two of "
            "these checks are judgments about something being *absent*, which "
            "truncation can manufacture:\n"
        )
        if thinking_cap:
            lines.append(
                f"- Agent thinking has no session-wide cap, but an individual thinking "
                f"block longer than {thinking_cap:,} characters is shown as its first "
                "and last portions only, explicitly marked as truncated."
            )
        if tool_cap:
            lines.append(
                f"- Each tool result is cut to {tool_cap:,} characters. The tool *call* "
                "is always visible, but a long test run's actual output may be clipped."
            )
        if user_cap:
            lines.append(
                f"- Each user message is cut to {user_cap:,} characters, so a spec "
                "buried at the end of a very long request can be lost — which pushes "
                "`no-spec-detected` toward false positives."
            )
        lines.append("")

    lines.append("## Results\n")
    lines.append(
        "\"Sessions\" counts sessions with at least one occurrence; \"Total "
        "occurrences\" counts every occurrence, so a session flagged three times "
        "contributes 1 and 3 respectively. Percentages are per-check: each LLM check "
        "is divided by the number of sessions where **that** call succeeded, and the "
        "metadata-only checks by all judged sessions — so if any calls failed, the "
        "denominators differ slightly between rows. See Run Reliability below.\n"
    )
    lines.append("| Check | Sessions | % of judged sessions | Total occurrences |")
    lines.append("|---|---|---|---|")
    for cat_name in ("not_enough_verification", "not_enough_specification"):
        denom = call_successes.get(cat_name, 0) or 1
        cat_sessions = category_counts.get(cat_name, 0)
        lines.append(f"| **`{cat_name}`** (any) | {cat_sessions:,} | {round(100*cat_sessions/denom)}% | — |")
        for subcat in SUBCATEGORY_DESCRIPTIONS[cat_name]:
            key = f"{cat_name}:{subcat}"
            c = counts.get(key, 0)
            occ = occurrence_counts.get(key, 0)
            lines.append(f"| &nbsp;&nbsp;`{subcat}` | {c:,} | {round(100*c/denom)}% | {occ:,} |")
    for name in SINGLE_SYMPTOM_DESCRIPTIONS:
        denom = call_successes.get(name, 0) or 1
        c = counts.get(name, 0)
        occ = occurrence_counts.get(name, 0)
        lines.append(f"| `{name}` | {c:,} | {round(100*c/denom)}% | {occ:,} |")
    for name in SCOPE_NAMES:
        denom = n_ok or 1
        c = counts.get(name, 0)
        lines.append(f"| `{name}` | {c:,} | {round(100*c/denom)}% | — |")
    lines.append("")

    lines.append("## Examples\n")
    lines.append(
        "For each subcategory/symptom flagged by the judge, real examples pulled "
        "from this run (session id, which message(s) the evidence came from, and "
        "the judge's one-line reason). Spot-check material, not proof.\n"
    )
    for cat_name in ("not_enough_verification", "not_enough_specification"):
        for subcat in SUBCATEGORY_DESCRIPTIONS[cat_name]:
            key = f"{cat_name}:{subcat}"
            lines.append(f"**`{cat_name}` → `{subcat}`**\n")
            samples = evidence_samples.get(key) or []
            if samples:
                for s in samples:
                    ev_clean = (s.get("evidence") or "").strip() or "(no evidence text returned)"
                    loc = s.get("location", "n/a")
                    lines.append(f"- [`{s['session_id']}`] ({loc}) {ev_clean}")
            else:
                lines.append("- (no examples captured in this run)")
            lines.append("")
    for name in SINGLE_SYMPTOM_DESCRIPTIONS:
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

    lines.append("## Run Reliability\n")
    call_failures = results.get("call_failures", {})
    n_empty = results.get("n_empty")
    n_skipped = results.get("n_skipped", 0)
    judge_model = results.get("judge_model")
    reliability = [
        f"**{n_ok:,}** sessions were judged"
        + (f" by `{judge_model}`." if judge_model else ".")
    ]
    if n_empty is not None:
        reliability.append(
            f"**{n_empty:,}** {'was' if n_empty == 1 else 'were'} skipped as having no "
            f"user messages at all, and **{n_skipped:,}** could not be downloaded or "
            "parsed."
        )
    else:
        reliability.append(
            f"**{n_skipped:,}** {'was' if n_skipped == 1 else 'were'} skipped."
        )
    lines.append(" ".join(reliability) + "\n")

    lines.append("| Call | Succeeded | Failed | % of judged sessions covered |")
    lines.append("|---|---|---|---|")
    for call_name in ("not_enough_verification", "not_enough_specification", "no_visual_reference"):
        ok = call_successes.get(call_name, 0)
        failed = call_failures.get(call_name, 0)
        coverage = round(100 * ok / n_ok) if n_ok else 0
        lines.append(f"| `{call_name}` | {ok:,} | {failed:,} | {coverage}% |")
    lines.append("")

    if not call_failures and "call_failures" not in results:
        lines.append(
            "_This results file predates per-call failure tracking, so failures are "
            "not broken out above; a call that never returned a usable result is "
            "simply missing from its \"Succeeded\" count._\n"
        )
    diagnostics = results.get("diagnostics") or {}
    if diagnostics:
        lines.append(
            "Failures and structured-output fallbacks, by reason (a non-empty list "
            "here means some sessions were judged through the plain-JSON fallback "
            "path rather than the enforced schema):\n"
        )
        for reason, n in sorted(diagnostics.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{n:,}×` {reason}")
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
        lines.append("| Call | Avg call time | Avg prompt size |")
        lines.append("|---|---|---|")
        for call_name in ("not_enough_verification", "not_enough_specification", "no_visual_reference"):
            c = timing.get("call_count_by_call", {}).get(call_name, 0) or 1
            avg_time = timing.get("call_total_s_by_call", {}).get(call_name, 0) / c
            avg_chars = timing.get("prompt_chars_total_by_call", {}).get(call_name, 0) / c
            lines.append(f"| `{call_name}` | {avg_time:.2f}s | {avg_chars:,.0f} chars |")
        lines.append("")

    lines.append("## A Note of Caution\n")
    lines.append(
        "These categories replace the earlier separate symptom list "
        "(`no_spec`, `no_closed_loop`, `no_acceptance_criteria`, "
        "`repetitive_fix_attempts`), regrouping them by root cause — a "
        "verification gap vs. a specification gap — and splitting "
        "\"repetitive fixes\" into two distinct subcategories depending on whether "
        "the repeated correction was about a technical bug or a requirements "
        "misunderstanding. Spot-checking the Examples section above against real "
        "transcripts is recommended before citing these numbers externally.\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    report_path = sys.argv[2] if len(sys.argv) > 2 else "vibe_fixing_report.md"
    results = load_results(results_path)
    write_markdown_report(results, report_path)