"""
report.py — reads a results.json (produced by run_experiment.py) and writes
the markdown report. No dependency on the dataset or the API.

Everything the report claims about the run — session count, sample or whole
dataset, rendering caps, failed calls, how much of the old prompt count was
harness noise — is read out of results.json rather than hardcoded, so the prose
can't drift from what run_experiment.py did. Older results files missing those
keys still render, degraded to what they can support.

Key decisions:
    - The results table reads Problems | Evidence | % of prompts. "Problems"
      is distinct (category-or-subcategory, cause prompt) pairs, so the
      subcategory rows do NOT sum to the category row. The report body says so
      too — a reader who tries the addition will otherwise assume it is broken.
    - A finding has two location parts: the cause prompt it is counted
      against, and the blocks where it is observable. Examples print both; for
      the repetitive-* subcategories they point at different prompts, and only
      the cause sends the reader somewhere with nothing to see.
    - The unit of an example is a PROBLEM — one (session, cause prompt) pair —
      not a finding. Sampling is per subcategory; expansion is per problem,
      bounded by the category, on the problem's first appearance in it.
    - Coordinates the judge got wrong are shown, not hidden: nothing upstream
      repairs them, so a cause on a harness wrapper or on a nonexistent number
      renders marked, and its frequency is a line under Run Reliability.
      "Problems" counts every finding, "% of prompts" only those on a real
      prompt — the two columns disagreeing is information, not a bug.
    - Pre-split results files (single "location"/"evidence" string) are not
      migrated — their prompt denominator counted harness wrappers — but they
      still render as the old-methodology artefacts they are.

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


# How a cause that didn't land on a real user prompt is labelled in the
# examples. Spelled out rather than left as a bare "system"/"out_of_range":
# the reader following the citation needs to know why it won't resolve, and
# these are not errors to discount — only misattributed findings.
CAUSE_KIND_LABELS = {
    "system": " — **not a user prompt**: this number is a tooling-generated "
              "`[SYSTEM]` block, so the attribution missed (the finding still counts, "
              "but not toward the per-prompt rate)",
    "out_of_range": " — **no such prompt in this session**: the judge cited a number "
                    "the timeline never assigned (the finding still counts, but not "
                    "toward the per-prompt rate)",
}


def _loc(sample):
    """Render a finding's CAUSE — the prompt whose work led to the problem,
    and the one it is counted against. _evidence_lines below renders the other
    half: every block where it can be seen.

    Nothing upstream corrects a cause, so this has to render one pointing at a
    `[SYSTEM]` block or at a number the session never had. It says so instead
    of printing "prompt 47" and letting a reader hunt for a prompt 47.

    Pre-split results files stored a single "location"; they are not migrated
    but still render."""
    cause = sample.get("cause_prompt")
    if cause is None and "cause_prompt" not in sample:
        legacy = sample.get("location")
        if isinstance(legacy, int):
            return f"prompt {legacy}"
        return str(legacy) if legacy else "n/a"

    where = f"prompt {cause}" if cause is not None else "n/a"
    if not isinstance(cause, int) or isinstance(cause, bool):
        # Not a number — the judge's answer verbatim, quoted so it is obvious
        # it is being shown rather than interpreted.
        where = f"`{cause!r}`" if cause is not None else "n/a"
    return where + CAUSE_KIND_LABELS.get(sample.get("cause_kind"), "")


def _evidence_lines(holder, indent="    "):
    """The evidence list as nested bullets: where to look, and what is there.

    Coordinates print in the joined `P.S` form the timeline headers use, so a
    reader can search for the label directly. (The judge must ANSWER in two
    integer fields; reading it back out joined is what makes it findable.)

    They are printed exactly as the judge gave them, including ones that don't
    resolve — an out-of-range prompt is flagged inline, a step past the end of
    its prompt printed unchanged. Tidying those away would leave a reader
    unable to tell a precise judge from a guessing one.

    An empty list means the judge returned no evidence for this finding."""
    evidence = holder.get("evidence")
    # Pre-split results files stored "evidence" as a single sentence with no
    # coordinate of its own.
    if isinstance(evidence, str):
        text = evidence.strip()
        return [f"{indent}- {text}"] if text else []
    if not isinstance(evidence, list):
        return []

    lines = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        step = item.get("step")
        if prompt is None:
            where = "unknown location"
        elif not isinstance(prompt, int) or isinstance(prompt, bool):
            # Not a number, so the joined form would read as a coordinate it
            # isn't ("4.2" + step 1 renders as "4.2.1"). Quote it instead.
            where = f"prompt `{prompt!r}`" + ("" if step is None else f", step {step}")
        elif step is None:
            where = f"prompt {prompt}"
        else:
            where = f"{prompt}.{step}"
        if item.get("prompt_kind") == "out_of_range":
            where += " (no such prompt in this session)"
        note = (item.get("note") or "").strip() or "(no note returned)"
        lines.append(f"{indent}- {where} — {note}")
    return lines


def _describe(cat_name, subcat):
    """The one-line meaning of a subcategory, used as the headline of a
    finding inside an example block. A finding carries no summary of its own,
    so this is what says WHAT was found before the notes say where."""
    if subcat in SINGLE_SYMPTOM_DESCRIPTIONS:
        return SINGLE_SYMPTOM_DESCRIPTIONS[subcat]
    return SUBCATEGORY_DESCRIPTIONS.get(cat_name, {}).get(subcat) or str(subcat)


def _problem_id(sample):
    """The identity of a problem: (session, cause) exactly as the judge gave
    it — a verbatim non-numeric cause groups with itself. Guards hashability
    so a malformed legacy entry can't raise."""
    cause = sample.get("cause_prompt", sample.get("location"))
    try:
        hash(cause)
    except TypeError:
        cause = repr(cause)
    return (sample.get("session_id"), cause)


def _problem_blocks(samples, cat_name, section_subcat, expanded):
    """Render one subcategory's examples, as PROBLEMS rather than findings.

    A drawn example is a (session, cause prompt) pair, expanded into
    everything the judge found on that prompt WITHIN THIS CATEGORY, including
    findings from a sibling subcategory, tagged with it. The category boundary
    is enforced by how run_experiment.py buckets findings, not here, so a
    specification finding cannot leak into a verification example.

    `expanded` is shared across one category's sections and maps a problem to
    the section that already showed it. A problem drawn under two
    subcategories gets a pointer the second time — repeating it would also
    make the section look like it found twice as much as it did.
    """
    if not samples:
        return ["- (no examples captured in this run)"]

    out = []
    for sample in samples:
        head = f"- [`{sample['session_id']}`] {_loc(sample)}"
        problem_id = _problem_id(sample)
        if problem_id in expanded:
            out.append(f"{head} — already shown under `{expanded[problem_id]}` above")
            continue
        expanded[problem_id] = section_subcat

        findings = sample.get("findings")
        if not isinstance(findings, list):
            # Pre-dating the problem-as-example unit: one flat finding per
            # entry, nothing to expand.
            out.append(head)
            out.extend(_evidence_lines(sample, indent="  ") or ["  - (no evidence returned)"])
            continue

        # The section's own subcategory leads, so the block opens by answering
        # the heading's question; siblings follow in encounter order.
        ordered = ([f for f in findings if f.get("subcategory") == section_subcat]
                   + [f for f in findings if f.get("subcategory") != section_subcat])
        n_evidence = sum(len(f.get("evidence") or []) for f in ordered)
        out.append(
            f"{head} — {len(ordered)} finding{'' if len(ordered) == 1 else 's'}, "
            f"{n_evidence} evidence"
        )
        for finding in ordered:
            subcat = finding.get("subcategory")
            tag = "" if subcat == section_subcat else f"[{subcat}] "
            out.append(f"  - {tag}{_describe(cat_name, subcat)}")
            out.extend(_evidence_lines(finding) or ["    - (no evidence returned)"])
    return out


def load_results(path="results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _threshold_sentence(results):
    """The actual cut-offs behind the two metadata checks. Read from the run,
    like the rendering caps, rather than restated here — they live in
    case_file.py, which this module deliberately does not import. Runs that
    predate the key say nothing instead of guessing a number."""
    thresholds = results.get("scope_thresholds") or {}
    files = thresholds.get("files_too_many")
    turns = thresholds.get("turns_too_long")
    if not files or not turns:
        return "This results file does not record the thresholds used.\n"
    return (
        f"A session is flagged at **{files:,} or more files touched** and at "
        f"**{turns:,} or more assistant turns** respectively.\n"
    )


def _scope_sentence(results):
    """Describe what this run covered, from the run's own metadata. Never
    hardcoded: SAMPLE in run_experiment.py is a knob, and a report claiming
    "all sessions" after a 15-session sample is simply wrong."""
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
    drawn_from = f" drawn from the {n_candidates:,} Claude Code sessions" if n_candidates else ""

    if results.get("sampling") == "random":
        seed = results.get("sample_seed")
        seed_note = f" (RNG seed `{seed}`)" if seed is not None else ""
        # Widest 95% interval for a proportion at this n: 1.96*sqrt(0.25/n).
        # Stated for the session percentages only — the per-prompt rates share
        # a denominator whose items are not independent (prompts cluster in
        # sessions), so the same arithmetic would understate their error.
        moe = 98.0 / (n_ok ** 0.5) if n_ok else 0.0
        return (
            f"I checked a **random sample of {n_ok:,} sessions**{drawn_from} in the "
            f"SWE-Chat dataset, drawn uniformly without replacement{seed_note}. "
            "Selection is therefore unbiased, but each session percentage below "
            f"carries a sampling error of up to ±{moe:.0f} points at 95% confidence, "
            "so read them as approximate. The seed reproduces the draw, not the run: "
            "the judge samples, so re-judging the same sessions will not reproduce "
            "the same counts. Short sessions were included alongside long ones.\n"
        )

    # No "sampling" key: predates random sampling, when the sample was
    # whatever came first in dataset order. Say so rather than inheriting the
    # stronger claim.
    return (
        f"I checked a **sample of {n_ok:,} sessions**{drawn_from} in the SWE-Chat "
        "dataset. The sample is the first N sessions in dataset order, **not a random "
        "sample**, so it is a spot check rather than a dataset-wide estimate. Short "
        "sessions were included alongside long ones.\n"
    )


def write_markdown_report(results, path="vibe_fixing_report.md"):
    counts = results["counts"]
    problems = results.get("problems", {})
    # The rate numerator is NOT `problems`: a cause that didn't land on a real
    # user prompt still counts as a problem but has nothing to be a fraction
    # of in a denominator made of real prompts.
    problems_real = results.get("problems_real", problems)
    evidence_counts = results.get("evidence_counts", {})
    # A pre-split results file has neither. Rendering 0 would read as "nothing
    # backs these up" instead of "never recorded", so both columns go blank.
    has_problem_stats = "problems" in results
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

    total_prompts = results.get("total_user_prompts")
    total_system_events = results.get("total_system_events")
    has_prompt_stats = bool(total_prompts)
    pdenom = total_prompts or 1
    # The wrapper share is stated twice — once here as the reason the
    # denominator excludes them, once under Results as the size of the
    # exclusion — so it is computed once and never written as a literal.
    n_text_events = (total_prompts or 0) + (total_system_events or 0)
    wrapper_share = 100 * total_system_events / n_text_events if total_system_events else None

    caps = results.get("rendering_caps", {})
    thinking_cap = caps.get("thinking_chars_per_block")
    reply_cap = caps.get("assistant_text_chars")
    tool_cap = caps.get("tool_result_chars")
    user_cap = caps.get("user_message_chars")

    lines.append("## How I Detected Them\n")
    lines.append(
        "**LLM-as-judge, one call per category** (3 calls per session total: "
        "`not_enough_verification`, `not_enough_specification`, `no_visual_reference`). "
        "Each session's raw transcript is rendered as a single chronological, "
        "typed-block timeline — every user message, every reply the agent wrote "
        "back to the user, every piece of agent thinking, and every tool call "
        "together with its raw result, in the exact order they happened (rendering "
        "approach inspired by [VCC](https://github.com/lllyasviel/VCC)). That same "
        "timeline is reused across a session's 3 calls.\n\n"
        "**Every block has a two-part address `P.S`.** A user message opens "
        "`[USER P.1]` and every block after it — thinking, tool call, reply — "
        "carries the same prompt number `P` with an increasing step `S`, so the "
        "judge reads both numbers off the block it is citing rather than counting "
        "messages itself.\n\n"
        "**Findings name a cause and its evidence, separately.** The `cause_prompt` "
        "is the single prompt whose work led to the problem — that is what the "
        "counting is done on. The `evidence` list is every block where the problem "
        "is actually visible, each with its own coordinate and a one-line note. The "
        "two are often the same prompt, but for the repetitive-* subcategories they "
        "are deliberately different: the cause is the earlier prompt whose work was "
        "called finished, while the only observable symptom is the user's later "
        "complaint. Reporting only the cause used to send a reader to a block where "
        "there is nothing to see. Every coordinate is checked against the blocks the "
        "session actually contains and labelled where it does not resolve — but "
        "nothing is corrected and no finding is dropped for citing badly, so the "
        "misses stay visible and countable.\n\n"
        "**Not every user message is a user prompt.** "
        + (f"{wrapper_share:.0f}% of the " if wrapper_share is not None else "Many ")
        + "text-carrying user events in these sessions are wrappers the tooling "
        "emitted itself — slash-command invocations, context-compaction summaries, "
        "`[Request interrupted by user]` markers. They are labelled `[SYSTEM P.1]`, "
        "shown to the judge in full (a `/clear` is the reason the context suddenly "
        "forgot everything, and an interruption is quotable evidence), and they keep "
        "their number so numbering stays purely positional — but they are excluded "
        "from the prompt denominator, because no human asked for anything in them.\n\n"
        "Each call returns **every occurrence** it finds, not just the first — a "
        "session can show the same subcategory multiple times (e.g. the agent asks "
        "for manual testing twice, or gets pushed back on requirements three times), "
        "and each one is recorded with its own cause and evidence.\n\n"
        "We do NOT pre-label which tool calls are \"tests\" or which files are "
        "\"specs\" using keyword lists or filename patterns — every tool call and "
        "its raw output are shown to the judge as-is, and it decides for itself.\n\n"
        "**Metadata-only rules.** `scope_files_too_many` and `scope_turns_too_long` "
        "don't need an LLM — just a count of files touched and assistant turns per "
        "session, flagged above a threshold. " + _threshold_sentence(results)
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
        if reply_cap:
            lines.append(
                f"- An agent reply longer than {reply_cap:,} characters is shown as "
                "its first and last portions only, explicitly marked as truncated. "
                "Replies are rendered separately from thinking, so a claim the agent "
                "only made privately is not counted as something it told the user."
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
        "A **problem** is one pair of (category, cause prompt). One prompt flagged "
        "with three subcategories of the same category is one problem with three "
        "confirmations, not three problems; the same prompt flagged under both "
        "verification and specification is two problems, one per category. "
        "\"Evidence\" counts the individual blocks the judge cited as showing it, "
        "and \"Sessions\" counts sessions with at least one finding.\n"
    )
    lines.append(
        "**The subcategory rows do not add up to the category row above them, and "
        "they are not supposed to.** They are counted at different levels: the "
        "category row collapses a prompt's subcategories into the single problem "
        "they describe, the subcategory rows keep them apart. A category row reading "
        "1 above three subcategory rows each reading 1 is the correct rendering of "
        "one prompt flagged three ways, not an arithmetic error.\n"
    )
    if has_prompt_stats:
        lines.append(
            f"The last column is the one to read for a rate: how many distinct prompts "
            f"ended in this problem, out of the **{total_prompts:,} real user prompts** "
            "across all judged sessions. Session percentages weigh a 2-prompt session "
            "the same as a 200-prompt one; the per-prompt rate does not.\n"
        )
        lines.append(
            "It counts only the problems whose cause landed on a real user prompt. "
            "Coordinates the judge returned are recorded as given, never corrected, so "
            "some findings name a `[SYSTEM]` block or a number the session never had; "
            "those are still counted under \"Problems\" and still appear in the "
            "examples, but they have no real prompt to be a fraction of, so \"% of "
            "prompts\" leaves them out. Where the two columns disagree, the gap is the "
            "judge's attribution error rate — see Run Reliability.\n"
        )
    if total_system_events:
        lines.append(
            f"That denominator excludes harness wrappers: of the {n_text_events:,} "
            f"text-carrying user events in these sessions, **{total_system_events:,} "
            f"({wrapper_share:.0f}%)** were slash-command invocations, context-compaction "
            "summaries or interruption markers rather than requests from a human. "
            "They stay visible in the timeline and keep their position in the "
            "numbering; they are simply not prompts. Earlier runs of this pipeline "
            "counted them, so their per-prompt rates were understated by roughly the "
            "same factor and are not comparable with the numbers below.\n"
        )
    lines.append(
        "Session percentages are per-check: each LLM check is divided by the number "
        "of sessions where **that** call succeeded, and the metadata-only checks by "
        "all judged sessions — so if any calls failed, the denominators differ "
        "slightly between rows. See Run Reliability below.\n"
    )

    header = "| Check | Sessions | % of sessions | Problems | Evidence |"
    divider = "|---|---|---|---|---|"
    if has_prompt_stats:
        header += " % of prompts |"
        divider += "---|"
    lines.append(header)
    lines.append(divider)

    def row(label, sessions, denom, key=None, indent=False, bold=False):
        name = f"&nbsp;&nbsp;`{label}`" if indent else (
            f"**`{label}`** (any)" if bold else f"`{label}`")
        n_problems = None if (key is None or not has_problem_stats) else problems.get(key, 0)
        cells = [name, f"{sessions:,}", f"{round(100*sessions/denom)}%",
                 "—" if n_problems is None else f"{n_problems:,}",
                 "—" if n_problems is None else f"{evidence_counts.get(key, 0):,}"]
        if has_prompt_stats:
            cells.append("—" if n_problems is None
                         else f"{100*problems_real.get(key, 0)/pdenom:.1f}%")
        lines.append("| " + " | ".join(cells) + " |")

    for cat_name in ("not_enough_verification", "not_enough_specification"):
        denom = call_successes.get(cat_name, 0) or 1
        # The category row keys off the bare call name, counted at the
        # category level upstream — NOT the sum of the subcategory keys.
        row(cat_name, category_counts.get(cat_name, 0), denom,
            key=cat_name, bold=True)
        for subcat in SUBCATEGORY_DESCRIPTIONS[cat_name]:
            key = f"{cat_name}:{subcat}"
            row(subcat, counts.get(key, 0), denom, key=key, indent=True)
    for name in SINGLE_SYMPTOM_DESCRIPTIONS:
        denom = call_successes.get(name, 0) or 1
        row(name, counts.get(name, 0), denom, key=name)
    for name in SCOPE_NAMES:
        # Metadata checks are whole-session properties: no per-prompt reading,
        # and no judge citation behind them.
        row(name, counts.get(name, 0), n_ok or 1)
    lines.append("")

    lines.append("## Examples\n")
    lines.append(
        "The unit here is a **problem**, not a finding: examples are drawn at random "
        f"from the (session, cause prompt) pairs found in this run — up to "
        f"{results.get('examples_per_key', 5)} per subcategory — and each drawn prompt "
        "is then shown with **everything this category found on it**, including "
        "findings that came in under a sibling subcategory. Those carry their own "
        "subcategory in square brackets. Having sent you to a specific prompt, the "
        "report may as well tell you the whole of what it saw there.\n"
    )
    lines.append(
        "Each finding lists the blocks the judge cited as showing it, as `P.S` — the "
        "same coordinate printed on the block in the timeline, so you can search for "
        "it directly. Open the session, go to those coordinates, and check. "
        "Spot-check material, not proof.\n"
    )
    if results.get("examples_seed") is not None:
        lines.append(
            f"The draw is seeded (`{results['examples_seed']}`), so re-rendering this "
            "results file selects the same examples.\n"
        )
    lines.append(
        "The category boundary is not crossed: a verification example never shows "
        "specification findings, even when both landed on the same prompt — the two "
        "categories are counted independently and are read independently. Within one "
        "category, a prompt drawn twice is expanded once and referenced thereafter.\n"
    )

    for cat_name in ("not_enough_verification", "not_enough_specification"):
        # Reset per category: "expand once" must not carry across a category
        # boundary, or a prompt found under both would be shown only once.
        expanded = {}
        for subcat in SUBCATEGORY_DESCRIPTIONS[cat_name]:
            key = f"{cat_name}:{subcat}"
            lines.append(f"**`{cat_name}` → `{subcat}`**\n")
            lines.extend(_problem_blocks(
                evidence_samples.get(key) or [], cat_name, subcat, expanded))
            lines.append("")
    for name in SINGLE_SYMPTOM_DESCRIPTIONS:
        # A single symptom is its own category with one subcategory: fresh
        # `expanded`, nothing to tag.
        lines.append(f"**`{name}`**\n")
        lines.extend(_problem_blocks(
            evidence_samples.get(name) or [], name, name, {}))
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

    # How often the judge pointed at something that isn't a user prompt — a
    # property of the JUDGE, not of the sessions, which is why it sits here
    # and not in Results. It bounds how far the rates above can be trusted.
    cause_kind_counts = results.get("cause_kind_counts") or {}
    n_findings = sum(cause_kind_counts.values())
    if n_findings:
        n_system = cause_kind_counts.get("system", 0)
        n_oor = cause_kind_counts.get("out_of_range", 0)
        n_missed = n_system + n_oor
        if n_missed:
            lines.append(
                f"**Attribution accuracy.** {n_missed:,} of {n_findings:,} findings "
                f"({100*n_missed/n_findings:.1f}%) named a cause that is not a real "
                f"user prompt: **{n_system:,}** pointed at a `[SYSTEM]` block (tooling "
                f"output, not a request) and **{n_oor:,}** at a number this session "
                "never assigned. Nothing was corrected or thrown away — those findings "
                "are counted under \"Problems\" and shown in the examples with a "
                "marker — but they are excluded from \"% of prompts\", which is why "
                "that column can be lower than the problem count implies. Treat this "
                "as the judge's citation error rate.\n"
            )
        else:
            lines.append(
                f"**Attribution accuracy.** All {n_findings:,} findings named a cause "
                "that resolves to a real user prompt in its session.\n"
            )
        lines.append(
            "This measures the `cause_prompt` only. The `evidence` coordinates are "
            "checked the same way and kept as given whether or not they resolve, but "
            "they are not aggregated into a rate — individual misses among them show "
            "up in the diagnostics list below and inline in the examples.\n"
        )

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
        # Two different things share this counter and they mean opposite
        # amounts of trouble: "validate:" entries are coordinates the judge got
        # wrong (recorded, never repaired), everything else is a call that fell
        # back to plain-JSON prompting or failed outright. Labelling the whole
        # list as fallbacks reads as a broken run whenever the judge merely
        # miscited a prompt number.
        validate, call_level = {}, {}
        for reason, n in diagnostics.items():
            if reason.startswith("validate: "):
                validate[reason.split(": ", 1)[1]] = n
            else:
                call_level[reason] = n

        if call_level:
            lines.append(
                "**Call-level failures and structured-output fallbacks.** Each entry "
                "is a call that failed outright or was answered through the plain-JSON "
                "fallback path rather than the enforced schema:\n"
            )
            for reason, n in sorted(call_level.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{n:,}×` {reason}")
            lines.append("")
        else:
            lines.append(
                "**No call-level failures.** Every judge call returned through the "
                "enforced schema; the plain-JSON fallback path was never used.\n"
            )

        if validate:
            lines.append(
                f"**Coordinate notes** — {sum(validate.values()):,} cited prompt or "
                "step numbers that did not resolve against the session. Recorded, not "
                "repaired: the findings carrying them are still counted:\n"
            )
            for reason, n in sorted(validate.items(), key=lambda kv: -kv[1]):
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
            f"Average parse time per session: {parse / n_for_avg * 1000:.0f}ms.\n"
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
    # The inflation factor is this run's own text-events-to-real-prompts
    # ratio, not a constant: it is a property of how many wrappers these
    # particular sessions contained, so a literal here goes stale every run.
    if has_prompt_stats and n_text_events:
        factor_clause = f"by {n_text_events / pdenom:.2f}x on this run's data"
    else:
        factor_clause = "by a factor this results file does not record"
    lines.append(
        "**Do not compare these per-prompt rates against earlier runs of this "
        "pipeline.** The denominator changed: harness wrappers used to be counted as "
        f"user prompts and no longer are, which lifts every per-prompt rate "
        f"{factor_clause}, with nothing about the sessions or the judge having "
        "changed. The prompt NUMBERS are unaffected — numbering stayed positional, so "
        "a citation still points at the same block it always did.\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    report_path = sys.argv[2] if len(sys.argv) > 2 else "vibe_fixing_report.md"
    results = load_results(results_path)
    write_markdown_report(results, report_path)