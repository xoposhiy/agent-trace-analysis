"""
run_experiment.py — the main loop: loads the dataset, runs every session
through classify.py's 3 judge calls (not_enough_verification,
not_enough_specification, no_visual_reference), aggregates results, and
saves everything to RESULTS_PATH so report.py can build the report later
without rerunning the experiment.

Run directly:

    python run_experiment.py            # default sample size
    python run_experiment.py 25         # 25 sessions, drawn at random
    python run_experiment.py 25 --seed 7
    python run_experiment.py --all

Key decisions:
    - The sample is drawn UNIFORMLY AT RANDOM (see select_sessions). Dataset
      order is chronological, so the old session_ids[:n] returned the earliest
      few days — a handful of projects, some the same work resumed next day.
      The seed is a CLI flag, recorded in results.json with the ids drawn.
    - 3 LLM calls per session: each grouped category covers its subcategories
      in one call.
    - A PROBLEM is the pair (category, cause_prompt). One prompt flagged with
      three subcategories of the same category is ONE problem with three
      confirmations; one flagged under both categories is TWO problems. Four
      numbers per key follow:
        * counts[key] — SESSIONS with at least one finding.
        * problems[key] — distinct (level_key, cause_prompt) pairs, counted at
          BOTH the category and the subcategory level. CONSEQUENCE: the
          subcategory rows do NOT sum to their category row.
        * problems_real[key] — the same, restricted to cause_kind == "real".
          This is the numerator of "% of prompts": the denominator is real
          prompts, so a problem on a wrapper or on a nonexistent number has
          nothing to be a fraction of and could push the rate over 100%.
        * evidence_counts[key] — evidence references returned. "How
          thoroughly was this shown", not a count of anything that happened.
    - evidence_samples is keyed by subcategory but its ENTRIES are problems:
      one (session_id, cause_prompt) pair plus every finding of that category
      on it. The draw is uniform over pairs — see EXAMPLES_PER_KEY.
    - Nothing is discarded for pointing at the wrong place. Misattributed
      findings count in problems, stay in the examples, and are tallied by
      cause_kind, which report.py turns into an attribution-miss line — a
      measure of the JUDGE, not of the sessions.
    - Failure accounting is explicit and separate: n_skipped (couldn't build),
      n_empty (no real user prompts), call_failures[call] (per call, so the
      rest of the session survives). Reasons land in classify.DIAGNOSTICS.
    - results.json also records the run's scope (sample_requested,
      n_candidates) and the rendering caps, so report.py never hardcodes
      claims about it.
    - Every judged session leaves an audit trail in LOGS_DIR: the exact prompt
      text plus the session's size and each classifier's verbatim answer.
    - total_user_prompts counts REAL prompts only. Wrappers used to inflate it
      by ~65% (508 of 1,284 text-carrying user events, 39.6%, across the 111
      cached transcripts), so rates are ~1.65x their old values with nothing
      about the sessions having changed. total_system_events is recorded so
      the report can state that share.
"""

import argparse
import json
import os
import random
import time
from collections import Counter

from openai import OpenAI

import case_file as cf
import classify as cl

SAMPLE = 10   # default when no n is given on the command line; None = every session
SEED = 20260809   # default RNG seed for the sample draw; override with --seed
RESULTS_PATH = "results.json"

# Per-session audit trail: exactly what was sent to the judge and exactly what
# came back. Without it a number in the report can only be checked by
# re-running the session, and TEMPERATURE is None, so sampling is not
# deterministic and the answer won't repeat.
WRITE_SESSION_LOGS = True
LOGS_DIR = "logs"

# Schema version of the .meta.json files. BUMP on any shape change, so a later
# analysis can filter instead of silently mixing revisions.
META_VERSION = 1

# Examples per subcategory in the report, drawn at RANDOM rather than in
# encounter order: with first-N, one chatty session can supply every example
# (on one run all five no_visual_reference examples came from one session out
# of six flagged).
#
# The unit drawn is a PROBLEM — a (session_id, cause_prompt) pair — and the
# example then shows everything found on that prompt within the category.
# Sampling findings instead left the rest of what the judge said about the
# same prompt scattered across other sections or missing.
EXAMPLES_PER_KEY = 5
EXAMPLES_SEED = 12345


def write_session_log(session_id, case_file, call_records, scope_flags, out_dir=LOGS_DIR):
    """Two files per judged session, sharing one prefix:

        logs/<session_id>.prompt.txt   what was sent to the judge
        logs/<session_id>.meta.json    how big it was + what came back

    The prompt file is rebuilt from the same functions the real calls use, so
    it cannot drift from what was sent. The shared context appears once
    because all calls reuse the identical system message.
    """
    os.makedirs(out_dir, exist_ok=True)

    shared = cl.build_shared_context(case_file)
    parts = [f"=== SHARED SESSION CONTEXT (system message) — {len(shared):,} chars ===\n{shared}\n"]
    for call_name in cl.CALL_ORDER:
        prompt = cl.build_symptom_prompt(call_name)
        parts.append(f"=== PROMPT: {call_name} (user message) — {len(prompt):,} chars ===\n{prompt}\n")

    prompt_path = os.path.join(out_dir, f"{session_id}.prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    meta = {
        "meta_version": META_VERSION,
        "session_id": session_id,
        "judge_model": cl.JUDGE_MODEL,
        "prompt_file": os.path.basename(prompt_path),
        "size": {
            "rendered_timeline_chars": len(cf.render_timeline(case_file)),
            "shared_context_chars": len(shared),
            **cf.timeline_stats(case_file),
        },
        "scope_flags": scope_flags,
        "calls": call_records,
    }
    meta_path = os.path.join(out_dir, f"{session_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def select_sessions(session_ids, sample, seed=SEED):
    """Draw the sessions to judge, UNIFORMLY AT RANDOM without replacement.

    The seed is recorded in results.json so the same sample can be redrawn.
    That makes the SELECTION reproducible, not the run: the judge still
    samples (TEMPERATURE is None), so the counts will not repeat exactly.
    """
    if sample is None:
        return list(session_ids)
    if sample > len(session_ids):
        print(f"  ! asked for {sample:,} sessions but only {len(session_ids):,} exist "
              f"— using all of them")
        return list(session_ids)
    return random.Random(seed).sample(list(session_ids), sample)


def run(session_ids, paths, sample=SAMPLE, seed=SEED):
    print(f"(sample = {'ALL' if sample is None else sample} sessions"
          f"{'' if sample is None else f', random, seed={seed}'})")

    client = OpenAI(base_url=cl.LITELLM_BASE_URL, timeout=60.0)
    ids = select_sessions(session_ids, sample, seed)
    total = len(ids)
    print(f"total sessions to process: {total:,}\n")

    all_keys = cl.all_result_keys()
    counts = Counter()               # key -> sessions with >=1 finding
    problems = Counter()             # key -> distinct (session, cause_prompt) pairs
    problems_real = Counter()        # same, cause_kind == "real" only — the rate numerator
    evidence_counts = Counter()      # key -> total evidence references returned
    cause_kind_counts = Counter()    # "real"/"system"/"out_of_range" -> findings
    category_counts = Counter()      # call_name -> sessions with >=1 finding anywhere in that call
    call_successes = Counter()       # call_name -> successful API calls (denominator)
    call_failures = Counter()        # call_name -> API calls that never produced a usable result
    n_ok = n_skipped = n_empty = 0
    total_user_prompts = 0           # REAL prompts — denominator for the per-prompt rates
    total_system_events = 0          # harness wrappers filtered out of that denominator

    # Example material, kept whole and sampled down at the end — that is what
    # makes the draw uniform over the run instead of biased toward whatever
    # came first.
    #
    # Two structures, because what is sampled and what is shown differ.
    # `example_pairs` is drawn FROM: per subcategory key, the distinct
    # (session_id, cause_prompt) problems it was seen on. `problem_findings`
    # is then EXPANDED: keyed by (call_name, session_id, cause_prompt), every
    # finding of that CATEGORY on that prompt, whatever subcategory it came
    # under. The call name in that key makes the category boundary structural
    # — a specification finding is never in a verification bucket at all,
    # rather than skipped at render time.
    example_pairs = {key: [] for key in all_keys}
    seen_example_pairs = {key: set() for key in all_keys}
    problem_findings = {}

    timing = {
        "download_total_s": 0.0,
        "parse_total_s": 0.0,
        "call_total_s_by_call": Counter(),
        "call_count_by_call": Counter(),
        "prompt_chars_total_by_call": Counter(),
    }

    run_start = time.perf_counter()
    debug_steps = 3

    for i, session_id in enumerate(ids, 1):
        show_debug = i <= debug_steps
        if i == 1 or i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] processing {session_id} "
                  f"(ok={n_ok} skipped={n_skipped})", flush=True)
        # Building the case file and judging it are guarded SEPARATELY. A
        # session that fails to download never entered the run (n_skipped);
        # one whose 2nd judge call blows up is already in n_ok and in the 1st
        # call's denominator, so counting it again would skew every per-call
        # percentage.
        try:
            case_file, download_s, parse_s = cf.build_case_file_with_timing(session_id, paths)
        except Exception as e:
            n_skipped += 1
            if show_debug:
                print(f"    -> could not build case file: {e}", flush=True)
            continue

        timing["download_total_s"] += download_s
        timing["parse_total_s"] += parse_s
        if show_debug:
            print(f"    -> download {download_s:.2f}s, parse {parse_s:.2f}s", flush=True)

        # "No user messages" means no REAL ones. A session of nothing but
        # slash commands has no human request to judge, and would add zero to
        # the denominator while still being able to add to the numerator.
        if not case_file["user_messages"]:
            n_empty += 1
            if show_debug:
                print(f"    -> no real user prompts, skipping", flush=True)
            continue
        n_ok += 1
        total_user_prompts += len(case_file["user_messages"])
        total_system_events += case_file["system_events"]

        scope_flags = cf.compute_scope_flags(case_file)
        for scope_name, is_flagged in scope_flags.items():
            if is_flagged:
                counts[scope_name] += 1

        call_records = {}
        for call_name in cl.CALL_ORDER:
            if show_debug:
                print(f"    -> judging {call_name}...", flush=True)
            try:
                result, meta = cl.judge_one_call(client, case_file, call_name)
            except Exception as e:
                call_failures[call_name] += 1
                cl.DIAGNOSTICS[f"judge_one_call raised: {type(e).__name__}"] += 1
                call_records[call_name] = {
                    "ok": False, "error": f"{type(e).__name__}: {e}",
                    "elapsed_seconds": None, "prompt_chars": None,
                    "reasoning": "", "findings": [],
                }
                if show_debug:
                    print(f"       raised: {e}", flush=True)
                continue

            timing["call_total_s_by_call"][call_name] += meta["elapsed_seconds"]
            timing["call_count_by_call"][call_name] += 1
            timing["prompt_chars_total_by_call"][call_name] += meta["prompt_chars"]

            if show_debug:
                print(f"       {meta['elapsed_seconds']:.2f}s, "
                      f"{meta['prompt_chars']:,} chars", flush=True)

            if not result or "error" in result:
                call_failures[call_name] += 1
                call_records[call_name] = {
                    "ok": False,
                    "error": (result or {}).get("error", "no result returned"),
                    "elapsed_seconds": meta["elapsed_seconds"],
                    "prompt_chars": meta["prompt_chars"],
                    "reasoning": "", "findings": [],
                }
                if show_debug and result:
                    print(f"       error: {result.get('error')}", flush=True)
                continue

            call_successes[call_name] += 1

            call_records[call_name] = {
                "ok": True, "error": None,
                "elapsed_seconds": meta["elapsed_seconds"],
                "prompt_chars": meta["prompt_chars"],
                "reasoning": result.get("reasoning", ""),
                "findings": list(result.get("findings") or []),
            }

            findings = result.get("findings", []) or []
            seen_keys_this_session = set()
            seen_problems_this_session = set()   # (key, cause_prompt) already counted
            for finding in findings:
                subcat = finding.get("subcategory", call_name)
                key = f"{call_name}:{subcat}" if subcat != call_name else call_name
                evidence = list(finding.get("evidence") or [])
                cause = finding.get("cause_prompt")
                cause_kind = finding.get("cause_kind", "real")
                cause_kind_counts[cause_kind] += 1

                if key not in seen_keys_this_session:
                    seen_keys_this_session.add(key)
                    counts[key] += 1

                # The collapsing step the judge was deliberately NOT asked to
                # do (see LOCATION_RULE). Two findings of the same subcategory
                # on one prompt are two confirmations of ONE problem, so the
                # per-prompt rate stays a rate only if they are deduplicated
                # here. Counted at BOTH levels: the category key collapses
                # across subcategories, the subcategory key does not.
                # `{key, call_name}` is one element for single symptoms.
                for level_key in {key, call_name}:
                    problem_key = (level_key, cause)
                    if problem_key not in seen_problems_this_session:
                        seen_problems_this_session.add(problem_key)
                        problems[level_key] += 1
                        # Same dedup for the rate numerator, real causes only.
                        # Inside the `if` so the two counters stay comparable:
                        # their difference IS the attribution-miss count.
                        if cause_kind == "real":
                            problems_real[level_key] += 1
                    evidence_counts[level_key] += len(evidence)

                # The cause is part of a dict key, which is safe because
                # classify._keep_verbatim guarantees a hashable scalar even
                # for a non-numeric answer — a verbatim '4.2' groups with
                # itself.
                problem_findings.setdefault((call_name, session_id, cause), []).append({
                    "subcategory": subcat,
                    "cause_kind": cause_kind,
                    "evidence": evidence,
                })
                # setdefault, not [key]: an unexpected subcategory must
                # degrade into an extra bucket rather than a KeyError that
                # takes the rest of the session down.
                pair = (session_id, cause)
                if pair not in seen_example_pairs.setdefault(key, set()):
                    seen_example_pairs[key].add(pair)
                    example_pairs.setdefault(key, []).append(pair)
            if findings:
                category_counts[call_name] += 1

        # Per session, not at the end of the run, so a crashed run still
        # leaves an audit trail for everything it got through.
        if WRITE_SESSION_LOGS:
            write_session_log(session_id, case_file, call_records, scope_flags)

    total_elapsed = time.perf_counter() - run_start

    # Draw uniformly from everything found, then order by session. A fresh
    # Random per key, seeded by the key, keeps the draw stable across reruns.
    def _pair_order(pair):
        """Sort key that survives a non-numeric cause: nothing is repaired
        upstream, so cause_prompt can be a string, and those sort after the
        numbers rather than raising TypeError. Not cosmetic — the list must be
        deterministically ordered BEFORE random.sample sees it, or the same
        seed picks different examples between runs."""
        session_id, cause = pair
        if isinstance(cause, int) and not isinstance(cause, bool):
            return (session_id, 0, cause, "")
        return (session_id, 1, 0, str(cause))

    evidence_samples = {}
    for key, pairs in example_pairs.items():
        # Everything before the first ":" is the call name; a single symptom's
        # key IS its call name.
        call_name = key.split(":", 1)[0]
        ordered = sorted(pairs, key=_pair_order)
        if len(ordered) <= EXAMPLES_PER_KEY:
            picked = ordered
        else:
            picked = random.Random(f"{EXAMPLES_SEED}:{key}").sample(ordered, EXAMPLES_PER_KEY)
        samples = []
        for session_id, cause in sorted(picked, key=_pair_order):
            group = problem_findings.get((call_name, session_id, cause), [])
            samples.append({
                "session_id": session_id,
                "cause_prompt": cause,
                # cause_kind is a function of (session, cause_prompt), so the
                # whole group carries the same one.
                "cause_kind": group[0]["cause_kind"] if group else None,
                "findings": [
                    {"subcategory": f["subcategory"], "evidence": f["evidence"]}
                    for f in group
                ],
            })
        evidence_samples[key] = samples

    print(f"\nsessions with usable data: {n_ok} | no user messages: {n_empty} "
          f"| failed to build: {n_skipped}")
    print(f"total wall-clock time: {total_elapsed:.1f}s")
    if any(call_failures.values()):
        print("failed judge calls: " + ", ".join(
            f"{name}={call_failures[name]}" for name in cl.CALL_ORDER if call_failures[name]))
    if cl.DIAGNOSTICS:
        print("diagnostics (why calls fell back or failed):")
        for reason, n in cl.DIAGNOSTICS.most_common():
            print(f"  {n:>4} x {reason}")
    print()

    pdenom = total_user_prompts or 1
    n_text_events = total_user_prompts + total_system_events
    print(f"(real user prompts across judged sessions: {total_user_prompts:,}; "
          f"{total_system_events:,} of {n_text_events:,} text user events "
          f"({100*total_system_events/max(n_text_events,1):.1f}%) were harness "
          "wrappers and are not in that denominator)")
    print(f"{'check':40s} {'sessions':>9s} {'%sess':>6s} {'probs':>7s} "
          f"{'evid':>6s} {'%prompt':>8s}")
    for call_name in cl.CALL_ORDER:
        denom = call_successes[call_name] or 1
        # The category row is not the sum of the rows under it: a prompt
        # flagged three ways is three subcategory problems but one category
        # problem. The percentage uses problems_real, not problems.
        print(f"{call_name:40s} {category_counts[call_name]:>9,} "
              f"{100*category_counts[call_name]/denom:>5.0f}% "
              f"{problems[call_name]:>7,} {evidence_counts[call_name]:>6,} "
              f"{100*problems_real[call_name]/pdenom:>7.1f}%")
        info = cl.CALLS[call_name]
        if info["kind"] != "category":
            continue
        for subcat in info["subcategories"]:
            key = f"{call_name}:{subcat}"
            print(f"  - {subcat:36s} {counts[key]:>9,} "
                  f"{100*counts[key]/denom:>5.0f}% {problems[key]:>7,} "
                  f"{evidence_counts[key]:>6,} {100*problems_real[key]/pdenom:>7.1f}%")
    for scope_name in ("scope_files_too_many", "scope_turns_too_long"):
        denom = n_ok or 1
        print(f"{scope_name:40s} {counts[scope_name]:>9,} {100*counts[scope_name]/denom:>5.0f}%")

    n_findings = sum(cause_kind_counts.values())
    n_missed = n_findings - cause_kind_counts["real"]
    if n_findings:
        print(f"\nattribution: {n_missed:,} of {n_findings:,} findings "
              f"({100*n_missed/n_findings:.1f}%) named a cause that is not a real "
              f"user prompt — {cause_kind_counts['system']:,} on a [SYSTEM] block, "
              f"{cause_kind_counts['out_of_range']:,} outside the session. Kept in "
              "'probs', excluded from '%prompt'.")

    print("\ntiming breakdown:")
    print(f"  total download time: {timing['download_total_s']:.1f}s "
          f"({timing['download_total_s']/max(n_ok,1):.2f}s/session avg)")
    print(f"  total parse time:    {timing['parse_total_s']:.1f}s "
          f"({timing['parse_total_s']/max(n_ok,1):.2f}s/session avg)")
    for call_name in cl.CALL_ORDER:
        c = timing["call_count_by_call"][call_name] or 1
        avg_time = timing["call_total_s_by_call"][call_name] / c
        avg_chars = timing["prompt_chars_total_by_call"][call_name] / c
        print(f"  {call_name:26s} avg call time: {avg_time:5.2f}s | avg prompt size: {avg_chars:,.0f} chars")

    return {
        "counts": dict(counts),
        # Distinct (category-or-subcategory, cause_prompt) pairs. Keys cover
        # BOTH levels: a bare call name is the category row, "call:subcat" a
        # subcategory row, and the second does not sum to the first. Unlike
        # "% of sessions", this weighs a 200-prompt session differently from a
        # 2-prompt one.
        "problems": dict(problems),
        # The rate numerator. Separate from "problems" so the pair reads as
        # "how much was found" vs. "how much could be placed on a real
        # prompt"; the difference is the attribution-miss count.
        "problems_real": dict(problems_real),
        "cause_kind_counts": dict(cause_kind_counts),
        "evidence_counts": dict(evidence_counts),
        "total_user_prompts": total_user_prompts,
        # Not a denominator — what the denominator excludes. Recorded so the
        # report can state how much of the old count was harness noise.
        "total_system_events": total_system_events,
        "examples_per_key": EXAMPLES_PER_KEY,
        "examples_seed": EXAMPLES_SEED,
        "category_counts": dict(category_counts),
        "call_successes": dict(call_successes),
        "call_failures": dict(call_failures),
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_empty": n_empty,
        # What this run covered, so report.py never hardcodes its scope.
        "sample_requested": sample,
        "n_candidates": len(session_ids),
        # report.py keys off "sampling": a results.json without it predates
        # random sampling and was a first-N-in-dataset-order spot check.
        "sampling": "all" if sample is None else "random",
        "sample_seed": None if sample is None else seed,
        # The exact draw, so it can be audited or re-judged without trusting
        # that the seed and the dataset revision still line up.
        "session_ids": list(ids),
        "judge_model": cl.JUDGE_MODEL,
        # Recorded here so the report can state the caps and cut-offs without
        # importing case_file.py and its dataset dependencies.
        "rendering_caps": {
            "thinking_chars_per_block": cf.MAX_THINKING_CHARS_PER_BLOCK,
            "assistant_text_chars": cf.MAX_ASSISTANT_TEXT_CHARS,
            "tool_result_chars": cf.MAX_TOOL_RESULT_CHARS,
            "user_message_chars": cf.MAX_USER_MESSAGE_CHARS,
        },
        "scope_thresholds": {
            "files_too_many": cf.SCOPE_FILES_TOO_MANY_THRESHOLD,
            "turns_too_long": cf.SCOPE_TURNS_TOO_LONG_THRESHOLD,
        },
        "diagnostics": dict(cl.DIAGNOSTICS),
        "evidence_samples": evidence_samples,
        "timing": {
            "download_total_s": timing["download_total_s"],
            "parse_total_s": timing["parse_total_s"],
            "call_total_s_by_call": dict(timing["call_total_s_by_call"]),
            "call_count_by_call": dict(timing["call_count_by_call"]),
            "prompt_chars_total_by_call": dict(timing["prompt_chars_total_by_call"]),
            "total_wall_clock_s": total_elapsed,
        },
    }


def save_results(results, path=RESULTS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Judge a random sample of SWE-Chat Claude Code sessions "
                    "for vibe-fixing symptoms.",
        epilog="examples:\n"
               "  python run_experiment.py 25\n"
               "  python run_experiment.py 25 --seed 7\n"
               "  python run_experiment.py --all\n"
               "  python run_experiment.py 25 --results results-25.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "n", nargs="?", type=int, default=SAMPLE,
        help=f"how many sessions to draw at random (default: {SAMPLE})",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"RNG seed for the draw, so a sample can be reproduced (default: {SEED})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="judge every parseable session instead of a sample (ignores n)",
    )
    parser.add_argument(
        "--results", default=RESULTS_PATH,
        help=f"where to write the aggregate results (default: {RESULTS_PATH})",
    )
    args = parser.parse_args(argv)
    if not args.all and args.n is not None and args.n < 1:
        parser.error("n must be at least 1")
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Loading dataset tables from HuggingFace (this can take a while, no progress bar)...")
    sessions, logs = cf.load_tables()
    print(f"Loaded {len(sessions):,} sessions and {len(logs):,} session logs.")

    all_ids = cf.claude_code_session_ids(sessions)
    paths = cf.path_map(logs)
    print(f"Filtered to {len(all_ids):,} Claude Code sessions.\n")

    results = run(all_ids, paths, sample=None if args.all else args.n, seed=args.seed)
    save_results(results, args.results)