"""
run_experiment.py — the main loop: loads the dataset, runs every session
through classify.py's 3 judge calls (not_enough_verification,
not_enough_specification, no_visual_reference), aggregates results, and
saves everything to RESULTS_PATH so report.py can build the report later
without rerunning the experiment.

Run directly:

    python run_experiment.py

DESIGN NOTES (this revision):
    - Only 3 LLM calls per session now (down from 5), since the two grouped
      categories each cover multiple subcategories in one call.
    - Tracks TWO different numbers per (call, subcategory) key:
        * counts[key] — how many SESSIONS had at least one occurrence
          (drives the "% of judged sessions" column, same as before).
        * occurrence_counts[key] — the TOTAL number of occurrences across
          all sessions (a session with the same subcategory 3 times counts
          3 here, but only 1 in counts[key]). This is the new column asked
          for alongside "% of judged sessions".
    - Failure accounting is explicit. A session that can't be built is
      n_skipped; a session with no user messages is n_empty; a judge call
      that never returned a usable result is call_failures[call], counted
      per call rather than throwing away the rest of the session. Every
      fallback/failure reason lands in classify.DIAGNOSTICS. All of it is
      saved to results.json so the report can be honest about how much of
      the run actually succeeded.
    - results.json also records what this run covered (sample_requested,
      n_candidates) and the rendering caps the judge saw, so report.py
      never has to hardcode claims about the run's scope.
"""

import json
import time
from collections import Counter

from openai import OpenAI

import case_file as cf
import classify as cl

SAMPLE = 15   # None = run on every parseable session
RESULTS_PATH = "results.json"


def run(session_ids, paths, sample=SAMPLE):
    print(f"(sample = {'ALL' if sample is None else sample} sessions)")

    client = OpenAI(base_url=cl.LITELLM_BASE_URL, timeout=60.0)
    ids = session_ids if sample is None else session_ids[:sample]
    total = len(ids)
    print(f"total sessions to process: {total:,}\n")

    all_keys = cl.all_result_keys()
    counts = Counter()               # key -> sessions with >=1 occurrence
    occurrence_counts = Counter()    # key -> total occurrences across all sessions
    category_counts = Counter()      # call_name -> sessions with >=1 finding anywhere in that call
    call_successes = Counter()       # call_name -> successful API calls (denominator)
    call_failures = Counter()        # call_name -> API calls that never produced a usable result
    n_ok = n_skipped = n_empty = 0
    evidence_samples = {key: [] for key in all_keys}

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
        # session that fails to download/parse never entered the run at all
        # (n_skipped); a session whose 2nd judge call blows up has already been
        # counted in n_ok and in the 1st call's denominator, so it must not also
        # land in n_skipped — that would double-count it and silently skew every
        # per-call percentage.
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

        if not case_file["user_messages"]:
            n_empty += 1
            if show_debug:
                print(f"    -> no user messages, skipping", flush=True)
            continue
        n_ok += 1

        scope_flags = cf.compute_scope_flags(case_file)
        for scope_name, is_flagged in scope_flags.items():
            if is_flagged:
                counts[scope_name] += 1

        for call_name in cl.CALL_ORDER:
            if show_debug:
                print(f"    -> judging {call_name}...", flush=True)
            try:
                result, meta = cl.judge_one_call(client, case_file, call_name)
            except Exception as e:
                call_failures[call_name] += 1
                cl.DIAGNOSTICS[f"judge_one_call raised: {type(e).__name__}"] += 1
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
                if show_debug and result:
                    print(f"       error: {result.get('error')}", flush=True)
                continue

            result = cl.apply_post_filter(call_name, result, case_file)
            call_successes[call_name] += 1

            findings = result.get("findings", []) or []
            seen_keys_this_session = set()
            for finding in findings:
                subcat = finding.get("subcategory", call_name)
                key = f"{call_name}:{subcat}" if subcat != call_name else call_name
                occurrence_counts[key] += 1
                if key not in seen_keys_this_session:
                    seen_keys_this_session.add(key)
                    counts[key] += 1
                # setdefault, not [key]: classify.py validates subcategories, but
                # an unexpected key must degrade into an extra bucket rather than
                # a KeyError that takes the rest of the session down with it.
                samples = evidence_samples.setdefault(key, [])
                if len(samples) < 5:
                    samples.append({
                        "session_id": session_id,
                        "location": finding.get("location", "n/a"),
                        "evidence": finding.get("evidence", ""),
                    })
            if findings:
                category_counts[call_name] += 1

    total_elapsed = time.perf_counter() - run_start

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

    print(f"{'check':40s} {'sessions':>9s} {'%':>6s} {'occurrences':>12s}")
    for call_name in cl.CALL_ORDER:
        denom = call_successes[call_name] or 1
        print(f"{call_name:40s} {category_counts[call_name]:>9,} "
              f"{100*category_counts[call_name]/denom:>5.0f}% {'':>12s}")
        info = cl.CALLS[call_name]
        if info["kind"] == "category":
            for subcat in info["subcategories"]:
                key = f"{call_name}:{subcat}"
                print(f"  - {subcat:36s} {counts[key]:>9,} "
                      f"{100*counts[key]/denom:>5.0f}% {occurrence_counts[key]:>12,}")
        else:
            print(f"  (single symptom, no subcategories) "
                  f"{'':>9s} {'':>6s} {occurrence_counts[call_name]:>12,}")
    for scope_name in ("scope_files_too_many", "scope_turns_too_long"):
        denom = n_ok or 1
        print(f"{scope_name:40s} {counts[scope_name]:>9,} {100*counts[scope_name]/denom:>5.0f}%")

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
        "occurrence_counts": dict(occurrence_counts),
        "category_counts": dict(category_counts),
        "call_successes": dict(call_successes),
        "call_failures": dict(call_failures),
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_empty": n_empty,
        # What this run actually covered, so report.py can describe its own
        # scope instead of hardcoding a claim that drifts from the code.
        "sample_requested": sample,
        "n_candidates": len(session_ids),
        "judge_model": cl.JUDGE_MODEL,
        # The caps the judge's view of each session was rendered under —
        # recorded here so the report can state them without importing
        # case_file.py (and its dataset dependencies).
        "rendering_caps": {
            "thinking_chars_per_block": cf.MAX_THINKING_CHARS_PER_BLOCK,
            "tool_result_chars": cf.MAX_TOOL_RESULT_CHARS,
            "user_message_chars": cf.MAX_USER_MESSAGE_CHARS,
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


if __name__ == "__main__":
    print("Loading dataset tables from HuggingFace (this can take a while, no progress bar)...")
    sessions, logs = cf.load_tables()
    print(f"Loaded {len(sessions):,} sessions and {len(logs):,} session logs.")

    all_ids = cf.claude_code_session_ids(sessions)
    paths = cf.path_map(logs)
    print(f"Filtered to {len(all_ids):,} Claude Code sessions.\n")

    results = run(all_ids, paths)
    save_results(results)