"""
run_experiment.py — the main loop: loads the dataset, runs every session
through classify.py's per-symptom judge, aggregates counts + timing/size
stats, and saves everything to a JSON file (RESULTS_PATH) so report.py can
build the markdown report later WITHOUT needing to rerun the experiment or
touch the API/dataset again.

Run directly:

    python run_experiment.py

SAMPLE below controls how many sessions to process (None = all parseable
Claude Code sessions in the dataset). Use a small SAMPLE first — this makes
5 LLM calls per session, so cost/time scale directly with it.
"""

import json
import time
from collections import Counter

from openai import OpenAI

import case_file as cf
import classify as cl

SAMPLE = 25   # None = run on every parseable session
RESULTS_PATH = "results.json"


def run(session_ids, paths, sample=SAMPLE):
    print(f"(sample = {'ALL' if sample is None else sample} sessions)")

    client = OpenAI(base_url=cl.LITELLM_BASE_URL, timeout=45.0)
    ids = session_ids if sample is None else session_ids[:sample]
    total = len(ids)
    print(f"total sessions to process: {total:,}\n")

    counts = Counter()
    call_successes = Counter()   # per-symptom denominator: sessions where that symptom's call succeeded
    n_ok = n_skipped = 0
    evidence_samples = {name: [] for name in cl.SYMPTOM_ORDER}

    # Timing/size instrumentation — so "find the slowest part" has an actual
    # answer instead of a guess. Tracked per stage and per symptom.
    timing = {
        "download_total_s": 0.0,
        "parse_total_s": 0.0,
        "call_total_s_by_symptom": Counter(),
        "call_count_by_symptom": Counter(),
        "prompt_chars_total_by_symptom": Counter(),
    }

    run_start = time.perf_counter()
    debug_steps = 3

    for i, session_id in enumerate(ids, 1):
        show_debug = i <= debug_steps
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] processing {session_id} "
                  f"(ok={n_ok} skipped={n_skipped})", flush=True)
        try:
            case_file, download_s, parse_s = cf.build_case_file_with_timing(session_id, paths)
            timing["download_total_s"] += download_s
            timing["parse_total_s"] += parse_s
            if show_debug:
                print(f"    -> download {download_s:.2f}s, parse {parse_s:.2f}s", flush=True)

            if not case_file["user_messages"]:
                if show_debug:
                    print(f"    -> no user messages, skipping", flush=True)
                continue
            n_ok += 1

            scope_flags = cf.compute_scope_flags(case_file)
            for scope_name, is_flagged in scope_flags.items():
                if is_flagged:
                    counts[scope_name] += 1

            for symptom_name in cl.SYMPTOM_ORDER:
                if show_debug:
                    print(f"    -> judging {symptom_name}...", flush=True)
                result, meta = cl.judge_one_symptom(client, case_file, symptom_name)
                timing["call_total_s_by_symptom"][symptom_name] += meta["elapsed_seconds"]
                timing["call_count_by_symptom"][symptom_name] += 1
                timing["prompt_chars_total_by_symptom"][symptom_name] += meta["prompt_chars"]

                if show_debug:
                    print(f"       {meta['elapsed_seconds']:.2f}s, "
                          f"{meta['prompt_chars']:,} chars", flush=True)

                if not result or "error" in result:
                    if show_debug and result:
                        print(f"       error: {result.get('error')}", flush=True)
                    continue

                result = cl.apply_post_filter(symptom_name, result, case_file)
                call_successes[symptom_name] += 1

                if result.get("present"):
                    counts[symptom_name] += 1
                    if len(evidence_samples[symptom_name]) < 5:
                        evidence_samples[symptom_name].append({
                            "session_id": session_id,
                            "location": result.get("location", "n/a"),
                            "evidence": result.get("evidence", ""),
                        })

        except Exception as e:
            n_skipped += 1
            if show_debug:
                print(f"    -> exception: {e}", flush=True)

    total_elapsed = time.perf_counter() - run_start

    print(f"\nsessions with usable data: {n_ok} | skipped: {n_skipped}")
    print(f"total wall-clock time: {total_elapsed:.1f}s\n")

    print(f"{'symptom':28s} {'count':>7s} {'% of judged sessions':>22s}")
    scope_names = ["scope_files_too_many", "scope_turns_too_long"]
    for name in cl.SYMPTOM_ORDER + scope_names:
        denom = call_successes[name] if name not in scope_names else n_ok
        denom = denom or 1
        print(f"{name:28s} {counts[name]:>7,} {100*counts[name]/denom:>21.0f}%")

    print("\ntiming breakdown (find the slowest part):")
    print(f"  total download time: {timing['download_total_s']:.1f}s "
          f"({timing['download_total_s']/max(n_ok,1):.2f}s/session avg)")
    print(f"  total parse time:    {timing['parse_total_s']:.1f}s "
          f"({timing['parse_total_s']/max(n_ok,1):.2f}s/session avg)")
    for name in cl.SYMPTOM_ORDER:
        c = timing["call_count_by_symptom"][name] or 1
        avg_time = timing["call_total_s_by_symptom"][name] / c
        avg_chars = timing["prompt_chars_total_by_symptom"][name] / c
        print(f"  {name:26s} avg call time: {avg_time:5.2f}s | avg prompt size: {avg_chars:,.0f} chars")

    return {
        "counts": dict(counts),
        "call_successes": dict(call_successes),
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "evidence_samples": evidence_samples,
        "timing": {
            "download_total_s": timing["download_total_s"],
            "parse_total_s": timing["parse_total_s"],
            "call_total_s_by_symptom": dict(timing["call_total_s_by_symptom"]),
            "call_count_by_symptom": dict(timing["call_count_by_symptom"]),
            "prompt_chars_total_by_symptom": dict(timing["prompt_chars_total_by_symptom"]),
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