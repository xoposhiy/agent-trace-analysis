"""
merge_results.py — combines two or more results.json files (from separate
run_experiment.py runs, e.g. different --seed batches covering different
sessions) into one merged results.json, so report.py and benchmark.py can
treat the combined dataset as a single run.

This does NOT just paste one file's content over another — every counter
gets summed key-by-key, overlapping sessions get deduplicated (not double-
counted), and anything that should be identical across runs (like the
judge model) is checked, not silently picked.

Usage:
    python3 merge_results.py results.json results_60b.json --out results_merged.json
    python3 merge_results.py a.json b.json c.json --out combined.json   # 3+ also works

MERGE RULES:
    - Per-key counters (counts, problems, problems_real, cause_kind_counts,
      evidence_counts, category_counts, call_successes, call_failures,
      diagnostics) are SUMMED key-by-key across all input files.
    - Scalar totals (n_ok, n_skipped, n_empty, total_user_prompts,
      total_system_events) are SUMMED.
    - session_ids are UNIONED. If the same session appears in more than one
      input file (overlapping random samples), it's counted only ONCE in
      the merged list, and a warning is printed — this usually means two
      runs used seeds that happened to overlap.
    - evidence_samples are CONCATENATED per key (kept in full, not re-capped
      to the original per-key limit — more real examples from a bigger
      combined run is strictly more useful for manual review, not a
      problem to trim back down).
    - timing fields are SUMMED (so downstream averages can be recomputed
      correctly from the combined totals).
    - Fields that SHOULD be identical across a real merge (judge_model,
      rendering_caps, scope_thresholds, examples_per_key) are compared; a
      mismatch prints a loud warning rather than silently keeping one
      value, since it usually means the code or config changed between the
      two runs and the merge may not be a fair apples-to-apples combination.
    - sample_requested / n_candidates / sampling / sample_seed / examples_seed
      become LISTS (one entry per source file) — there's no single
      meaningful value for these once you've combined differently-seeded
      runs, so this keeps the real per-run history visible instead of
      picking one arbitrarily.
"""

import argparse
import json
from collections import Counter

SUM_DICT_FIELDS = [
    "counts", "problems", "problems_real", "cause_kind_counts",
    "evidence_counts", "category_counts", "call_successes", "call_failures",
    "diagnostics",
]
SUM_SCALAR_FIELDS = [
    "n_ok", "n_skipped", "n_empty", "total_user_prompts", "total_system_events",
]
MUST_MATCH_FIELDS = [
    "judge_model", "rendering_caps", "scope_thresholds", "examples_per_key",
    "n_candidates",  # total dataset size — fixed, not a per-run sampling choice
]
PER_FILE_LIST_FIELDS = [
    "sample_requested", "sampling", "sample_seed", "examples_seed",
]


def _merge_numeric_dict(dicts):
    out = Counter()
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] += v
    return dict(out)


def _merge_timing(timings):
    return {
        "download_total_s": sum(t.get("download_total_s", 0) or 0 for t in timings),
        "parse_total_s": sum(t.get("parse_total_s", 0) or 0 for t in timings),
        "total_wall_clock_s": sum(t.get("total_wall_clock_s", 0) or 0 for t in timings),
        "call_total_s_by_call": _merge_numeric_dict([t.get("call_total_s_by_call", {}) for t in timings]),
        "call_count_by_call": _merge_numeric_dict([t.get("call_count_by_call", {}) for t in timings]),
        "prompt_chars_total_by_call": _merge_numeric_dict([t.get("prompt_chars_total_by_call", {}) for t in timings]),
    }


def _merge_evidence_samples(samples_list):
    keys = set()
    for s in samples_list:
        keys.update((s or {}).keys())
    out = {}
    for k in keys:
        combined = []
        for s in samples_list:
            combined.extend((s or {}).get(k, []) or [])
        out[k] = combined
    return out


def merge_results(paths, out_path):
    results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))

    merged = {}

    for field in SUM_DICT_FIELDS:
        merged[field] = _merge_numeric_dict([r.get(field, {}) for r in results])

    for field in SUM_SCALAR_FIELDS:
        merged[field] = sum(r.get(field, 0) or 0 for r in results)

    all_ids = []
    seen = set()
    dup_count = 0
    for r in results:
        for sid in r.get("session_ids", []) or []:
            if sid in seen:
                dup_count += 1
                continue
            seen.add(sid)
            all_ids.append(sid)
    merged["session_ids"] = all_ids
    if dup_count:
        print(f"WARNING: {dup_count} session id(s) appeared in more than one input file "
              f"(overlapping samples) — counted only once in the merge, not double-counted.")

    merged["evidence_samples"] = _merge_evidence_samples([r.get("evidence_samples", {}) for r in results])
    merged["timing"] = _merge_timing([r.get("timing", {}) for r in results])

    for field in MUST_MATCH_FIELDS:
        values = [r.get(field) for r in results]
        if any(v != values[0] for v in values[1:]):
            print(f"WARNING: '{field}' differs across input files: {values} — keeping the "
                  f"first file's value. This usually means the code/config changed between "
                  f"runs; double-check the merge is a fair apples-to-apples combination.")
        merged[field] = values[0]

    for field in PER_FILE_LIST_FIELDS:
        merged[field] = [r.get(field) for r in results]

    merged["merged_from"] = paths
    merged["n_source_files"] = len(paths)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"\nWrote {out_path}, merged from {len(paths)} file(s): {', '.join(paths)}")
    print(f"combined n_ok: {merged.get('n_ok')} | combined session_ids: {len(merged['session_ids'])}")
    print("\nper-key counts (combined):")
    for key, count in sorted(merged["counts"].items()):
        print(f"  {key:55s} {count}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="two or more results.json files to merge")
    parser.add_argument("--out", default="results_merged.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("need at least 2 input files to merge")
    merge_results(args.inputs, args.out)