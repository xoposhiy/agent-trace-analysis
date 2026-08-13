"""
review_pair.py — like review.py, but instead of a random sample, shows ONLY
the sessions where a SPECIFIC PAIR of subcategories both occurred. Useful
for checking a specific hypothesis, e.g. "when not-tested and
repetitive-bug-fixes both show up, does the evidence actually support a
causal link, or are they just coincidentally in the same session?"

Requires review.py in the same folder (reuses its session-rendering so the
two tools show findings in an identical format).

Usage:
    # every session where both occurred anywhere:
    python3 review_pair.py not_enough_verification:not-tested \\
        not_enough_verification:repetitive-bug-fixes --logs-dir logs

    # narrower: only sessions where a repetitive-bug-fixes finding shares
    # the EXACT same cause_prompt as a not-tested finding (the strongest
    # form of "these are actually the same underlying problem"):
    python3 review_pair.py not_enough_verification:not-tested \\
        not_enough_verification:repetitive-bug-fixes --logs-dir logs --same-cause-only

    # if there are many matches, review a random subset instead of all:
    python3 review_pair.py ... --sample 5 --seed 1
"""

import argparse
import glob
import json
import os
import random

from review import render_session_review


def _session_ids_from_logs(logs_dir):
    metas = sorted(glob.glob(os.path.join(logs_dir, "*.meta.json")))
    return [os.path.basename(m)[: -len(".meta.json")] for m in metas]


def _findings_for_key(meta, key):
    """key looks like 'call_name:subcategory', or just 'call_name' for a
    single-symptom call (no_visual_reference) with no subcategory suffix."""
    call_name, _, subcat = key.partition(":")
    if not subcat:
        subcat = call_name
    call = meta.get("calls", {}).get(call_name, {})
    if not call.get("ok"):
        return []
    return [f for f in call.get("findings", []) or [] if f.get("subcategory") == subcat]


def matches_pair(meta, key_a, key_b, same_cause_only):
    finds_a = _findings_for_key(meta, key_a)
    finds_b = _findings_for_key(meta, key_b)
    if not finds_a or not finds_b:
        return False
    if not same_cause_only:
        return True
    causes_a = {f.get("cause_prompt") for f in finds_a}
    return any(f.get("cause_prompt") in causes_a for f in finds_b)


def find_matching_sessions(logs_dir, key_a, key_b, same_cause_only):
    matches = []
    for session_id in _session_ids_from_logs(logs_dir):
        meta_path = os.path.join(logs_dir, f"{session_id}.meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if matches_pair(meta, key_a, key_b, same_cause_only):
            matches.append((session_id, meta))
    return matches


def main(args):
    matches = find_matching_sessions(args.logs_dir, args.key_a, args.key_b, args.same_cause_only)
    condition = f"BOTH '{args.key_a}' and '{args.key_b}'"
    if args.same_cause_only:
        condition += " (sharing the same cause_prompt)"
    print(f"{len(matches)} session(s) have {condition}.\n")

    if not matches:
        return

    chosen = matches
    if args.sample and args.sample < len(matches):
        chosen = random.Random(args.seed).sample(matches, args.sample)
        chosen.sort(key=lambda m: m[0])
        print(f"showing a random sample of {args.sample} of them (seed={args.seed}):\n")
    else:
        chosen.sort(key=lambda m: m[0])

    blocks = [render_session_review(sid, meta, args.logs_dir) for sid, meta in chosen]
    output = "\n\n".join(blocks)
    print(output)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n\n(also saved to {args.out})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("key_a", help="e.g. not_enough_verification:not-tested")
    parser.add_argument("key_b", help="e.g. not_enough_verification:repetitive-bug-fixes")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--same-cause-only", action="store_true",
                         help="only match if a key_b finding shares the exact cause_prompt as a key_a finding")
    parser.add_argument("--sample", type=int, default=None,
                         help="if there are many matches, review a random N of them instead of all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="also save the review text to this file")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)