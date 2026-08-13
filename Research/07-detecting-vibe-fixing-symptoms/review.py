"""
review.py — quick, NON-blind review of what the judge found in a random
sample of already-judged sessions.

Unlike benchmark.py (which deliberately hides the model's answer so you can
judge independently, feeding a formal precision/recall score), this SHOWS
the model's reasoning and findings directly, for a fast plausibility check —
"does this look right at a glance?" — not a measured accuracy number.

Usage:
    python3 review.py 10 --logs-dir logs --seed 1
    python3 review.py 10 --logs-dir logs --seed 1 --out review.txt

For each of the N sampled sessions, prints:
    - the session id and where its full prompt/transcript lives, so you can
      open it directly if a finding looks questionable
    - basic size stats (turns, tool calls, etc.)
    - for each of the 3 calls: the model's reasoning, and every finding with
      its subcategory, cause_prompt, and evidence notes

Sampling uses the same uniform-random approach as benchmark.py (same seed +
n gives the same sessions from the same logs dir, for comparability).
"""

import argparse
import glob
import json
import os
import random

REASONING_PREVIEW_CHARS = 500
EVIDENCE_NOTE_CHARS = 200


def _session_ids_from_logs(logs_dir):
    metas = sorted(glob.glob(os.path.join(logs_dir, "*.meta.json")))
    return [os.path.basename(m)[: -len(".meta.json")] for m in metas]


def render_session_review(session_id, meta, logs_dir):
    lines = []
    lines.append("=" * 78)
    lines.append(f"SESSION: {session_id}")
    lines.append(f"prompt file: {os.path.join(logs_dir, meta.get('prompt_file', '?'))}")
    size = meta.get("size", {})
    lines.append(
        f"size: {size.get('rendered_timeline_chars', '?'):,} chars"
        if isinstance(size.get("rendered_timeline_chars"), int) else
        f"size: {size.get('rendered_timeline_chars', '?')} chars"
    )
    lines.append(
        f"  {size.get('user_prompts_real', '?')} real prompts, "
        f"{size.get('system_events', '?')} system/wrapper events, "
        f"{size.get('assistant_turns', '?')} assistant turns, "
        f"{size.get('tool_calls', '?')} tool calls, "
        f"{size.get('files_touched', '?')} files touched"
    )
    lines.append("=" * 78)

    for call_name, call in meta.get("calls", {}).items():
        lines.append(f"\n--- {call_name} ---")
        if not call.get("ok"):
            lines.append(f"  [call failed: {call.get('error')}]")
            continue

        reasoning = (call.get("reasoning") or "").strip()
        if reasoning:
            trimmed = reasoning[:REASONING_PREVIEW_CHARS]
            suffix = "..." if len(reasoning) > REASONING_PREVIEW_CHARS else ""
            lines.append(f"  reasoning: {trimmed}{suffix}")

        findings = call.get("findings") or []
        if not findings:
            lines.append("  findings: (none)")
            continue

        for finding in findings:
            lines.append(
                f"  FINDING: {finding.get('subcategory')}  "
                f"(cause_prompt={finding.get('cause_prompt')}, "
                f"cause_kind={finding.get('cause_kind')})"
            )
            for ev in finding.get("evidence", []) or []:
                note = (ev.get("note") or "").strip()[:EVIDENCE_NOTE_CHARS]
                lines.append(f"      @ prompt {ev.get('prompt')}.{ev.get('step')}: {note}")

    return "\n".join(lines)


def review(logs_dir, n, seed, out_path=None):
    ids = _session_ids_from_logs(logs_dir)
    if not ids:
        raise SystemExit(f"no *.meta.json files found in '{logs_dir}'")
    if n > len(ids):
        print(f"  ! asked for {n} sessions but only {len(ids)} are available — using all of them")
        n = len(ids)
    picked = sorted(random.Random(seed).sample(ids, n))

    blocks = []
    for session_id in picked:
        meta_path = os.path.join(logs_dir, f"{session_id}.meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        blocks.append(render_session_review(session_id, meta, logs_dir))

    output = "\n\n".join(blocks)
    print(output)
    print(f"\n\n({len(picked)} sessions reviewed, seed={seed})")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"(also saved to {out_path})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("n", type=int, nargs="?", default=10, help="how many sessions to sample (default 10)")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="also save the review text to this file")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    review(args.logs_dir, args.n, args.seed, args.out)