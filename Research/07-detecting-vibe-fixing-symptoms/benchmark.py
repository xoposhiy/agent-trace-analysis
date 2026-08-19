"""
benchmark.py — a small, reproducible accuracy check for the vibe-fixing
judge, built on top of the per-session logs that run_experiment.py already
writes (logs/<session_id>.prompt.txt + .meta.json).

This does NOT call the LLM again. It only reads what the judge already said
(from the .meta.json files) and compares it against a human's independent
judgment, which you fill in by hand.

WORKFLOW (session-level, original):

  1. Generate a template (picks N sessions at random, no LLM calls):
       python benchmark.py generate 20 --logs-dir logs --out benchmark.json

  2. Fill in benchmark.json by hand:
       - Open each session's .prompt.txt (path is printed in the template)
       - Read it YOURSELF, without looking at what the model said
       - For each subcategory, write true / false / null (null = "unsure,
         skip this one" — it won't count against or for the model)
       - Optionally add a one-line note explaining your call

  3. Score it:
       python benchmark.py score --template benchmark.json --logs-dir logs

     Prints, per subcategory: true positives / false positives / false
     negatives / true negatives, precision, recall, and overall agreement —
     plus a confidence-calibration breakdown (does the model's own 0-1
     confidence actually predict whether it's right?) and every specific
     disagreement, so you can go look at exactly which sessions the judge
     got wrong (in either direction).

WORKFLOW (per-finding, newer — use this one if you want a real
precision/recall/accuracy confusion matrix, swept across confidence
thresholds):

  1. Sample low-confidence findings + a few zero-finding sessions:
       python benchmark.py sample-findings --logs-dir logs --total 20 \\
           --zero-finding 5 --out benchmark_findings.json

     This picks the model's LOWEST-confidence findings first (they're the
     most likely to be wrong, so checking them is the most informative use
     of a small manual sample) — but low-confidence sampling can ONLY ever
     surface false positives, never false negatives (a missed finding has
     no confidence value to sort by in the first place). So a handful of
     sessions where the model found NOTHING are included too, specifically
     so you can check for things it silently missed.

  2. Fill in benchmark_findings.json by hand:
       - For "finding" items: open prompt_file, check the model's specific
         claim, set human_verdict to "correct" / "incorrect" / "unsure"
       - For "zero_finding_session" items: read the whole session; if you
         spot something the model should have caught, log it in
         human_missed_findings. It starts as null ("not reviewed yet",
         excluded from scoring) — set it to [] once checked and clean, or
         to a list of entries if you found something.

  3. Score it, with a confidence-threshold sweep:
       python benchmark.py score-findings --template benchmark_findings.json \\
           --logs-dir logs --thresholds 0.0,0.5,0.7,0.9

     Prints a full TP/FP/FN/TN confusion matrix (+ accuracy/precision/
     recall) AT EACH threshold, so you can see how raising the confidence
     bar trades recall for precision — "if we only trust findings the model
     was >=0.8 confident about, what does accuracy look like?" — rather
     than one flat number that hides that tradeoff.

SCOPE (original workflow only): the session-level workflow above checks
SESSION-LEVEL presence ("did this subcategory occur at least once in this
session"), not the finer-grained cause_prompt/evidence locations. The
per-finding workflow checks individual findings directly, which is the more
precise (and recommended) option going forward.
"""

import argparse
import glob
import json
import os
import random

# Mirrors classify.py's CALLS/all_result_keys(), duplicated here on purpose
# (same reasoning as report.py not importing case_file.py) — this script
# should never need network access, the OpenAI SDK, or the HF dataset just
# to generate or score a template.
ALL_KEYS = [
    "not_enough_verification:not-tested",
    "not_enough_verification:self-report",
    "not_enough_verification:ask-for-manual-testing",
    "not_enough_verification:repetitive-bug-fixes",
    "not_enough_specification:no-spec-detected",
    "not_enough_specification:repetitive-requirements-fixes",
    "not_enough_specification:self-report",
    "no_visual_reference",
]

KEY_DESCRIPTIONS = {
    "not_enough_verification:not-tested": "Agent claims finished but never verified it (no test, no manual check).",
    "not_enough_verification:self-report": "Agent itself says some important part wasn't tested.",
    "not_enough_verification:ask-for-manual-testing": "Agent asks the human to test something manually.",
    "not_enough_verification:repetitive-bug-fixes": "After agent called it done, user tested manually and reported bugs.",
    "not_enough_specification:no-spec-detected": "User asked for an implementation without a detailed enough spec.",
    "not_enough_specification:repetitive-requirements-fixes": "Agent fixed it the wrong way, user pushed back, repeatedly.",
    "not_enough_specification:self-report": "Agent itself says it doesn't have enough specification.",
    "no_visual_reference": "User asked for a UI/visual change but gave no image or design file.",
}

INSTRUCTIONS = (
    "For each session: open the file at 'prompt_file' and read it yourself "
    "BEFORE looking at what the model found (don't peek at logs/<id>.meta.json "
    "first, or you're just grading the model against itself). For each key "
    "below, replace null with true (it really happened, at least once in this "
    "session) or false (it did not happen). Leave it as null if you're not "
    "sure — those are excluded from scoring rather than guessed. 'note' is "
    "optional, for your own reference."
)


DEFAULT_THRESHOLDS = [0.0, 0.5, 0.7, 0.9]

FINDINGS_INSTRUCTIONS = (
    "Two kinds of items here.\n"
    "  'finding' items: the model already made this specific claim (shown "
    "inline below). Open prompt_file, check the evidence yourself against "
    "the real session, and set human_verdict to 'correct' (this really "
    "happened), 'incorrect' (it did not — a false positive), or 'unsure' "
    "(excluded from scoring).\n"
    "  'zero_finding_session' items: the model found NOTHING in this "
    "session, across every key. Open prompt_file and read it — if you spot "
    "something the model should have caught, add an entry like "
    "{\"call_name\": \"...\", \"subcategory\": \"...\", \"cause_prompt\": ..., "
    "\"note\": \"...\"} to human_missed_findings. IMPORTANT: it starts as "
    "null (not []), meaning 'not yet reviewed' — sessions left as null are "
    "EXCLUDED from scoring entirely, not silently counted as clean. Once "
    "you've actually read a session, set human_missed_findings to [] if it "
    "really is clean (a confirmed true negative), or to a list of entries "
    "if you found something (a confirmed miss)."
)


def _session_ids_from_logs(logs_dir):
    metas = sorted(glob.glob(os.path.join(logs_dir, "*.meta.json")))
    return [os.path.basename(m)[: -len(".meta.json")] for m in metas]


def generate_template(logs_dir, n, seed, out_path):
    ids = _session_ids_from_logs(logs_dir)
    if not ids:
        raise SystemExit(f"no *.meta.json files found in '{logs_dir}'")
    if n > len(ids):
        print(f"  ! asked for {n} sessions but only {len(ids)} are available — using all of them")
        n = len(ids)
    picked = sorted(random.Random(seed).sample(ids, n))

    sessions = []
    for session_id in picked:
        sessions.append({
            "session_id": session_id,
            "prompt_file": os.path.join(logs_dir, f"{session_id}.prompt.txt"),
            "human_judgment": {key: None for key in ALL_KEYS},
            "note": "",
        })

    template = {
        "instructions": INSTRUCTIONS,
        "key_descriptions": KEY_DESCRIPTIONS,
        "seed": seed,
        "logs_dir": logs_dir,
        "sessions": sessions,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print(f"Wrote {out_path} with {len(sessions)} sessions to judge by hand.")
    print("Next: open each session's prompt_file, fill in human_judgment, then run 'score'.")


def _usable_confidence(finding):
    """A finding's confidence, or 0.0 if missing/malformed. 0.0 is a
    deliberate convention, not a guess: at threshold 0.0 (trust everything)
    such a finding is still counted as predicted-positive, matching
    unthresholded behavior; at any real threshold above 0.0 it gets
    filtered out, since a broken confidence value shouldn't be trusted more
    than the lowest real one would be."""
    c = finding.get("confidence")
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return c
    return 0.0


def _all_calls_ok(meta):
    calls = meta.get("calls", {})
    return bool(calls) and all(c.get("ok") for c in calls.values())


def _session_has_any_finding(meta):
    return any(c.get("findings") for c in meta.get("calls", {}).values() if c.get("ok"))


def _all_findings_with_context(meta, session_id):
    out = []
    for call_name, call in meta.get("calls", {}).items():
        if not call.get("ok"):
            continue
        for f in call.get("findings", []) or []:
            out.append({
                "session_id": session_id,
                "call_name": call_name,
                "subcategory": f.get("subcategory"),
                "cause_prompt": f.get("cause_prompt"),
                "confidence": f.get("confidence"),
                "evidence": f.get("evidence", []),
            })
    return out


def sample_findings_template(logs_dir, total_n, n_zero_finding, max_per_session, seed, out_path):
    """Selects a mixed sample for a real precision/recall/accuracy check:
    mostly the model's LOWEST-confidence findings (most likely to be wrong,
    so most informative per item reviewed), plus a handful of sessions
    where the model found nothing at all (the only way to catch things it
    silently missed — recall is otherwise unmeasurable from findings alone,
    since a missed finding has no confidence value to sample by)."""
    ids = _session_ids_from_logs(logs_dir)
    if not ids:
        raise SystemExit(f"no *.meta.json files found in '{logs_dir}'")

    metas = {}
    for sid in ids:
        with open(os.path.join(logs_dir, f"{sid}.meta.json"), encoding="utf-8") as f:
            metas[sid] = json.load(f)

    usable_ids = [sid for sid in ids if _all_calls_ok(metas[sid])]
    skipped = len(ids) - len(usable_ids)
    if skipped:
        print(f"  ! {skipped} session(s) had a failed call and were excluded from sampling")

    zero_finding_ids = [sid for sid in usable_ids if not _session_has_any_finding(metas[sid])]
    finding_pool = []
    for sid in usable_ids:
        finding_pool.extend(_all_findings_with_context(metas[sid], sid))

    # Lowest confidence first. Missing/invalid confidence sorts to the very
    # front — a finding with no usable confidence is itself worth a look,
    # not just the ones honestly reporting low confidence.
    def sort_key(f):
        c = f["confidence"]
        return c if isinstance(c, (int, float)) and not isinstance(c, bool) else -1
    finding_pool.sort(key=sort_key)

    n_zero = min(n_zero_finding, len(zero_finding_ids))
    if n_zero < n_zero_finding:
        print(f"  ! only {len(zero_finding_ids)} zero-finding sessions available, using all of them")
    picked_zero = sorted(random.Random(seed).sample(zero_finding_ids, n_zero)) if n_zero else []

    n_findings_wanted = max(total_n - n_zero, 0)
    per_session_count = {}
    picked_findings = []
    for f in finding_pool:
        if len(picked_findings) >= n_findings_wanted:
            break
        sid = f["session_id"]
        if per_session_count.get(sid, 0) >= max_per_session:
            continue
        picked_findings.append(f)
        per_session_count[sid] = per_session_count.get(sid, 0) + 1

    if len(picked_findings) < n_findings_wanted:
        print(f"  ! only found {len(picked_findings)} eligible findings under --max-per-session="
              f"{max_per_session} (wanted {n_findings_wanted}) — try raising --max-per-session")

    items = []
    for f in picked_findings:
        items.append({
            "type": "finding",
            "session_id": f["session_id"],
            "prompt_file": os.path.join(logs_dir, f"{f['session_id']}.prompt.txt"),
            "call_name": f["call_name"],
            "subcategory": f["subcategory"],
            "cause_prompt": f["cause_prompt"],
            "confidence": f["confidence"],
            "evidence": f["evidence"],
            "human_verdict": None,
            "note": "",
        })
    for sid in picked_zero:
        items.append({
            "type": "zero_finding_session",
            "session_id": sid,
            "prompt_file": os.path.join(logs_dir, f"{sid}.prompt.txt"),
            "human_missed_findings": None,
            "note": "",
        })

    template = {
        "instructions": FINDINGS_INSTRUCTIONS,
        "key_descriptions": KEY_DESCRIPTIONS,
        "seed": seed,
        "logs_dir": logs_dir,
        "total_n": total_n,
        "n_zero_finding": len(picked_zero),
        "max_per_session": max_per_session,
        "items": items,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    n_sessions_touched = len(set(f["session_id"] for f in picked_findings)) + len(picked_zero)
    print(f"Wrote {out_path}: {len(picked_findings)} low-confidence findings + "
          f"{len(picked_zero)} zero-finding sessions, across {n_sessions_touched} distinct sessions.")
    print("Next: open each item's prompt_file, judge it, then run 'score-findings'.")


def _model_flags(meta, key):
    """True if the judge's findings for this session include this key at
    least once, mirroring exactly what 'counts[key]' measures in
    run_experiment.py — this benchmark checks the same thing the report
    reports, not a different, easier-to-pass definition. Returns
    (flagged, confidence) — confidence is the highest confidence among
    matching findings (the model's strongest claim), or None if it never
    flagged this key or the value wasn't a usable number."""
    call_name = key.split(":", 1)[0]
    call = meta.get("calls", {}).get(call_name)
    if not call or not call.get("ok"):
        return None, None  # call failed/missing — can't score this session for this key
    subcat_wanted = key.split(":", 1)[1] if ":" in key else call_name
    matches = [f for f in call.get("findings", []) or [] if f.get("subcategory") == subcat_wanted]
    if not matches:
        return False, None
    confidences = [f.get("confidence") for f in matches if isinstance(f.get("confidence"), (int, float))
                   and not isinstance(f.get("confidence"), bool) and 0 <= f.get("confidence") <= 1]
    return True, (max(confidences) if confidences else None)


def _confidence_bucket(confidence):
    """Coarse buckets for a calibration check — is the model's own
    confidence actually informative, or just noise? 'unusable' covers a
    missing, non-numeric, or out-of-range confidence — anything
    _coerce_confidence in classify.py would have flagged rather than
    repaired. Validated here directly (not just relying on the caller to
    have already filtered), since a bad value should never silently land
    in "high" just because a stray 1.5 clears the ">= 0.8" check."""
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return "unusable"
    if not 0 <= confidence <= 1:
        return "unusable"
    if confidence < 0.5:
        return "low (<0.5)"
    if confidence < 0.8:
        return "medium (0.5-0.8)"
    return "high (>=0.8)"


def score(template_path, logs_dir=None):
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)
    logs_dir = logs_dir or template.get("logs_dir", "logs")

    # key -> {"tp":0,"fp":0,"fn":0,"tn":0}
    stats = {key: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for key in ALL_KEYS}
    # Calibration is pooled across ALL keys, not per-key — at benchmark-sized
    # samples (tens of sessions), per-key buckets would mostly be empty.
    calibration = {"low (<0.5)": {"tp": 0, "fp": 0}, "medium (0.5-0.8)": {"tp": 0, "fp": 0},
                   "high (>=0.8)": {"tp": 0, "fp": 0}, "unusable": {"tp": 0, "fp": 0}}
    disagreements = []
    n_scored_sessions = 0
    n_skipped_sessions = 0

    for session in template["sessions"]:
        session_id = session["session_id"]
        meta_path = os.path.join(logs_dir, f"{session_id}.meta.json")
        if not os.path.exists(meta_path):
            print(f"  ! no meta.json for {session_id}, skipping entirely")
            n_skipped_sessions += 1
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        any_filled = False
        for key, human in session.get("human_judgment", {}).items():
            if human is None:
                continue  # unsure/unfilled — excluded from scoring
            any_filled = True
            model, confidence = _model_flags(meta, key)
            if model is None:
                continue  # that call failed for this session — can't compare
            if human and model:
                stats[key]["tp"] += 1
                calibration[_confidence_bucket(confidence)]["tp"] += 1
            elif not human and not model:
                stats[key]["tn"] += 1
            elif model and not human:
                stats[key]["fp"] += 1
                calibration[_confidence_bucket(confidence)]["fp"] += 1
                disagreements.append((session_id, key, "human=false, model=TRUE (false positive)"))
            elif human and not model:
                stats[key]["fn"] += 1
                disagreements.append((session_id, key, "human=true, model=FALSE (false negative)"))
        if any_filled:
            n_scored_sessions += 1

    print(f"scored {n_scored_sessions} sessions with at least one filled-in judgment "
          f"({n_skipped_sessions} skipped: no matching meta.json)\n")

    print(f"{'key':55s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} "
          f"{'precision':>10s} {'recall':>7s} {'agree':>7s}")
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for key in ALL_KEYS:
        s = stats[key]
        for k in total:
            total[k] += s[k]
        n = s["tp"] + s["fp"] + s["fn"] + s["tn"]
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else float("nan")
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else float("nan")
        agree = (s["tp"] + s["tn"]) / n if n else float("nan")
        print(f"{key:55s} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {s['tn']:>4} "
              f"{'n/a' if precision!=precision else f'{precision:.0%}':>10s} "
              f"{'n/a' if recall!=recall else f'{recall:.0%}':>7s} "
              f"{'n/a' if agree!=agree else f'{agree:.0%}':>7s}")

    n_total = sum(total.values())
    if n_total:
        overall_precision = total["tp"] / (total["tp"] + total["fp"]) if (total["tp"] + total["fp"]) else float("nan")
        overall_recall = total["tp"] / (total["tp"] + total["fn"]) if (total["tp"] + total["fn"]) else float("nan")
        overall_agree = (total["tp"] + total["tn"]) / n_total
        print(f"\nOVERALL ({n_total} judged key-instances): "
              f"precision={overall_precision:.0%} recall={overall_recall:.0%} "
              f"agreement={overall_agree:.0%}")

    print("\nconfidence calibration (pooled across all keys — only meaningful "
          "with a reasonable number of positive predictions):")
    print(f"{'bucket':20s} {'TP':>4s} {'FP':>4s} {'accuracy when model said yes':>30s}")
    for bucket in ("high (>=0.8)", "medium (0.5-0.8)", "low (<0.5)", "unusable"):
        c = calibration[bucket]
        n = c["tp"] + c["fp"]
        acc = c["tp"] / n if n else float("nan")
        print(f"{bucket:20s} {c['tp']:>4} {c['fp']:>4} "
              f"{'n/a (no positive predictions)' if acc != acc else f'{acc:.0%}':>30s}")
    print("If accuracy drops as the bucket gets lower, confidence is doing its "
          "job — the model 'knows what it doesn't know'. If it doesn't, the "
          "number isn't currently worth trusting on its own.")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements (go check these specific sessions):")
        for session_id, key, note in disagreements:
            print(f"  [{session_id}] {key} — {note}")
    else:
        print("\nno disagreements — every filled-in judgment matched the model")


def score_findings(template_path, logs_dir=None, thresholds=None):
    """Confusion matrix (TP/FP/FN/TN) + accuracy/precision/recall, swept
    across confidence thresholds. Raising the threshold should trade
    recall for precision — that trend is the actual point, not any single
    number on its own, especially at a manual-review sample size."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)
    logs_dir = logs_dir or template.get("logs_dir", "logs")

    finding_items = [it for it in template["items"] if it["type"] == "finding"]
    zero_items = [it for it in template["items"] if it["type"] == "zero_finding_session"]

    # Zero-finding sessions contribute a FIXED number of TN/FN, independent
    # of threshold — the model made zero claims there, so there is nothing
    # for a confidence cutoff to filter. IMPORTANT: human_missed_findings
    # starts as None ("not reviewed yet") when sampled. A None here means
    # exactly that — NOT reviewed — and is excluded from scoring entirely.
    # Only [] (explicitly reviewed and confirmed clean) counts as true
    # negatives; conflating "never checked" with "checked, found nothing"
    # would silently inflate accuracy/recall with unverified assumptions.
    fixed_tn = 0
    fixed_fn = 0
    fn_details = []
    n_unreviewed_zero = 0
    for it in zero_items:
        missed = it.get("human_missed_findings")
        if missed is None:
            n_unreviewed_zero += 1
            continue
        missed_keys = set()
        for m in missed:
            call_name = m.get("call_name", "")
            subcat = m.get("subcategory", call_name)
            key = f"{call_name}:{subcat}" if subcat and subcat != call_name else call_name
            missed_keys.add(key)
            fn_details.append((it["session_id"], key, m.get("note", "")))
        fixed_fn += len(missed_keys)
        fixed_tn += max(len(ALL_KEYS) - len(missed_keys), 0)

    scored = [it for it in finding_items if it.get("human_verdict") in ("correct", "incorrect")]
    n_unsure = len(finding_items) - len(scored)
    n_bad_confidence = sum(1 for it in scored if not isinstance(it.get("confidence"), (int, float))
                           or isinstance(it.get("confidence"), bool))
    n_reviewed_zero = len(zero_items) - n_unreviewed_zero

    print(f"{len(scored)} findings judged ({n_unsure} left unsure/blank, excluded), "
          f"{n_reviewed_zero}/{len(zero_items)} zero-finding sessions reviewed "
          f"({fixed_fn} missed finding(s) logged across them)")
    if n_unreviewed_zero:
        print(f"WARNING: {n_unreviewed_zero} zero-finding session(s) NOT yet reviewed — "
              f"excluded from scoring below, NOT counted as clean. Recall/accuracy here "
              f"only reflect the {n_reviewed_zero} actually checked.")
    if n_bad_confidence:
        print(f"note: {n_bad_confidence} judged finding(s) have a missing/invalid confidence — "
              f"treated as 0.0 (see _usable_confidence)")
    print()

    for T in thresholds:
        tp = fp = fn = tn = 0
        for it in scored:
            predicted_positive = _usable_confidence(it) >= T
            is_real = it["human_verdict"] == "correct"
            if predicted_positive and is_real:
                tp += 1
            elif predicted_positive and not is_real:
                fp += 1
            elif is_real:
                fn += 1
            else:
                tn += 1
        tn += fixed_tn
        fn += fixed_fn
        n_total = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        accuracy = (tp + tn) / n_total if n_total else float("nan")

        def pct(x):
            return "n/a" if x != x else f"{x:.0%}"

        print(f"--- threshold >= {T} ---")
        print(f"{'':16s}{'predicted +':>12s}{'predicted -':>12s}")
        print(f"{'actual +':16s}{tp:>12d}{fn:>12d}")
        print(f"{'actual -':16s}{fp:>12d}{tn:>12d}")
        print(f"accuracy={pct(accuracy)}  precision={pct(precision)}  recall={pct(recall)}  (N={n_total})\n")

    if fn_details:
        print("missed findings logged in zero-finding sessions:")
        for sid, key, note in fn_details:
            print(f"  [{sid}] {key} — {note}")

    incorrect = [it for it in scored if it["human_verdict"] == "incorrect"]
    if incorrect:
        print(f"\n{len(incorrect)} finding(s) marked incorrect (false positives) — go check these:")
        for it in incorrect:
            print(f"  [{it['session_id']}] {it['call_name']}:{it['subcategory']} "
                  f"(confidence={it.get('confidence')}) — {it.get('note', '')}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    gen = sub.add_parser("generate", help="create a blank ground-truth template from existing logs")
    gen.add_argument("n", type=int, help="how many sessions to sample")
    gen.add_argument("--logs-dir", default="logs")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", default="benchmark.json")

    sc = sub.add_parser("score", help="score a filled-in template against the logged model answers")
    sc.add_argument("--template", default="benchmark.json")
    sc.add_argument("--logs-dir", default=None, help="defaults to the logs_dir recorded in the template")

    sf = sub.add_parser("sample-findings",
                         help="sample low-confidence findings + zero-finding sessions for per-finding ground truth")
    sf.add_argument("--logs-dir", default="logs")
    sf.add_argument("--total", type=int, default=20, help="total items to sample (default 20)")
    sf.add_argument("--zero-finding", type=int, default=5,
                     help="how many zero-finding sessions to include, out of --total (default 5)")
    sf.add_argument("--max-per-session", type=int, default=2,
                     help="cap on how many findings can come from the same session (default 2)")
    sf.add_argument("--seed", type=int, default=42)
    sf.add_argument("--out", default="benchmark_findings.json")

    sc2 = sub.add_parser("score-findings", help="score a filled-in per-finding template, swept across confidence thresholds")
    sc2.add_argument("--template", default="benchmark_findings.json")
    sc2.add_argument("--logs-dir", default=None)
    sc2.add_argument("--thresholds", default="0.0,0.5,0.7,0.9",
                      help="comma-separated confidence thresholds to sweep (default 0.0,0.5,0.7,0.9)")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "generate":
        generate_template(args.logs_dir, args.n, args.seed, args.out)
    elif args.mode == "score":
        score(args.template, args.logs_dir)
    elif args.mode == "sample-findings":
        sample_findings_template(args.logs_dir, args.total, args.zero_finding,
                                  args.max_per_session, args.seed, args.out)
    elif args.mode == "score-findings":
        thresholds = [float(x) for x in args.thresholds.split(",")]
        score_findings(args.template, args.logs_dir, thresholds)