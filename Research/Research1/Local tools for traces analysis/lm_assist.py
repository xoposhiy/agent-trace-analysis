"""Thin client + CLI for the lm-assist observability API (Claude Code session traces).

The lm-assist service must be running locally:  `lm-assist status`
By default the API listens on http://localhost:3100.

Usage:
    python lm_assist.py projects                 # list projects with session counts
    python lm_assist.py sessions                  # list all sessions across projects
    python lm_assist.py show <session_id>         # summarize one session
    python lm_assist.py tools <session_id>        # tool-use frequency for one session
    python lm_assist.py cost                       # cost/token rollup across all sessions

Env:
    LM_ASSIST_URL   override base URL (default http://localhost:3100)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

import requests

BASE_URL = os.environ.get("LM_ASSIST_URL", "http://localhost:3100")


def _get(path: str) -> dict:
    """GET an API path and return the unwrapped `data` payload."""
    resp = requests.get(f"{BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success", False):
        raise RuntimeError(f"API error for {path}: {body}")
    return body["data"]


def list_projects() -> list[dict]:
    return _get("/projects")["projects"]


def list_sessions() -> list[dict]:
    return _get("/projects/sessions")["sessions"]


def get_session(session_id: str) -> dict:
    return _get(f"/sessions/{session_id}")


def _usd(x) -> str:
    try:
        return f"${float(x):.4f}"
    except (TypeError, ValueError):
        return "$0.0000"


def cmd_projects(_args: list[str]) -> None:
    for p in list_projects():
        flag = " [git]" if p.get("isGitProject") else ""
        print(f"{p['sessionCount']:>3} sessions  {p['path']}{flag}")


def cmd_sessions(_args: list[str]) -> None:
    rows = list_sessions()
    print(f"{'session':36}  {'prompts':>7}  active  project")
    for s in rows:
        sid = s["sessionId"]
        prompts = s.get("userPromptCount", s.get("totalUserPrompts", 0))
        active = "yes" if s.get("isActive") else "no"
        proj = os.path.basename(s.get("projectPath", ""))
        print(f"{sid:36}  {prompts:>7}  {active:>6}  {proj}")
    print(f"\n{len(rows)} session(s).")


def cmd_show(args: list[str]) -> None:
    if not args:
        sys.exit("usage: show <session_id>")
    s = get_session(args[0])
    print(f"Session     {s['sessionId']}")
    print(f"Project     {s.get('projectPath')}")
    print(f"Model       {s.get('model')}  (CC {s.get('claudeCodeVersion')})")
    print(f"Status      {s.get('status')}  active={s.get('isActive')}")
    print(f"Turns       {s.get('totalTurns', s.get('numTurns'))}")
    print(f"Prompts     {s.get('totalUserPrompts')}")
    print(f"Subagents   {s.get('totalSubagents')}")
    print(f"Duration    {(s.get('durationMs') or 0) / 1000:.1f}s")
    print(f"Cost        {_usd(s.get('totalCostUsd'))}")
    tools = s.get("toolUses") or []
    print(f"Tool uses   {len(tools)}")
    files = s.get("fileChanges") or []
    print(f"File changes {len(files)}")
    summary = s.get("sessionSummary")
    if summary:
        print(f"\nSummary: {summary}")


def cmd_tools(args: list[str]) -> None:
    if not args:
        sys.exit("usage: tools <session_id>")
    s = get_session(args[0])
    counts = Counter(
        t.get("name") or t.get("tool") or t.get("toolName") or "?"
        for t in (s.get("toolUses") or [])
    )
    if not counts:
        print("No tool uses recorded.")
        return
    for name, n in counts.most_common():
        print(f"{n:>4}  {name}")


def cmd_tokens(args: list[str]) -> None:
    """Print token consumption for one session (or all, with --all)."""
    if args and args[0] == "--all":
        targets = list_sessions()
    elif args:
        targets = [{"sessionId": args[0]}]
    else:
        sys.exit("usage: tokens <session_id> | tokens --all")

    grand = Counter()
    for row in targets:
        s = get_session(row["sessionId"])
        u = s.get("usage") or {}
        inp = u.get("inputTokens", 0)
        out = u.get("outputTokens", 0)
        cc = u.get("cacheCreationInputTokens", 0)
        cr = u.get("cacheReadInputTokens", 0)
        total = inp + out + cc + cr
        grand.update(input=inp, output=out, cache_create=cc, cache_read=cr, total=total)
        print(f"\n{s['sessionId']}  {_usd(s.get('totalCostUsd'))}")
        print(f"  {'Input':<16}{inp:>14,}")
        print(f"  {'Output':<16}{out:>14,}")
        print(f"  {'Cache creation':<16}{cc:>14,}")
        print(f"  {'Cache read':<16}{cr:>14,}")
        print(f"  {'TOTAL':<16}{total:>14,}")

    if len(targets) > 1:
        print(f"\n=== ALL {len(targets)} SESSIONS ===")
        for k in ("input", "output", "cache_create", "cache_read", "total"):
            print(f"  {k:<16}{grand[k]:>14,}")


# Claude Opus 4.x standard rates, USD per million tokens. Adjust if your
# model/plan differs — the `turns` command cross-checks its total against the
# API's reported totalCostUsd so you can see if these rates are off.
PRICE = {
    "input": 15.0,
    "output": 75.0,
    "cache_5m": 18.75,   # 5-minute cache write (1.25x input)
    "cache_1h": 30.0,    # 1-hour cache write   (2x input)
    "cache_read": 1.50,  # cache read           (0.1x input)
}


def _turn_cost(inp, out, c5, c1, cread) -> float:
    return (
        inp * PRICE["input"]
        + out * PRICE["output"]
        + c5 * PRICE["cache_5m"]
        + c1 * PRICE["cache_1h"]
        + cread * PRICE["cache_read"]
    ) / 1_000_000


def cmd_turns(args: list[str]) -> None:
    """Per-turn token + estimated cost breakdown (input/output/5m+1h cache/read)."""
    if not args:
        sys.exit("usage: turns <session_id>")
    path = _session_file(args[0])

    rows = []
    running = 0.0
    seen_ids = set()  # one assistant turn spans multiple JSONL lines (one per
                      # content block), all repeating the same usage — dedup by id
    for line in open(path):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = o.get("message", {})
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        u = m.get("usage")
        if not u:
            continue
        mid = m.get("id")
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        inp = u.get("input_tokens", 0)
        out = u.get("output_tokens", 0)
        cread = u.get("cache_read_input_tokens", 0)
        cc = u.get("cache_creation") or {}
        c5 = cc.get("ephemeral_5m_input_tokens", 0)
        c1 = cc.get("ephemeral_1h_input_tokens", 0)
        cost = _turn_cost(inp, out, c5, c1, cread)
        running += cost
        rows.append((len(rows) + 1, inp, out, c5, c1, cread, cost))

    hdr = f"{'#':>3} {'in':>7} {'out':>7} {'c5m':>8} {'c1h':>8} {'cRead':>10} {'$cost':>9}"
    print(hdr)
    print("-" * len(hdr))
    for n, inp, out, c5, c1, cread, cost in rows:
        spike = "  <-- spike" if cost > 0.10 else ""
        print(f"{n:>3} {inp:>7,} {out:>7,} {c5:>8,} {c1:>8,} {cread:>10,} {cost:>9.4f}{spike}")
    print("-" * len(hdr))
    print(f"{len(rows)} assistant turns · estimated total ${running:.4f}")
    try:
        actual = float(get_session(args[0]).get("totalCostUsd") or 0)
        print(f"API-reported total ${actual:.4f}  (rate check: "
              f"{'OK' if abs(actual - running) < max(0.05, actual * 0.1) else 'rates may be off'})")
    except Exception:
        pass
    print(
        "\nColumns:\n"
        f"  #      turn number (row label, not tokens)\n"
        f"  in     NEW uncached input tokens sent      (${PRICE['input']:.2f}/M)\n"
        f"  out    tokens the model generated          (${PRICE['output']:.2f}/M, priciest)\n"
        f"  c5m    tokens written to 5-min cache       (${PRICE['cache_5m']:.2f}/M)\n"
        f"  c1h    tokens written to 1-hour cache      (${PRICE['cache_1h']:.2f}/M)\n"
        f"  cRead  tokens read back from cache         (${PRICE['cache_read']:.2f}/M, grows all session)\n"
        f"  $cost  estimated USD this turn; '<-- spike' = over $0.10"
    )


def cmd_detail(args: list[str]) -> None:
    """Everything per turn: tokens, cost, thinking, tool calls, subagents spawned."""
    if not args:
        sys.exit("usage: detail <session_id>")
    path = _session_file(args[0])

    # Reassemble each assistant turn by message id (blocks span multiple lines).
    turns = {}
    order = []
    for line in open(path):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = o.get("message", {})
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        mid = m.get("id")
        if mid not in turns:
            turns[mid] = {"usage": None, "tools": [], "subagents": [], "think": False}
            order.append(mid)
        t = turns[mid]
        if m.get("usage") and t["usage"] is None:
            t["usage"] = m["usage"]
        for b in m.get("content") or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "thinking":
                t["think"] = True
            elif bt == "tool_use":
                name = b.get("name")
                if name in ("Task", "Agent"):
                    sub = (b.get("input") or {}).get("subagent_type", "?")
                    t["subagents"].append(sub)
                t["tools"].append(name)

    # Pull the spawned-subagent rollup once so workflow/Task turns can be annotated.
    sub_rows, sub_grand, sub_tools, _ = _collect_subagents(args[0])
    spawned_total = {
        "agents": len(sub_rows),
        "tools": sum(sub_tools.values()),
        "cost": sub_grand["cost"],
    }
    spawned_attributed = False  # attribute the rollup to the first Workflow/Task turn

    hdr = f"{'#':>3} {'T':1} {'out':>6} {'cRead':>8} {'$':>7}  tool calls / subagents"
    print(hdr)
    print("-" * 72)
    for n, mid in enumerate(order, 1):
        t = turns[mid]
        u = t["usage"] or {}
        out = u.get("output_tokens", 0)
        cread = u.get("cache_read_input_tokens", 0)
        cc = u.get("cache_creation") or {}
        cost = _turn_cost(
            u.get("input_tokens", 0), out,
            cc.get("ephemeral_5m_input_tokens", 0),
            cc.get("ephemeral_1h_input_tokens", 0), cread,
        )
        tools = Counter(t["tools"])
        tool_str = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in tools.items()) or "(none)"
        # inline-spawned subagents (Task/Agent) recorded on the turn itself
        if t["subagents"]:
            tool_str += f"  ⟶ {', '.join(t['subagents'])}"
        # workflow turn: attribute the descended rollup here
        if ("Workflow" in tools or "Task" in tools) and not spawned_attributed and sub_rows:
            spawned_attributed = True
            kind = "workflow agents" if "Workflow" in tools else "subagents"
            tool_str += (f"  ⟶ {kind}: {spawned_total['agents']} agents, "
                         f"{spawned_total['tools']} tool calls, est. ${spawned_total['cost']:.2f}")
        think = "T" if t["think"] else " "
        print(f"{n:>3} {think} {out:>6,} {cread:>8,} {cost:>7.3f}  {tool_str}")

    print("-" * 72)
    main_tools = Counter(name for mid in order for name in turns[mid]["tools"])
    main_cost = sum(
        _turn_cost(
            (turns[m]["usage"] or {}).get("input_tokens", 0),
            (turns[m]["usage"] or {}).get("output_tokens", 0),
            ((turns[m]["usage"] or {}).get("cache_creation") or {}).get("ephemeral_5m_input_tokens", 0),
            ((turns[m]["usage"] or {}).get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0),
            (turns[m]["usage"] or {}).get("cache_read_input_tokens", 0),
        )
        for m in order
    )
    print(f"MAIN THREAD: {len(order)} turns · {sum(main_tools.values())} tool calls · est. ${main_cost:.2f}")
    print(f"  tool totals: {', '.join(f'{k}×{v}' for k, v in main_tools.most_common())}")
    if sub_rows:
        print(f"SPAWNED SUBAGENTS: {spawned_total['agents']} agents · "
              f"{spawned_total['tools']} tool calls · est. ${spawned_total['cost']:.2f}")
        print(f"GRAND TOTAL est. ${main_cost + spawned_total['cost']:.2f}  "
              f"(run `subagents {args[0]}` for the per-agent breakdown)")
    print("legend: T=thinking present · out=output tokens · cRead=cache-read tokens · $=est. cost")


def _aggregate_jsonl(path: str) -> dict:
    """Sum tokens (dedup by msg id), cost, and tool calls for one JSONL transcript."""
    seen = set()
    agg = {"in": 0, "out": 0, "c5m": 0, "c1h": 0, "cread": 0, "tools": Counter(), "prompt": ""}
    for line in open(path):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = o.get("message", {})
        if not isinstance(m, dict):
            continue
        if not agg["prompt"] and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                agg["prompt"] = c[:80]
            elif isinstance(c, list):
                agg["prompt"] = " ".join(
                    b.get("text", "") for b in c if isinstance(b, dict)
                )[:80]
        for b in m.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                agg["tools"][b.get("name")] += 1
        if m.get("role") == "assistant" and m.get("usage"):
            mid = m.get("id")
            if mid in seen:
                continue
            seen.add(mid)
            u = m["usage"]
            cc = u.get("cache_creation") or {}
            agg["in"] += u.get("input_tokens", 0)
            agg["out"] += u.get("output_tokens", 0)
            agg["c5m"] += cc.get("ephemeral_5m_input_tokens", 0)
            agg["c1h"] += cc.get("ephemeral_1h_input_tokens", 0)
            agg["cread"] += u.get("cache_read_input_tokens", 0)
    agg["cost"] = _turn_cost(agg["in"], agg["out"], agg["c5m"], agg["c1h"], agg["cread"])
    return agg


def _collect_subagents(session_id: str):
    """Return (rows, grand totals, tool Counter, n_files) for a session's subagent transcripts."""
    base = os.path.dirname(_session_file(session_id))
    sub_root = os.path.join(base, session_id, "subagents")
    files = []
    for root, _dirs, names in os.walk(sub_root):
        for n in names:
            if n.endswith(".jsonl"):
                files.append(os.path.join(root, n))

    rows = []
    grand = {"in": 0, "out": 0, "c5m": 0, "c1h": 0, "cread": 0, "cost": 0.0}
    all_tools = Counter()
    for f in files:
        a = _aggregate_jsonl(f)
        if a["out"] == 0 and not a["tools"]:
            continue  # empty / orchestrator stub
        rows.append((a["cost"], os.path.basename(f)[:18], a))
        for k in grand:
            grand[k] += a[k]
        all_tools.update(a["tools"])
    return rows, grand, all_tools, len(files)


def cmd_subagents(args: list[str]) -> None:
    """Descend into workflow/subagent transcripts and aggregate tokens, cost, tool calls."""
    if not args:
        sys.exit("usage: subagents <session_id>")
    rows, grand, all_tools, n_files = _collect_subagents(args[0])
    if not rows:
        print("No subagent transcripts found.")
        return

    print(f"Subagent rollup for {args[0]}")
    print(f"  {len(rows)} agents with activity (of {n_files} transcript files)\n")
    print(f"{'agent':20} {'out':>8} {'cRead':>10} {'$':>8}  top tools")
    print("-" * 72)
    for cost, name, a in sorted(rows, reverse=True)[:25]:
        tools = ", ".join(f"{k}×{v}" for k, v in a["tools"].most_common(3)) or "(none)"
        print(f"{name:20} {a['out']:>8,} {a['cread']:>10,} {cost:>8.3f}  {tools}")
    if len(rows) > 25:
        print(f"  ... and {len(rows) - 25} more (showing top 25 by cost)")
    print("-" * 72)
    print(f"TOTAL: {len(rows)} agents · {sum(all_tools.values())} tool calls · "
          f"est. ${grand['cost']:.4f}")
    print(f"  tokens — in {grand['in']:,} · out {grand['out']:,} · "
          f"cache_read {grand['cread']:,} · cache_write {grand['c5m']+grand['c1h']:,}")
    print(f"  tool totals: {', '.join(f'{k}×{v}' for k, v in all_tools.most_common())}")


def cmd_loops(args: list[str]) -> None:
    """Detect repetitive / stuck behavior: same tool 3+ consecutive, or repeated commands."""
    if not args:
        sys.exit("usage: loops <session_id>")
    path = _session_file(args[0])

    seq = []           # ordered (tool_name, command-or-None)
    for line in open(path):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = o.get("message", {})
        if not isinstance(m, dict):
            continue
        for b in _blocks(m):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                name = b.get("name")
                cmd = (b.get("input") or {}).get("command") if name == "Bash" else None
                seq.append((name, cmd.strip() if cmd else None))

    # 1) runs of the same tool 3+ times in a row
    runs = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j][0] == seq[i][0]:
            j += 1
        if j - i >= 3:
            runs.append((seq[i][0], j - i, i + 1))
        i = j

    # 2) identical commands repeated (anywhere)
    cmd_counts = Counter(c for _, c in seq if c)
    repeated = {c: n for c, n in cmd_counts.items() if n >= 2}

    print(f"Loop / repetition report for {args[0]}")
    print(f"  {len(seq)} tool calls scanned\n")

    if runs:
        print("Consecutive same-tool runs (3+):")
        for name, length, pos in runs:
            print(f"  {length}x {name} in a row (starting at call #{pos})")
    else:
        print("No 3+ consecutive same-tool runs.")

    if repeated:
        print("\nIdentical commands repeated:")
        for c, n in sorted(repeated.items(), key=lambda kv: -kv[1]):
            print(f"  {n}x  {c[:80]}")
    else:
        print("\nNo identical commands repeated.")

    if not runs and not repeated:
        print("\n=> No looping/stuck patterns detected.")


def _session_file(session_id: str) -> str:
    """Resolve the raw JSONL path for a session via the API."""
    for s in list_sessions():
        if s["sessionId"] == session_id:
            return s["filePath"]
    raise SystemExit(f"session {session_id} not found")


# Phrases in a user turn that signal the previous agent step missed the mark.
FRUSTRATION = re.compile(
    r"(does(n'?t| not) work|not working|still (broke|fail|not)|that'?s not|"
    r"\bwrong\b|\bnope\b|\bundo\b|\brevert\b|why (did|is|isn'?t)|"
    r"isn'?t work|\bbroken\b|doesn'?t)",
    re.IGNORECASE,
)


def _blocks(msg: dict) -> list:
    c = msg.get("content")
    return c if isinstance(c, list) else []


def cmd_friction(args: list[str]) -> None:
    """Surface friction points in a session: errors, non-zero exits, retries, corrections."""
    if not args:
        sys.exit("usage: friction <session_id>")
    path = _session_file(args[0])

    tool_names = {}          # tool_use id -> name (to label results)
    bash_cmds = Counter()    # command string -> times run (retry signal)
    events = []              # (line_index, kind, detail)

    for i, line in enumerate(open(path)):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = o.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        for b in _blocks(msg):
            if not isinstance(b, dict):
                continue
            btype = b.get("type")

            if btype == "tool_use":
                tool_names[b.get("id")] = b.get("name")
                if b.get("name") == "Bash":
                    cmd = (b.get("input") or {}).get("command", "")
                    if cmd:
                        bash_cmds[cmd.strip()] += 1

            elif btype == "tool_result":
                name = tool_names.get(b.get("tool_use_id"), "?")
                text = b.get("content")
                if isinstance(text, list):
                    text = " ".join(
                        str(x.get("text", "")) for x in text if isinstance(x, dict)
                    )
                text = str(text or "")
                if b.get("is_error"):
                    events.append((i, "ERROR", f"{name}: {text[:120].strip()}"))
                else:
                    # The harness prefixes a failed Bash result with "Exit code N"
                    # at the very start. Anchor to ^ so we don't match exit codes
                    # quoted inside command output.
                    m = re.match(r"\s*Exit code (\d+)", text)
                    if m and m.group(1) != "0":
                        events.append(
                            (i, "EXIT", f"{name} exit {m.group(1)}: {text[:100].strip()}")
                        )

        # genuine user turns (tool_result blocks also carry role=user; skip those)
        if role == "user" and not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in _blocks(msg)
        ):
            text = msg.get("content")
            if isinstance(text, list):
                text = " ".join(
                    b.get("text", "") for b in text if isinstance(b, dict)
                )
            text = str(text or "")
            if FRUSTRATION.search(text):
                events.append((i, "USER", text[:120].strip().replace("\n", " ")))

    retries = {c: n for c, n in bash_cmds.items() if n > 1}

    print(f"Friction report for {args[0]}")
    print(f"  source: {path}\n")
    print(f"  errors / non-zero exits / corrections: {len(events)}")
    print(f"  repeated commands (retries): {len(retries)}\n")

    if events:
        print("Timeline:")
        for line_idx, kind, detail in events:
            print(f"  L{line_idx:<4} [{kind:5}] {detail}")
    if retries:
        print("\nRepeated commands:")
        for cmd, n in sorted(retries.items(), key=lambda kv: -kv[1]):
            print(f"  {n}x  {cmd[:90]}")
    if not events and not retries:
        print("No friction signals detected.")


def cmd_cost(_args: list[str]) -> None:
    total = 0.0
    rows = []
    for s in list_sessions():
        detail = get_session(s["sessionId"])
        c = float(detail.get("totalCostUsd") or 0)
        total += c
        rows.append((c, s["sessionId"], os.path.basename(s.get("projectPath", ""))))
    for c, sid, proj in sorted(rows, reverse=True):
        print(f"{_usd(c):>10}  {sid}  {proj}")
    print(f"\nTotal across {len(rows)} session(s): {_usd(total)}")


COMMANDS = {
    "projects": cmd_projects,
    "sessions": cmd_sessions,
    "show": cmd_show,
    "tools": cmd_tools,
    "tokens": cmd_tokens,
    "turns": cmd_turns,
    "detail": cmd_detail,
    "subagents": cmd_subagents,
    "loops": cmd_loops,
    "friction": cmd_friction,
    "cost": cmd_cost,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python lm_assist.py <{'|'.join(COMMANDS)}> [args]")
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
