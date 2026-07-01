from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json

print(">>> step5_savings.py <<<")

logs = load_dataset("SALT-NLP/SWE-chat", "session_logs", split="train")
sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

long_ids = [s["session_id"] for s in sessions
            if s["input_tokens"] and s["turn_count"] and s["turn_count"] > 30
            and s["agent"] == "Claude Code"]
path_by_id = {l["session_id"]: l["transcript_path"] for l in logs}

# Opus rates ($/token)
BASE_INPUT = 5.00
P_WRITE = BASE_INPUT * 1.25 / 1_000_000     # cache write
P_READ  = BASE_INPUT * 0.10 / 1_000_000     # cache read
FILTER_RATE = 0.60                          # SWE-Edit Viewer removes ~60% of read code

def is_exploration(tool):
    t = tool.lower()
    return any(k in t for k in ["read", "grep", "glob", "lsp", "search", "fetch"])

def text_len(x):
    if x is None: return 0
    if isinstance(x, str): return len(x)
    return len(json.dumps(x))

total_expl_cost = 0.0     # lifetime billed cost of exploration tokens
total_all_cost  = 0.0     # lifetime billed cost of ALL tool-result tokens (for context)
n_ok = 0

for sid in long_ids:
    try:
        p = hf_hub_download("SALT-NLP/SWE-chat", path_by_id[sid], repo_type="dataset")
        # first pass: collect assistant turns in order, and map tool_use id -> name
        events = []
        with open(p) as f:
            for line in f:
                try: events.append(json.loads(line))
                except: continue

        id_to_name = {}
        # count total assistant turns = context "length" for lifetime calc
        assistant_turns = [e for e in events if e.get("type") == "assistant"]
        N = len(assistant_turns)
        if N == 0: continue

        # index each tool_use by the turn number it appears at
        turn_no = 0
        for e in events:
            msg = e.get("message", {})
            content = msg.get("content") if isinstance(msg, dict) else None
            if e.get("type") == "assistant":
                turn_no += 1
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            id_to_name[b.get("id")] = (b.get("name",""), turn_no)
            # results
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        name, t = id_to_name.get(b.get("tool_use_id"), (None, turn_no))
                        R = text_len(b.get("content")) / 4        # ~tokens
                        remaining = max(N - t, 0)
                        life_cost = R * (P_WRITE + remaining * P_READ)
                        total_all_cost += life_cost
                        if name and is_exploration(name):
                            total_expl_cost += life_cost
        n_ok += 1
    except Exception:
        pass

viewer_saving = total_expl_cost * FILTER_RATE

print(f"\nsessions read: {n_ok}")
print(f"lifetime cost of ALL tool-result tokens:      ${total_all_cost:,.0f}")
print(f"lifetime cost of EXPLORATION tokens:          ${total_expl_cost:,.0f}  "
      f"({100*total_expl_cost/total_all_cost:.0f}% of tool-result cost)")
print(f"estimated Viewer saving (60% of exploration): ${viewer_saving:,.0f}")