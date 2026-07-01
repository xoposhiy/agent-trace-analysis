from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import Counter
import json

logs = load_dataset("SALT-NLP/SWE-chat", "session_logs", split="train")
sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

long_ids = [s["session_id"] for s in sessions
            if s["input_tokens"] and s["turn_count"] and s["turn_count"] > 30
            and s["agent"] == "Claude Code"]
path_by_id = {l["session_id"]: l["transcript_path"] for l in logs}

def categorize(tool):
    t = tool.lower()
    if any(k in t for k in ["read", "grep", "glob", "lsp", "search", "fetch"]):
        return "exploration"
    if any(k in t for k in ["edit", "write"]) and "todowrite" not in t:
        return "editing"
    if "bash" in t:
        return "execution"
    return "coordination"

def text_len(x):
    if x is None: return 0
    if isinstance(x, str): return len(x)
    return len(json.dumps(x))

cat_calls = Counter()
cat_tokens = Counter()
n_ok = 0

for sid in long_ids:
    try:
        p = hf_hub_download("SALT-NLP/SWE-chat", path_by_id[sid], repo_type="dataset")
        id_to_cat = {}
        with open(p) as f:
            for line in f:
                try: e = json.loads(line)
                except: continue
                if e.get("type") == "assistant":
                    for block in e["message"].get("content", []):
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            id_to_cat[block.get("id")] = categorize(block.get("name",""))
                            cat_calls[categorize(block.get("name",""))] += 1
                # result: match on sourceToolUseID -> the call's id
                res = e.get("toolUseResult")
                if res is not None:
                    tid = e.get("sourceToolUseID")
                    cat = id_to_cat.get(tid, "coordination")
                    cat_tokens[cat] += text_len(res) / 4
        n_ok += 1
    except Exception:
        pass

print("sessions read:", n_ok)
tot_calls = sum(cat_calls.values())
tot_tokens = sum(cat_tokens.values())
print(f"\n{'category':14s} {'calls':>10s} {'call%':>7s} {'tokens(M)':>11s} {'token%':>8s}")
for cat in ["exploration","editing","execution","coordination"]:
    c = cat_calls[cat]; tk = cat_tokens[cat]
    print(f"{cat:14s} {c:>10,} {100*c/tot_calls:>6.0f}% {tk/1e6:>10.1f}M {100*tk/tot_tokens:>7.0f}%")