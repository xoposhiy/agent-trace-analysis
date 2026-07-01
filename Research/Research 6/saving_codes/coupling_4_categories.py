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

# --- sort each tool name into a category ---
def categorize(tool):
    t = tool.lower()
    if any(k in t for k in ["read", "grep", "glob", "lsp", "search", "fetch"]):
        return "exploration"
    if any(k in t for k in ["edit", "write"]) and "todowrite" not in t:
        return "editing"
    if "bash" in t:
        return "execution"
    return "coordination"   # todo, task, plan, issue-tracker, skill, etc.

cat_counts = Counter()
n_ok = 0
for sid in long_ids:                     # ALL long sessions this time
    try:
        p = hf_hub_download("SALT-NLP/SWE-chat", path_by_id[sid], repo_type="dataset")
        with open(p) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except:
                    continue             # skip malformed lines
                if e.get("type") != "assistant":
                    continue
                for block in e["message"].get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        cat_counts[categorize(block.get("name", ""))] += 1
        n_ok += 1
    except Exception as err:
        pass

total = sum(cat_counts.values())
print("sessions read:", n_ok)
print("total tool calls:", f"{total:,}")
print()
for cat in ["exploration", "editing", "execution", "coordination"]:
    c = cat_counts[cat]
    print(f"  {cat:14s} {c:>8,}  ({100*c/total:.0f}%)")