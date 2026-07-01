from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json

logs = load_dataset("SALT-NLP/SWE-chat", "session_logs", split="train")
sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")
long_ids = [s["session_id"] for s in sessions
            if s["input_tokens"] and s["turn_count"] and s["turn_count"] > 30
            and s["agent"] == "Claude Code"]
path_by_id = {l["session_id"]: l["transcript_path"] for l in logs}

p = hf_hub_download("SALT-NLP/SWE-chat", path_by_id[long_ids[0]], repo_type="dataset")
with open(p) as f:
    events = [json.loads(ln) for ln in f if ln.strip()]

# 1) find one tool_use and print its id
print("=== a tool_use block ===")
for e in events:
    if e.get("type") == "assistant":
        for b in e["message"].get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                print("tool name:", b.get("name"))
                print("tool_use id:", b.get("id"))
                break
        else:
            continue
        break

# 2) find how a tool RESULT is stored — check user events and toolUseResult
print("\n=== a user event (where results usually live) ===")
for e in events:
    if e.get("type") == "user":
        print("keys:", sorted(e.keys()))
        msg = e.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict):
                        print("  content block type:", b.get("type"),
                              "| keys:", sorted(b.keys()))
        # also show the top-level toolUseResult if present
        if "toolUseResult" in e:
            tur = e["toolUseResult"]
            print("  toolUseResult type:", type(tur).__name__,
                  "| preview:", str(tur)[:120])
        break

# 3) show which top-level keys carry a tool id anywhere
print("\n=== keys containing 'tool' across event types ===")
seen = {}
for e in events:
    for k in e.keys():
        if "tool" in k.lower():
            seen.setdefault(k, str(e.get(k))[:60])
for k, v in seen.items():
    print(f"  {k}: {v}")