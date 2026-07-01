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

# collect every tool_use id we can find
call_ids = []
for e in events:
    if e.get("type") == "assistant":
        for b in e["message"].get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                call_ids.append(b.get("id"))
print("first 5 tool_use ids:", call_ids[:5])

# find first event that has a toolUseResult and dump its top-level id fields
print("\n=== first event with toolUseResult ===")
for e in events:
    if e.get("toolUseResult") is not None:
        print("top-level keys:", sorted(e.keys()))
        print("  sourceToolUseID:", e.get("sourceToolUseID"))
        print("  toolUseID:", e.get("toolUseID"))
        print("  parentToolUseID:", e.get("parentToolUseID"))
        # also check inside message.content for a tool_result block
        msg = e.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        print("  >> tool_result block found; keys:", sorted(b.keys()))
                        print("     tool_use_id:", b.get("tool_use_id"))
        break

# does that sourceToolUseID actually appear in our call_ids?
src = None
for e in events:
    if e.get("toolUseResult") is not None:
        src = e.get("sourceToolUseID"); break
print("\nsource id from a result:", src)
print("is it in call_ids?", src in call_ids)