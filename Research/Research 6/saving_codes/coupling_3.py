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

# look at a small sample of sessions and count which tools the agent uses
tool_counts = Counter()
SAMPLE = 20
for sid in long_ids[:SAMPLE]:
    try:
        p = hf_hub_download("SALT-NLP/SWE-chat", path_by_id[sid], repo_type="dataset")
        with open(p) as f:
            for line in f:
                e = json.loads(line)
                if e.get("type") != "assistant":
                    continue
                # tool calls live inside the message content
                for block in e["message"].get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_counts[block.get("name", "unknown")] += 1
    except Exception as err:
        print("skip", sid, err)

print("tool usage across", SAMPLE, "long sessions:")
for name, count in tool_counts.most_common():
    print(f"  {name:20s} {count:,}")