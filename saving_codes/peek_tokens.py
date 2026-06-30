from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    repo_id="SALT-NLP/SWE-chat",
    filename="transcripts/2026-01-05-0bb9a2d9-a51e-4d5a-8275-566c766a0ff2.jsonl",
    repo_type="dataset",
)

with open(path) as f:
    events = [json.loads(ln) for ln in f]

# count event types
from collections import Counter
print("event types:", Counter(e.get("type") for e in events))

# find the first assistant message and show its structure
for e in events:
    if e.get("type") == "assistant":
        msg = e.get("message", {})
        print("\n--- first assistant message keys ---")
        print(sorted(msg.keys()))
        print("\n--- usage block (the tokens!) ---")
        print(json.dumps(msg.get("usage", "NO USAGE FIELD"), indent=2))
        print("\n--- model ---")
        print(msg.get("model", "no model field"))
        break