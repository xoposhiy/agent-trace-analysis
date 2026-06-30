from huggingface_hub import hf_hub_download
import json

# download ONE transcript file
path = hf_hub_download(
    repo_id="SALT-NLP/SWE-chat",
    filename="transcripts/2026-01-05-0bb9a2d9-a51e-4d5a-8275-566c766a0ff2.jsonl",
    repo_type="dataset",
)
print("downloaded to:", path)

# read the first few lines (each line = one event)
with open(path) as f:
    lines = f.readlines()

print("total events in this transcript:", len(lines))
print("\n--- first event ---")
print(json.dumps(json.loads(lines[0]), indent=2)[:2000])

print("\n--- keys that appear across events ---")
keys = set()
for ln in lines:
    keys.update(json.loads(ln).keys())
print(sorted(keys))