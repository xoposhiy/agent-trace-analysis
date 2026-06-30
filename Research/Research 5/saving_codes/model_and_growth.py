from huggingface_hub import hf_hub_download
from datasets import load_dataset
from collections import Counter
import json

# --- Part 1: what models do the long sessions use? ---
# we can read model per session cheaply from each transcript's first assistant msg,
# but that needs downloading all of them. Instead, sample: check the `agent` field
# you already have, plus confirm via a few transcripts.
# Quick version: just report what THIS transcript's model is, and check the
# sessions table's `agent` field distribution for long sessions.
sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")
agents = Counter()
for s in sessions:
    if s["input_tokens"] and s["turn_count"] and s["turn_count"] > 30:
        agents[s["agent"]] += 1
print("agent field for long sessions:", agents)

# --- Part 2: how does cache-read (context) grow across turns? ---
path = hf_hub_download(
    repo_id="SALT-NLP/SWE-chat",
    filename="transcripts/2026-01-05-0bb9a2d9-a51e-4d5a-8275-566c766a0ff2.jsonl",
    repo_type="dataset",
)
with open(path) as f:
    events = [json.loads(ln) for ln in f]

print("\nturn-by-turn context size (cache_read tokens) for assistant turns:")
turn = 0
for e in events:
    if e.get("type") == "assistant":
        u = e["message"].get("usage", {})
        ctx = u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        turn += 1
        if turn % 20 == 0 or turn <= 5:   # print first 5 then every 20th
            print(f"  turn {turn:3d}: context ≈ {ctx:,} tokens")