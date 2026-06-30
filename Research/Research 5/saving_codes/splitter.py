from huggingface_hub import hf_hub_download
from datasets import load_dataset
import json

BASE_INPUT, BASE_OUTPUT = 5.00, 25.00          # Opus
P_IN    = BASE_INPUT  / 1_000_000
P_OUT   = BASE_OUTPUT / 1_000_000
P_WRITE = BASE_INPUT * 1.25 / 1_000_000
P_READ  = BASE_INPUT * 0.10 / 1_000_000

LIMIT = 120_000          # token ceiling that triggers a "split" (tune later)
HANDOFF_TOKENS = 5_000   # summary re-fed to each new part (realistic handoff cost)

def turn_cost(u):
    return (u.get("input_tokens",0)*P_IN
          + u.get("output_tokens",0)*P_OUT
          + u.get("cache_creation_input_tokens",0)*P_WRITE
          + u.get("cache_read_input_tokens",0)*P_READ)

def analyze(events):
    orig = split_free = split_real = 0.0
    for e in events:
        if e.get("type") != "assistant": continue
        u = e["message"].get("usage", {})
        ctx = u.get("cache_read_input_tokens",0) + u.get("cache_creation_input_tokens",0)
        orig += turn_cost(u)
        if ctx > LIMIT:
            reduced = dict(u); reduced["cache_read_input_tokens"] = 0
            split_free += turn_cost(reduced)
            split_real += turn_cost(reduced) + HANDOFF_TOKENS*P_WRITE
        else:
            split_free += turn_cost(u)
            split_real += turn_cost(u)
    return orig, split_free, split_real

logs = load_dataset("SALT-NLP/SWE-chat", "session_logs", split="train")
sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")
long_ids = [s["session_id"] for s in sessions
            if s["input_tokens"] and s["turn_count"] and s["turn_count"]>30
            and s["agent"]=="Claude Code"]
path_by_id = {l["session_id"]: l["transcript_path"] for l in logs}

SAMPLE = 797   # start small; raise later
o=sf=sr=0.0; n=0
for sid in long_ids[:SAMPLE]:
    try:
        p = hf_hub_download("SALT-NLP/SWE-chat", path_by_id[sid], repo_type="dataset")
        with open(p) as f: ev=[json.loads(x) for x in f]
        a,b,c = analyze(ev); o+=a; sf+=b; sr+=c; n+=1
    except Exception as err:
        print("skip", sid, err)

print(f"\nsessions analyzed: {n}")
print(f"original cost:        ${o:,.2f}")
print(f"split (free handoff): ${sf:,.2f}   saving ${o-sf:,.2f}  ({100*(o-sf)/o:.0f}%)")
print(f"split (real handoff): ${sr:,.2f}   saving ${o-sr:,.2f}  ({100*(o-sr)/o:.0f}%)")


#Free handoff assumes the summary costs nothing — as
#               if you could split with zero penalty,
#               magically carrying over all needed knowledge
#               for free. This is impossible in reality. It's
#               the best-case ceiling — the absolute most you
#               could ever save. In your run, that was 30%.

#Real handoff charges a realistic cost for the summary (we
#               used 5,000 tokens per split, written into each
#               new part). This is closer to what splitting would
#               actually cost in practice. That gave 16%.