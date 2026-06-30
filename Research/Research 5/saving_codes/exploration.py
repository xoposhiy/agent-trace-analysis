from datasets import load_dataset

sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

total_research = 0
total_action = 0
n = 0
for s in sessions:
    if not s["input_tokens"]: continue
    if not s["turn_count"] or s["turn_count"] <= 30: continue
    if s["agent"] != "Claude Code": continue
    if not s["research_count"] or not s["action_count"]: continue
    total_research += s["research_count"]
    total_action += s["action_count"]
    n += 1

both = total_research + total_action
print("long Claude Code sessions analyzed:", n)
print(f"exploration (research) turns: {total_research:,}  ({100*total_research/both:.0f}%)")
print(f"editing (action) turns:       {total_action:,}  ({100*total_action/both:.0f}%)")