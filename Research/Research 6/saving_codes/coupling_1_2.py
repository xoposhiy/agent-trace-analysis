from datasets import load_dataset

sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

# --- the long-session filter (same one we used before) ---
def is_long(s):
    return (s["input_tokens"]                       # has token data
            and s["turn_count"] and s["turn_count"] > 30   # long
            and s["agent"] == "Claude Code")               # Claude Code only

# --- STEP 1: how many of these sessions are there? ---
long_sessions = [s for s in sessions if is_long(s)]
print("STEP 1 — count")
print("  long Claude Code sessions (>30 turns):", len(long_sessions))
print()

# --- STEP 2: exploration vs editing, only using sessions that have both counts ---
total_research = 0
total_action = 0
usable = 0
for s in long_sessions:
    if not s["research_count"] or not s["action_count"]:
        continue
    total_research += s["research_count"]
    total_action += s["action_count"]
    usable += 1

both = total_research + total_action
print("STEP 2 — exploration vs editing")
print("  sessions with both counts:", usable)
print(f"  exploration (research) turns: {total_research:,.0f}  ({100*total_research/both:.0f}%)")
print(f"  editing (action) turns:       {total_action:,.0f}  ({100*total_action/both:.0f}%)")

# also useful: the average split *per session* (not just the pooled total)
per_session_expl = []
for s in long_sessions:
    if not s["research_count"] or not s["action_count"]:
        continue
    tot = s["research_count"] + s["action_count"]
    per_session_expl.append(s["research_count"] / tot)

avg = sum(per_session_expl) / len(per_session_expl)
print(f"  average exploration share per session: {100*avg:.0f}%")