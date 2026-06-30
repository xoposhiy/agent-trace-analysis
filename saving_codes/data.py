from datasets import load_dataset
s = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")
total = len(s)
long_turns   = sum(1 for x in s if x["turn_count"]   and x["turn_count"]   > 30)
long_prompts = sum(1 for x in s if x["prompt_count"] and x["prompt_count"] > 30)
print(f"by turn_count >30:   {long_turns}  ({100*long_turns/total:.1f}%)")
print(f"by prompt_count >30: {long_prompts}  ({100*long_prompts/total:.1f}%)")