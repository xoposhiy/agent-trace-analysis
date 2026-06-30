from datasets import load_dataset

sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

# Opus 4.5 pricing (Sonnet = 3/15, Opus = 5/25, Haiku = 1/5)
BASE_INPUT  = 5.00
BASE_OUTPUT = 25.00

PRICE_INPUT          = BASE_INPUT  / 1_000_000
PRICE_OUTPUT         = BASE_OUTPUT / 1_000_000
PRICE_CACHE_CREATION = BASE_INPUT  * 1.25 / 1_000_000   # cache write
PRICE_CACHE_READ     = BASE_INPUT  * 0.10 / 1_000_000   # cache read (90% off)

def cost_of(s):
    return (s["input_tokens"]          * PRICE_INPUT
          + s["output_tokens"]         * PRICE_OUTPUT
          + s["cache_creation_tokens"] * PRICE_CACHE_CREATION
          + s["cache_read_tokens"]     * PRICE_CACHE_READ)

all_cost = 0.0
long_cost = 0.0
all_count = 0
long_count = 0

for s in sessions:
    if not s["input_tokens"]:
        continue
    all_cost += cost_of(s)
    all_count += 1
    if s["turn_count"] and s["turn_count"] > 30:
        long_cost += cost_of(s)
        long_count += 1

print("sessions with token data:", all_count)
print("  of those, long (>30 turns):", long_count,
      f"({100*long_count/all_count:.1f}%)")
print("total spend: $", round(all_cost, 2))
print("  spent on long sessions: $", round(long_cost, 2),
      f"({100*long_cost/all_cost:.1f}%)")