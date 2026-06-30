from datasets import load_dataset

sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

# --- set these to the real Claude prices (dollars per 1 token) ---
# check current prices at docs.claude.com — these are EXAMPLES only
PRICE_INPUT          = 3.00  / 1_000_000   # fresh input
PRICE_OUTPUT         = 15.00 / 1_000_000   # output
PRICE_CACHE_CREATION = 3.75  / 1_000_000   # writing to cache
PRICE_CACHE_READ     = 0.30  / 1_000_000   # reading from cache (cheap!)

total_cost = 0.0
long_count = 0

for s in sessions:
    # skip rows with no token data
    if not s["input_tokens"]:
        continue
    # keep only long sessions (more than 30 turns)
    if not s["turn_count"] or s["turn_count"] <= 30:
        continue

    cost = (s["input_tokens"]          * PRICE_INPUT
          + s["output_tokens"]         * PRICE_OUTPUT
          + s["cache_creation_tokens"] * PRICE_CACHE_CREATION
          + s["cache_read_tokens"]     * PRICE_CACHE_READ)

    total_cost += cost
    long_count += 1

print("long sessions with token data:", long_count)
print("total cost of those sessions: $", round(total_cost, 2))