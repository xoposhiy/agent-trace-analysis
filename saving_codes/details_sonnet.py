from datasets import load_dataset

sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")

BASE_INPUT, BASE_OUTPUT = 3.00, 15.00
P_IN    = BASE_INPUT  / 1_000_000
P_OUT   = BASE_OUTPUT / 1_000_000
P_WRITE = BASE_INPUT * 1.25 / 1_000_000
P_READ  = BASE_INPUT * 0.10 / 1_000_000

c_in = c_out = c_write = c_read = 0.0
for s in sessions:
    if not s["input_tokens"]:                         continue
    if not s["turn_count"] or s["turn_count"] <= 30:  continue
    c_in    += s["input_tokens"]          * P_IN
    c_out   += s["output_tokens"]         * P_OUT
    c_write += s["cache_creation_tokens"] * P_WRITE
    c_read  += s["cache_read_tokens"]     * P_READ

total = c_in + c_out + c_write + c_read
print(f"fresh input:   ${c_in:,.0f}  ({100*c_in/total:.0f}%)")
print(f"output:        ${c_out:,.0f}  ({100*c_out/total:.0f}%)")
print(f"cache writes:  ${c_write:,.0f}  ({100*c_write/total:.0f}%)")
print(f"cache reads:   ${c_read:,.0f}  ({100*c_read/total:.0f}%)")
print(f"TOTAL:         ${total:,.0f}")