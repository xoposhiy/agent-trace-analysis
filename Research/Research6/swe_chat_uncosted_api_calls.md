# SWE-chat: most API calls are not cost-registered in `conversations`

**Finding: only ~10.6% of real API calls have their token cost recorded in the
`conversations` table.** A turn stores the cost of its *final reply only*, not
the tool-loop calls inside it.

## The numbers
| unit | count |
|---|---|
| Real API calls (`sessions.api_call_count`, sum) | **364,173** |
| Rows in `conversations` carrying tokens (`assistant_response`) | **38,549** |
| → share of API calls cost-registered | **10.6%** |

So ~89% of model invocations leave no costed row.

## Why
One agent **turn** ≠ one API call. Inside a turn the agent runs a tool-use loop:

```
think → call tool → (API call) → think → call tool → (API call) → … → final reply
```

Each arrow is a **separately billed** model inference (re-sends the growing
context as `cache_read` + emits `output`). The `conversations` table stamps
tokens onto the **single final `assistant_response` row** per turn and omits the
~dozens of intermediate tool-step calls. Tool *execution* is free; the model
*re-prompting* each tool call triggers is **not**.

## Proof (sum sessions vs sum conversations)
| token type | `sessions` (all calls) | `conversations` (final rows) | ratio |
|---|---|---|---|
| input | 3.59 B | 35.6 M | 100.8× |
| output | 119.0 M | 10.1 M | 11.8× |
| cache_creation | 1.04 B | 123 M | 8.4× |
| cache_read | 47.6 B | 4.9 B | 9.7× |

If the final-reply row captured the full turn, these would be 1.0×. The ~10×
gap (dominated by repeated `cache_read` of the growing context) is the cost of
the uncosted intermediate API calls — i.e. the context tax.

## Implication
Do **not** sum `conversations` per-row tokens for spend — it undercounts ~10–100×.
Use `sessions` aggregates for absolute cost; use `conversations` only for the
*shape* of recorded final replies.
