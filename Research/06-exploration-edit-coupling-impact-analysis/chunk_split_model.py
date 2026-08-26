"""Reusable math model for the cost of splitting a long agent session into chunks.

WHAT THIS MODULE IS
-------------------
The pure, data-free math behind Research7's "split a session at a context
threshold and pay a summary-carry overhead" analysis. Nothing here reads files
or prints -- every function takes numbers and returns numbers, so it can be
imported and reused (e.g. for SEMANTIC splitting, where chunk boundaries are not
evenly spaced: just pass your own `chunk_heights` list to the general functions).

THE PHYSICAL PICTURE (read this once, the code follows it exactly)
-----------------------------------------------------------------
Over a session the context grows ~linearly from 0 up to a peak P across N API
calls. Every call re-reads the whole current context, so the total cache-read is
the area under that ramp = N*P/2. Splitting means: whenever context would exceed
a threshold T, we RESET to a fresh chunk (seeded with a summary of the previous
chunk). Each chunk then only re-reads its OWN (smaller) content -> the big
cache-read triangle is replaced by k small ones. The price of that reset is the
summary carried into every new chunk.

HOW THE SPLIT IS PRICED (exact geometry)
----------------------------------------
`cache_read_after_split` (+ `uniform_chunk_heights`) works on an explicit list of
chunk heights; the last chunk climbs only its remainder R = P-(k-1)*T (partial),
so it is smaller in every bucket. This is the reusable core -- for arbitrary
(incl. semantic) splits, just pass your own heights list.

REUSE FOR SEMANTIC SPLITTING
----------------------------
Build a `chunk_heights` list from your semantic boundaries (token size of each
chunk, summing to P), then:
    calls   = calls_per_chunk(chunk_heights, api_calls, peak)
    cr_tok  = cache_read_after_split(chunk_heights, api_calls, peak)
    oh, *_  = summary_overhead_cost(calls[1:], summary_tokens(T, alpha))
    cost    = base_cost_after_split(it, ot, cw, cr_tok) + oh
Everything else (num_chunks / uniform_chunk_heights) is just the uniform-spaced helper.

All dollar figures are token-based ESTIMATES at blended Anthropic rates.
"""
import math
from dataclasses import dataclass

# ============================================================================
# SECTION 1 -- PRICING  (blended Opus-4.x $/MTok; override by passing your own)
# ============================================================================


@dataclass(frozen=True)
class Rates:
    input_per_mtok: float = 4.75      # fresh input tokens
    output_per_mtok: float = 23.75    # generated tokens
    cache_write_mult: float = 1.25    # writing a token into the cache (x input rate)
    cache_read_mult: float = 0.10     # re-reading a cached token (x input rate)


DEFAULT_RATES = Rates()


# ============================================================================
# SECTION 2 -- WHOLE-SESSION COST (no splitting): the "AS-IS" baseline
# ============================================================================


def full_session_cost(input_tok, output_tok, cache_write_tok, cache_read_tok,
                      rates=DEFAULT_RATES):
    """Cost of a session billed as-is (input + output + cache-write + cache-read)."""
    return (input_tok * rates.input_per_mtok
            + output_tok * rates.output_per_mtok
            + cache_write_tok * rates.input_per_mtok * rates.cache_write_mult
            + cache_read_tok * rates.input_per_mtok * rates.cache_read_mult) / 1e6


# ============================================================================
# SECTION 3 -- CHUNK GEOMETRY  (the reusable heart -- works for ANY split)
#   peak_context_tokens ....... invert the linear ramp: P from cache-read + calls
#   num_chunks ................ how many T-sized chunks a peak needs
#   uniform_chunk_heights ..... [T, T, ..., remainder]  (uniform boundaries, last partial)
#   calls_per_chunk ........... share the N calls out by each chunk's height
#   cache_read_after_split .... exact cache-read for arbitrary chunk heights
# ============================================================================


def peak_context_tokens(cache_read_tok, api_calls):
    """Peak context P implied by a session's cache-read and call count.

    Linear-ramp inversion: total cache-read = api_calls * P / 2  =>  P = 2*cr/N.
    """
    return 2.0 * cache_read_tok / api_calls


def num_chunks(peak, threshold):
    """How many chunks of size `threshold` are needed to cover `peak`."""
    if peak <= threshold:
        return 1
    return math.ceil(peak / threshold)


def uniform_chunk_heights(peak, threshold):
    """Uniform chunk heights: (k-1) full chunks of T, last = remainder R.

    The EXACT geometry (last chunk partial). For semantic splitting, build your
    own heights list instead of calling this.
    """
    k = num_chunks(peak, threshold)
    if k <= 1:
        return [peak]
    remainder = peak - (k - 1) * threshold
    return [threshold] * (k - 1) + [remainder]


def calls_per_chunk(chunk_heights, api_calls, peak):
    """Distribute the N API calls across chunks in proportion to their height.

    On a linear ramp, a chunk climbing height h occupies h/P of the calls.
    Returns one call-count per chunk (floats; they sum to `api_calls`).
    """
    return [h * api_calls / peak for h in chunk_heights]


def cache_read_after_split(chunk_heights, api_calls, peak):
    """EXACT cache-read tokens after splitting into the given chunk heights.

    Each chunk re-reads only its own content -> its triangle area is m_j*h_j/2;
    with m_j = h_j*N/P this sums to (N / 2P) * sum(h_j^2). General: valid for
    even, exact-remainder, OR semantic (uneven) chunk heights.
    """
    return (api_calls / (2.0 * peak)) * sum(h * h for h in chunk_heights)


# ============================================================================
# SECTION 4 -- SUMMARY-CARRY OVERHEAD  (the price of resetting into new chunks)
# ============================================================================


def summary_tokens(threshold, alpha):
    """Size of the carried summary: a fraction `alpha` of the threshold T."""
    return alpha * threshold


def summary_overhead_cost(new_chunk_call_counts, summary_tok, rates=DEFAULT_RATES):
    """Cost of seeding a summary into every NEW chunk (chunks 2..k).

    `new_chunk_call_counts` = the call count of each chunk that starts with a
    summary (i.e. all chunks except the first). For each such chunk of m calls:
      OUTPUT     to generate the summary        : out_rate * s
      CACHE-WRITE to load it at chunk start     : 1.25 * in_rate * s
      CACHE-READ  re-read on the chunk's other
                  (m-1) calls                    : 0.10 * in_rate * s * (m-1)
    Returns (total, gen, write, reread), all in dollars. Empty list -> all 0.
    """
    gen = write = reread = 0.0
    for m in new_chunk_call_counts:
        gen += rates.output_per_mtok * summary_tok
        write += rates.cache_write_mult * rates.input_per_mtok * summary_tok
        reread += rates.cache_read_mult * rates.input_per_mtok * summary_tok * max(m - 1.0, 0.0)
    return (gen + write + reread) / 1e6, gen / 1e6, write / 1e6, reread / 1e6


# ============================================================================
# SECTION 5 -- ASSEMBLED SPLIT COST  (base buckets at split cache-read + summary)
# ============================================================================


def base_cost_after_split(input_tok, output_tok, cache_write_tok,
                          cache_read_split_tok, rates=DEFAULT_RATES):
    """Session cost with the ORIGINAL input/output/cache-write but the REDUCED
    (post-split) cache-read. Add `summary_overhead_cost(...)` on top for the total.
    """
    return (input_tok * rates.input_per_mtok
            + output_tok * rates.output_per_mtok
            + cache_write_tok * rates.input_per_mtok * rates.cache_write_mult
            + cache_read_split_tok * rates.input_per_mtok * rates.cache_read_mult) / 1e6


def fresh_chunk_base_cost(input_tok, output_tok, cache_write_tok, call_fraction,
                          rates=DEFAULT_RATES):
    """A single chunk's share of input/output/cache-write, billed as a fresh
    mini-session: it owns `call_fraction` (= its calls / total calls) of each
    bucket. Used by the per-chunk price breakdown.
    """
    return (input_tok * call_fraction * rates.input_per_mtok
            + output_tok * call_fraction * rates.output_per_mtok
            + cache_write_tok * call_fraction * rates.input_per_mtok * rates.cache_write_mult) / 1e6


def chunk_own_cache_read_cost(chunk_calls, chunk_height, rates=DEFAULT_RATES):
    """Cache-read cost of a single chunk re-reading only its own content:
    triangle area m*h/2 priced at the cache-read rate.
    """
    return (chunk_calls * chunk_height / 2.0) * rates.input_per_mtok * rates.cache_read_mult / 1e6
