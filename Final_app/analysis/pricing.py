"""Dollars per token, by Claude model.

Anthropic bills four channels at different rates: fresh input, output, and two
cache channels (a write when new content enters the prompt cache, a read when
a later call re-uses it). ``analysis.attribution`` already knows exactly which
call produced or re-read which content and at which model — this module only
supplies the $/token rate for that model, so cost can be attributed with the
same per-call precision tokens already are, even across a session that mixes
models (a Haiku subagent inside a Sonnet main thread, for instance).

Rates below are Anthropic's first-party API list prices. Current-generation
figures are the ones on Anthropic's pricing page as of 2026-06-24; older
figures are their well-established historical prices, not re-verified against
a live page. Cache rates are rarely listed per model — Anthropic applies the
same multipliers to the base input price universally for the standard
(5-minute) cache: a write costs 1.25x input, a read costs 0.1x input. Where a
model's own cache rate is not separately known, that formula fills it in.

An unrecognized model (a future release, a fine-tune, a vendor this dashboard
does not otherwise support) is priced at ``DEFAULT_PRICE`` with
``is_estimated=True`` rather than raising or silently pricing at $0 — CLAUDE.md
§5: a failure here should degrade to a visibly-flagged guess, not vanish or
crash the page.
"""

from __future__ import annotations

from dataclasses import dataclass

# ----------------------------------------------------------------------
# Rate lookup
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPrice:
    """Dollars per single token, one model."""

    input: float
    output: float
    cache_read: float
    cache_write: float
    is_estimated: bool = False


CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def _priced(input_per_mtok: float, output_per_mtok: float,
            cache_read_per_mtok: float | None = None,
            cache_write_per_mtok: float | None = None,
            is_estimated: bool = False) -> ModelPrice:
    input_rate = input_per_mtok / 1_000_000
    output_rate = output_per_mtok / 1_000_000
    cache_read = (cache_read_per_mtok / 1_000_000 if cache_read_per_mtok is not None
                  else input_rate * CACHE_READ_MULTIPLIER)
    cache_write = (cache_write_per_mtok / 1_000_000 if cache_write_per_mtok is not None
                   else input_rate * CACHE_WRITE_MULTIPLIER)
    return ModelPrice(input_rate, output_rate, cache_read, cache_write, is_estimated)


# Keyed by the model-id substring Claude Code records at ``message.model``.
# Matching is longest-substring-wins (see ``price_for_model``), so a dated
# snapshot id like ``claude-opus-4-6-20260115`` still resolves to its family
# without every dated build needing its own entry.
PRICE_TABLE: dict[str, ModelPrice] = {
    # Current generation ($/MTok, cached 2026-06-24).
    "claude-opus-5": _priced(5.00, 25.00),
    "claude-sonnet-5": _priced(3.00, 15.00),
    "claude-fable-5": _priced(10.00, 50.00),
    "claude-mythos-5": _priced(10.00, 50.00),
    "claude-opus-4-8": _priced(5.00, 25.00),
    "claude-opus-4-7": _priced(5.00, 25.00),
    "claude-opus-4-6": _priced(5.00, 25.00),
    "claude-sonnet-4-6": _priced(3.00, 15.00),
    "claude-haiku-4-5": _priced(1.00, 5.00),
    # Earlier models, still findable in older transcripts. Long-standing
    # public list prices.
    "claude-sonnet-4-5": _priced(3.00, 15.00),
    "claude-sonnet-4": _priced(3.00, 15.00),
    "claude-opus-4-1": _priced(15.00, 75.00),
    "claude-opus-4": _priced(15.00, 75.00),
    "claude-sonnet-3-7": _priced(3.00, 15.00),
    "claude-haiku-3-5": _priced(0.80, 4.00),
    "claude-haiku-3": _priced(0.25, 1.25),
    "claude-opus-3": _priced(15.00, 75.00),
}

# Used when a model has no entry above — a release newer than this table, or
# a vendor build this dashboard does not otherwise track. Priced at the
# Sonnet-tier rate as the least-wrong single guess, and flagged as an
# estimate so a caller can choose to say so rather than present it as billed
# fact.
DEFAULT_PRICE = _priced(3.00, 15.00, is_estimated=True)


def price_for_model(model: str) -> ModelPrice:
    """Look up a model's per-token rates, longest matching key wins.

    Falls back to ``DEFAULT_PRICE`` (flagged ``is_estimated``) for anything
    unrecognized or blank, rather than raising — a session with one unknown
    model must still render a bar, just with a visibly approximate cost.
    """
    if model:
        matches = [key for key in PRICE_TABLE if key in model]
        if matches:
            return PRICE_TABLE[max(matches, key=len)]
    return DEFAULT_PRICE
