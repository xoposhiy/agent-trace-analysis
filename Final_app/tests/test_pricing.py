"""Tests for the model price table (CLAUDE.md §6)."""

from __future__ import annotations

from Final_app.analysis.pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    DEFAULT_PRICE,
    price_for_model,
)


def test_a_known_model_prices_at_its_listed_rate():
    price = price_for_model("claude-opus-4-6")

    assert price.input == 5.00 / 1_000_000
    assert price.output == 25.00 / 1_000_000
    assert not price.is_estimated


def test_a_dated_snapshot_id_still_resolves_to_its_family():
    """Claude Code records ``message.model`` with a date suffix in practice."""
    dated = price_for_model("claude-opus-4-6-20260115")
    family = price_for_model("claude-opus-4-6")

    assert dated == family


def test_the_longer_more_specific_key_wins_over_a_shorter_prefix():
    """``claude-opus-4-6`` must not be priced as the plainer ``claude-opus-4``.

    ``"claude-opus-4"`` is a substring of ``"claude-opus-4-6-20260115"`` too,
    and the two families are priced 3x apart — the wrong match would be loud.
    """
    opus_4 = price_for_model("claude-opus-4-20250101")
    opus_4_6 = price_for_model("claude-opus-4-6-20260115")

    assert opus_4.input == 15.00 / 1_000_000
    assert opus_4_6.input == 5.00 / 1_000_000

    haiku_3 = price_for_model("claude-haiku-3-20240307")
    haiku_3_5 = price_for_model("claude-haiku-3-5-20241022")
    assert haiku_3.input != haiku_3_5.input


def test_cache_rates_default_from_the_standard_anthropic_multipliers():
    price = price_for_model("claude-sonnet-4-6")

    assert price.cache_write == price.input * CACHE_WRITE_MULTIPLIER
    assert price.cache_read == price.input * CACHE_READ_MULTIPLIER


def test_an_unrecognized_model_falls_back_to_a_flagged_estimate():
    price = price_for_model("claude-future-model-9000")

    assert price == DEFAULT_PRICE
    assert price.is_estimated


def test_a_blank_model_also_falls_back_rather_than_raising():
    assert price_for_model("") == DEFAULT_PRICE
    assert price_for_model(None) == DEFAULT_PRICE  # type: ignore[arg-type]
