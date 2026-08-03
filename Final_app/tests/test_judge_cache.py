"""Judge caching tests.

**No test here calls an LLM.** The client is stubbed or the cache is
pre-seeded. A suite that needs VPN is not a suite (CLAUDE.md §6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Final_app.judge import summary as judge


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the cache at ``tmp_path`` and reset the in-process copy.

    ``_cache`` is a module-level singleton, so without the reset one test's
    entries leak into the next.
    """
    monkeypatch.setattr(judge, "_CACHE_FILE", tmp_path / "summaries.json")
    monkeypatch.setattr(judge, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(judge, "_cache", None)
    yield
    monkeypatch.setattr(judge, "_cache", None)


# ----------------------------------------------------------------------
# Key derivation
# ----------------------------------------------------------------------

def test_same_prompts_produce_the_same_key():
    assert judge.cache_key(["a", "b"]) == judge.cache_key(["a", "b"])


def test_different_prompts_produce_different_keys():
    assert judge.cache_key(["a"]) != judge.cache_key(["b"])


def test_prompt_order_is_part_of_the_key():
    assert judge.cache_key(["a", "b"]) != judge.cache_key(["b", "a"])


def test_changing_the_model_invalidates_the_key(monkeypatch: pytest.MonkeyPatch):
    """A new model must re-summarise rather than serve the old model's text."""
    before = judge.cache_key(["a"])
    monkeypatch.setattr(judge, "JUDGE_MODEL", "some-other-model")

    assert judge.cache_key(["a"]) != before


def test_whitespace_differences_do_not_split_the_key():
    assert judge.cache_key(["hello  world"]) == judge.cache_key(["hello world"])


# ----------------------------------------------------------------------
# Cache behaviour
# ----------------------------------------------------------------------

def test_cached_summary_never_calls_the_llm(monkeypatch: pytest.MonkeyPatch):
    """The list page uses this, which is why it renders with no VPN."""
    def explode():
        raise AssertionError("cached_summary must not reach the network")

    monkeypatch.setattr(judge, "llm_available", explode)

    assert judge.cached_summary(["anything"]) == ""


def test_cached_summary_returns_a_stored_entry():
    prompts = ["build a dashboard"]
    judge._load_cache()[judge.cache_key(prompts)] = "Built a dashboard."

    assert judge.cached_summary(prompts) == "Built a dashboard."


def test_a_cache_hit_skips_the_llm_entirely(monkeypatch: pytest.MonkeyPatch):
    prompts = ["cached already"]
    judge._load_cache()[judge.cache_key(prompts)] = "Stored answer."
    monkeypatch.setattr(judge, "llm_available",
                        lambda: pytest.fail("should not be consulted"))

    assert judge.summarize(prompts) == "Stored answer."


def test_summarize_persists_across_a_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    prompts = ["persist me"]
    judge._load_cache()[judge.cache_key(prompts)] = "Persisted."
    judge._save_cache()

    monkeypatch.setattr(judge, "_cache", None)  # force a re-read from disk

    assert judge.cached_summary(prompts) == "Persisted."


def test_a_corrupt_cache_file_is_ignored_not_fatal(tmp_path: Path, monkeypatch):
    (tmp_path / "summaries.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(judge, "_cache", None)

    assert judge._load_cache() == {}


def test_empty_prompts_are_never_summarised():
    assert judge.summarize([]) == ""
    assert judge.cached_summary([]) == ""


def test_summarize_returns_empty_when_the_llm_is_unavailable(
    monkeypatch: pytest.MonkeyPatch
):
    """No key, no network, no VPN — the caller gets "" and the UI copes."""
    monkeypatch.setattr(judge, "llm_available", lambda: False)

    assert judge.summarize(["uncached prompt"]) == ""


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

def test_prompt_includes_every_message_in_order():
    prompt = judge.build_summary_prompt(["first", "second"])

    assert "1. first" in prompt
    assert "2. second" in prompt


def test_prompt_caps_how_many_messages_it_sends():
    prompt = judge.build_summary_prompt([f"msg{i}" for i in range(100)])

    assert f"{judge.MAX_PROMPTS}. msg{judge.MAX_PROMPTS - 1}" in prompt
    assert f"{judge.MAX_PROMPTS + 1}." not in prompt


def test_prompt_truncates_very_long_messages():
    prompt = judge.build_summary_prompt(["x" * 5000])

    assert "x" * (judge.MAX_PROMPT_CHARS + 1) not in prompt
