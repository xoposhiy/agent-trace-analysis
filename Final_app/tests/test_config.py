"""LLM diagnostics: config inspection, and the live probe.

``llm_diagnostics`` reads configuration. ``probe_llm`` makes a real call, so
every test here injects a fake ``openai`` module — a diagnostic test that needs
the VPN would be exactly the failure it is meant to detect (CLAUDE.md §6).

The distinction the probe exists for: configuration can be entirely valid while
the LLM is unusable. The proxy in use here is VPN-only, so a correct key
against an unreachable host is the ordinary failure, and before this it looked
identical to "working" — classification quietly fell back to the shell
heuristic and the UI said nothing.
"""

from __future__ import annotations

import sys
import types

import pytest

from Final_app import config


@pytest.fixture(autouse=True)
def fresh_probe_cache():
    """The probe memoises for a minute; tests must not inherit each other's."""
    config._probe_cache.clear()
    yield
    config._probe_cache.clear()


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    """A valid-looking configuration, so probes reach the call itself."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://proxy.example/v1")


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, on_call):
    """Stand a fake ``openai`` module up so no request ever leaves the process.

    ``on_call`` runs in place of ``chat.completions.create``; raise from it to
    simulate a failure, return anything to simulate success.
    """
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return on_call(**kwargs)

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = _Chat()

    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


# ----------------------------------------------------------------------
# Configuration inspection
# ----------------------------------------------------------------------

def test_a_missing_key_is_reported_with_a_reason(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    diagnostics = config.llm_diagnostics()

    assert diagnostics["enabled"] is False
    assert any("OPENAI_API_KEY" in reason for reason in diagnostics["reasons"])


def test_a_configured_key_is_masked_never_echoed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-abcdefghijklmnop")
    diagnostics = config.llm_diagnostics()

    assert diagnostics["enabled"] is True
    assert "secret-abcdefghijkl" not in diagnostics["key_masked"]
    assert diagnostics["key_masked"].endswith("mnop")


# ----------------------------------------------------------------------
# The live probe
# ----------------------------------------------------------------------

def test_an_unconfigured_llm_fails_the_probe_without_calling_out(
    monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(**kwargs):
        raise AssertionError("probe must not call an unconfigured LLM")

    install_fake_openai(monkeypatch, forbidden)
    probe = config.probe_llm()

    assert probe["ok"] is False
    assert probe["configured"] is False
    assert "not configured" in probe["reason"]
    assert probe["hint"], "an unconfigured probe must say what is missing"


def test_a_working_llm_probes_ok(monkeypatch: pytest.MonkeyPatch, configured):
    install_fake_openai(monkeypatch, lambda **kwargs: object())
    probe = config.probe_llm()

    assert probe["ok"] is True
    assert probe["configured"] is True
    assert probe["reason"] == ""
    assert probe["model"] == config.JUDGE_MODEL


def test_the_probe_asks_for_a_single_token(monkeypatch: pytest.MonkeyPatch,
                                           configured):
    """A diagnostic that costs real money would not get run often enough."""
    calls = install_fake_openai(monkeypatch, lambda **kwargs: object())
    config.probe_llm()

    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 1
    assert calls[0]["model"] == config.JUDGE_MODEL


@pytest.mark.parametrize("error_name,message,expected", [
    ("APITimeoutError", "request timed out", "did not respond in time"),
    ("APIConnectionError", "connection refused", "could not be reached"),
    ("AuthenticationError", "401 Unauthorized", "key was rejected"),
    ("NotFoundError", "404 model not found", "was not found"),
    ("PermissionDeniedError", "403 forbidden", "not allowed"),
    ("RateLimitError", "429 slow down", "rate limiting"),
])
def test_each_failure_mode_gets_its_own_message(
    monkeypatch: pytest.MonkeyPatch, configured,
    error_name: str, message: str, expected: str
):
    """"It failed" is not a diagnostic — the reason has to point somewhere."""
    failure = type(error_name, (Exception,), {})

    def raiser(**kwargs):
        raise failure(message)

    install_fake_openai(monkeypatch, raiser)
    probe = config.probe_llm()

    assert probe["ok"] is False
    assert probe["configured"] is True
    assert expected in probe["reason"], probe["reason"]


def test_a_timeout_blames_the_vpn(monkeypatch: pytest.MonkeyPatch, configured):
    """The single most common failure here, so the hint must name it."""
    failure = type("APITimeoutError", (Exception,), {})

    def raiser(**kwargs):
        raise failure("timed out")

    install_fake_openai(monkeypatch, raiser)
    probe = config.probe_llm()

    assert "VPN" in probe["hint"]


def test_an_unexpected_failure_still_reports_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, configured
):
    """A diagnostic that can break the page it diagnoses is worse than none."""
    def raiser(**kwargs):
        raise ValueError("something nobody anticipated")

    install_fake_openai(monkeypatch, raiser)
    probe = config.probe_llm()

    assert probe["ok"] is False
    assert "ValueError" in probe["reason"]
    assert "something nobody anticipated" in probe["hint"]


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------

def test_the_result_is_reused_rather_than_re_probed(
    monkeypatch: pytest.MonkeyPatch, configured
):
    calls = install_fake_openai(monkeypatch, lambda **kwargs: object())

    first = config.probe_llm()
    second = config.probe_llm()

    assert len(calls) == 1, "the second call should have been served from cache"
    assert first["cached"] is False
    assert second["cached"] is True


def test_forcing_ignores_the_cache(monkeypatch: pytest.MonkeyPatch, configured):
    """The Retry button exists for the moment the VPN comes up."""
    calls = install_fake_openai(monkeypatch, lambda **kwargs: object())

    config.probe_llm()
    forced = config.probe_llm(force=True)

    assert len(calls) == 2
    assert forced["cached"] is False


def test_a_stale_result_is_re_probed(monkeypatch: pytest.MonkeyPatch, configured):
    calls = install_fake_openai(monkeypatch, lambda **kwargs: object())

    config.probe_llm()
    # Age the entry past its TTL rather than sleeping through it.
    config._probe_cache["at"] -= config.PROBE_TTL_S + 1
    config.probe_llm()

    assert len(calls) == 2
