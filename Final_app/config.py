"""Configuration and ``.env`` loading.

Mirrors ``Local_app/split_advisor.py``'s conventions so the same ``.env`` works
for both tools: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and a model variable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_ENV_LOAD_INFO: dict = {"dotenv": False, "found": [], "loaded": []}


def _load_project_env() -> None:
    """Fill unset vars from ``.env``, without ever clobbering a real value.

    We merge with ``dotenv_values`` rather than ``load_dotenv`` on purpose. A
    blank ``OPENAI_API_KEY=`` line (both templates ship one) loaded via
    ``load_dotenv(override=False)`` writes an empty string into the
    environment, which then blocks a real key from a later ``.env`` — and makes
    it look like the key "wasn't picked up". Here an empty value never
    overrides anything, so shell exports win and blank lines are inert.

    A missing ``python-dotenv`` is silent but recorded, because it is the most
    common reason a key in ``.env`` is never read.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    _ENV_LOAD_INFO["dotenv"] = True

    candidates = [
        Path(__file__).resolve().parent / ".env",   # next to the code
        Path.cwd() / ".env",                        # wherever it was launched
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        _ENV_LOAD_INFO["found"].append(str(path))
        applied = False
        for key, value in dotenv_values(path).items():
            if value and not os.environ.get(key):
                os.environ[key] = value
                applied = True
        if applied:
            _ENV_LOAD_INFO["loaded"].append(str(path))


_load_project_env()


# --- LLM (OpenAI-compatible: OpenAI, OpenRouter, a LiteLLM proxy, ...) -------

# TRACELENS_MODEL wins; SPLIT_ADVISOR_MODEL is honoured so an existing Local_app
# .env works unchanged.
JUDGE_MODEL = (
    os.environ.get("TRACELENS_MODEL")
    or os.environ.get("SPLIT_ADVISOR_MODEL")
    or "gpt-4o-mini"
)
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None -> SDK default

# Where Claude Code stores transcripts.
CLAUDE_CONFIG_DIR = os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude"))

# Cache for judge output, so re-opening a session costs nothing.
CACHE_DIR = Path(
    os.environ.get("TRACELENS_CACHE_DIR", Path.home() / ".cache" / "tracelens")
)

# Seconds before giving up on the LLM. Started at 20 — enough for the VPN-down
# case (calls hang ~16s with no VPN) and for a fast model like
# claude-haiku-4-5. A reasoning model changes the shape of the problem: timed
# real judge calls against ``openai/gpt-5.6-luna`` on 2026-08-25 ranged 7-40s
# and one legitimately needed 62s, all well past 20s. A judge call that times
# out is indistinguishable here from one that was never asked, so a timeout
# this short was silently discarding real answers, not just falling back
# cleanly.
LLM_TIMEOUT_S = float(os.environ.get("TRACELENS_LLM_TIMEOUT", "60"))


def llm_diagnostics() -> dict:
    """Why the LLM will or will not be used. Surfaced at ``/api/health``."""
    key = os.environ.get("OPENAI_API_KEY", "")
    try:
        import openai  # noqa: F401
        installed = True
    except ImportError:
        installed = False

    reasons: list[str] = []
    if not installed:
        reasons.append("the `openai` package is not installed (pip install openai)")
    if not key:
        reasons.append("OPENAI_API_KEY is not set")
        if not _ENV_LOAD_INFO["dotenv"]:
            reasons.append("python-dotenv is not installed, so .env was never read")
        elif not _ENV_LOAD_INFO["found"]:
            reasons.append("no .env file was found")

    return {
        "enabled": installed and bool(key),
        "model": JUDGE_MODEL,
        "base_url": LLM_BASE_URL or "(OpenAI SDK default)",
        "key_set": bool(key),
        "key_masked": (key[:6] + "..." + key[-4:]) if len(key) > 12 else ("set" if key else ""),
        "openai_installed": installed,
        "dotenv_installed": _ENV_LOAD_INFO["dotenv"],
        "env_files_found": _ENV_LOAD_INFO["found"],
        "env_files_loaded": _ENV_LOAD_INFO["loaded"],
        "reasons": reasons,
    }


# --- chat completions ---------------------------------------------------
#
# Reasoning-style models (observed with ``openai/gpt-5.6-luna`` on the litellm
# proxy here, same family as OpenAI's o-series) reject any ``temperature``
# other than the default (1) with a 400. Every other model on the proxy
# accepts an explicit ``temperature`` fine, so this only pays the retry cost
# when the host actually objects, rather than hardcoding a model list that
# would need updating for every future reasoning model.

def chat_completion(client, *, model: str, messages: list, max_tokens: int,
                     temperature: float | None = None):
    """``client.chat.completions.create``, retried without ``temperature`` if rejected."""
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as error:
        if temperature is None or "temperature" not in str(error).lower():
            raise
        kwargs.pop("temperature")
        return client.chat.completions.create(**kwargs)


# --- live probe --------------------------------------------------------
#
# ``llm_diagnostics`` only reads configuration. Configuration can be perfectly
# valid and the LLM still unusable — the LiteLLM proxy here is VPN-only, so the
# common failure is a correct key against an unreachable host. That looked
# identical to "working" in the UI: blocks quietly fell back to the shell
# heuristic and nothing said why.
#
# The probe is the judge's own call shape (a chat completion against
# ``JUDGE_MODEL``), so it exercises the key, the host, *and* the model name.
# ``max_tokens=16`` keeps it cheap. It cannot be 1: reasoning-style models
# (observed with ``openai/gpt-5.6-luna``) spend part of the token budget on
# hidden reasoning tokens before any visible output, so ``max_tokens=1`` left
# nothing for the answer and the probe always failed with "max_tokens...
# reached" even though the model and key were both fine.

# Deliberately shorter than LLM_TIMEOUT_S: this runs while someone waits for a
# banner, not while a session is being classified.
PROBE_TIMEOUT_S = float(os.environ.get("TRACELENS_PROBE_TIMEOUT", "8"))

# Re-probing on every page load would add a round trip to every navigation for
# a state that changes only when a VPN goes up or down.
PROBE_TTL_S = 60.0

_probe_cache: dict = {}


def _describe_probe_failure(error: Exception) -> tuple[str, str]:
    """``(short reason, what to do about it)`` for a failed probe.

    Mapped by exception name rather than by class, so this module never has to
    import ``openai`` just to catch its errors.
    """
    name = type(error).__name__
    text = str(error)

    if "Timeout" in name or "timed out" in text.lower():
        return ("the LLM did not respond in time",
                "The proxy is usually reachable only over VPN — check you are connected.")
    if "Connection" in name or "APIConnectionError" in name:
        return ("the LLM host could not be reached",
                "Check OPENAI_BASE_URL and, if this is an internal proxy, your VPN.")
    if "Authentication" in name or "401" in text:
        return ("the API key was rejected",
                "OPENAI_API_KEY is set but not accepted by this host.")
    if "NotFound" in name or "404" in text:
        return (f"the model {JUDGE_MODEL!r} was not found",
                "Set TRACELENS_MODEL to a model this host serves.")
    if "PermissionDenied" in name or "403" in text:
        return ("this key is not allowed to use that model",
                f"The host refused {JUDGE_MODEL!r} for this key.")
    if "RateLimit" in name or "429" in text:
        return ("the LLM is rate limiting", "Try again shortly.")
    return (f"the LLM call failed ({name})", text[:200])


def probe_llm(force: bool = False) -> dict:
    """Actually call the LLM. Returns ``{ok, reason, hint, latency_ms, ...}``.

    Never raises: a diagnostic that can break the page it diagnoses is worse
    than no diagnostic.
    """
    now = time.monotonic()
    cached = _probe_cache.get("result")
    if not force and cached and now - _probe_cache["at"] < PROBE_TTL_S:
        return dict(cached, cached=True)

    config = llm_diagnostics()
    if not config["enabled"]:
        result = {
            "ok": False,
            "configured": False,
            "reason": "the LLM is not configured",
            "hint": "; ".join(config["reasons"]),
            "latency_ms": 0,
            "model": JUDGE_MODEL,
        }
        _probe_cache.update(result=result, at=now)
        return dict(result, cached=False)

    started = time.monotonic()
    try:
        from openai import OpenAI

        client = (OpenAI(base_url=LLM_BASE_URL, timeout=PROBE_TIMEOUT_S)
                  if LLM_BASE_URL else OpenAI(timeout=PROBE_TIMEOUT_S))
        chat_completion(
            client,
            model=JUDGE_MODEL,
            max_tokens=16,
            temperature=0,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as error:  # noqa: BLE001 - reported, never raised
        reason, hint = _describe_probe_failure(error)
        result = {
            "ok": False,
            "configured": True,
            "reason": reason,
            "hint": hint,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "model": JUDGE_MODEL,
        }
    else:
        result = {
            "ok": True,
            "configured": True,
            "reason": "",
            "hint": "",
            "latency_ms": round((time.monotonic() - started) * 1000),
            "model": JUDGE_MODEL,
        }

    _probe_cache.update(result=result, at=now)
    return dict(result, cached=False)
