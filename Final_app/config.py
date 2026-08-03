"""Configuration and ``.env`` loading.

Mirrors ``Local_app/split_advisor.py``'s conventions so the same ``.env`` works
for both tools: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and a model variable.
"""

from __future__ import annotations

import os
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

# Seconds before giving up on the LLM. The proxy needs VPN; without it calls
# hang ~16s. A short timeout keeps the UI responsive and falls back cleanly.
LLM_TIMEOUT_S = float(os.environ.get("TRACELENS_LLM_TIMEOUT", "20"))


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
