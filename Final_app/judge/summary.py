"""LLM-as-a-judge session summaries.

One short sentence per session, from the user's prompts — the same shape as
``Local_app``'s ``build_summary_prompt``. Results are cached on disk keyed by
the prompt content, so a session is summarised once.

Everything here degrades to "" rather than raising: the dashboard must work
with no key, no network, and no VPN (the LiteLLM proxy is VPN-only and calls
time out at ~16s without it).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from Final_app.config import CACHE_DIR, JUDGE_MODEL, LLM_BASE_URL, LLM_TIMEOUT_S

_CACHE_FILE = CACHE_DIR / "summaries.json"
_cache: Optional[dict] = None

# Enough prompts to characterise the session without paying for the whole thing.
MAX_PROMPTS = 20
MAX_PROMPT_CHARS = 400


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(_cache, dict):
            _cache = {}
    except (OSError, json.JSONDecodeError):
        _cache = {}
    return _cache


def _save_cache() -> None:
    cache = _load_cache()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        tmp.replace(_CACHE_FILE)
    except OSError:
        pass


def _trim(prompts: list[str]) -> list[str]:
    out = []
    for p in prompts[:MAX_PROMPTS]:
        p = " ".join(p.split())
        out.append(p[:MAX_PROMPT_CHARS])
    return out


def cache_key(prompts: list[str]) -> str:
    payload = json.dumps([_trim(prompts), JUDGE_MODEL], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def build_summary_prompt(prompts: list[str]) -> str:
    """Ask for one concrete sentence. Pure string; no API call."""
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(_trim(prompts)))
    return (
        "Here are the user's messages from one coding session, in order:\n\n"
        f"{numbered}\n\n"
        "In ONE short, concrete sentence, describe what task the user worked on "
        "in this session (name the feature/bug/goal, not just 'coding').\n\n"
        "Answer with ONLY a JSON object, no other text:\n"
        '{"summary": "<one sentence>"}'
    )


def llm_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def cached_summary(prompts: list[str]) -> str:
    """The cached summary, or "" — never calls out."""
    if not prompts:
        return ""
    return _load_cache().get(cache_key(prompts), "")


def summarize(prompts: list[str], force: bool = False) -> str:
    """Summarise a session, using the cache. Returns "" on any failure."""
    if not prompts:
        return ""
    key = cache_key(prompts)
    cache = _load_cache()
    if not force and key in cache:
        return cache[key]
    if not llm_available():
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_S) if LLM_BASE_URL \
            else OpenAI(timeout=LLM_TIMEOUT_S)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=120,
            temperature=0,
            messages=[{"role": "user", "content": build_summary_prompt(prompts)}],
        )
        text = (response.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        summary = (json.loads(text).get("summary") or "").strip()
    except Exception:
        return ""

    if summary:
        cache[key] = summary
        _save_cache()
    return summary
