"""Exact per-content-block token counts, so every tool call carries its own.

Why this exists
---------------
``usage`` in a transcript is reported once per assistant *message*, never per
content block (the API bills per request, and a message's prose and its tool
calls are generated in one pass). The adapter therefore *divides* that one
figure across the message's Events — exact in total, an estimate per Event.

That estimate is not good enough to answer "how big was this Read". So this
module measures each content block directly with Anthropic's ``count_tokens``,
which tokenises a block on its own.

The number this produces is a different number from the billed one, and the
difference is not a bug in either. Measured on 24 single-block, no-thinking
messages from real transcripts, where the whole billed ``output`` provably
belongs to one visible block:

    tool_use   count_tokens=2564   billed_output=3311   ratio 1.29
    tool_use   count_tokens=  31   billed_output=  56   ratio 1.81
    text       count_tokens= 141   billed_output= 171   ratio 1.21
                                    median 1.47, range 1.23-2.74

Billed output exceeds the tokens of the content on disk because it also covers
things the transcript never stores — unsaved thinking, response framing. So:

  * ``Event.tokens``          — share of what you were **billed**. Sums exactly
                                to the session total. Use it for cost.
  * ``Event.content_tokens``  — the **measured size** of this one block. Exact
                                for the block. Does *not* sum to the bill.

Both are kept, and the UI must never present one as the other.

Degrading
---------
Counting needs the LiteLLM proxy, which is VPN-only. Everything here returns
0 rather than raising, and 0 means "not counted yet" — never "empty". Counts
are cached on disk by content hash, so a block is measured once ever and the
VPN is only needed the first time a given call is seen.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from Final_app.config import CACHE_DIR, LLM_BASE_URL, LLM_TIMEOUT_S
from Final_app.ir.models import EV_ASSISTANT, EV_TOOL_USE, EV_USER, Event

# ----------------------------------------------------------------------
# Which model's tokeniser
# ----------------------------------------------------------------------

# Any Claude model shares a tokeniser, so this is not the judge model and does
# not need to track it. Pinned so a cached count never silently means a
# different tokeniser than the one that produced it.
COUNT_MODEL = os.environ.get("TRACELENS_COUNT_MODEL", "claude-haiku-4-5")

# `count_tokens` lives on the Anthropic-native path, not under /v1 (verified
# against the LiteLLM proxy: `/v1` returns 404, the bare host works).
def _count_base_url() -> Optional[str]:
    if not LLM_BASE_URL:
        return None
    return LLM_BASE_URL.rstrip("/").removesuffix("/v1") or None


# ----------------------------------------------------------------------
# Disk cache
# ----------------------------------------------------------------------

_CACHE_FILE = CACHE_DIR / "content_tokens.json"
_cache: Optional[dict] = None
_cache_lock = threading.Lock()
_dirty = False


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        loaded = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        _cache = loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        _cache = {}
    return _cache


def save_cache() -> None:
    """Flush counts to disk. Safe to call when nothing changed."""
    global _dirty
    with _cache_lock:
        if not _dirty:
            return
        cache = _load_cache()
        _dirty = False
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        tmp.replace(_CACHE_FILE)
    except OSError:
        # A cache we cannot write is a slow cache, not a broken dashboard.
        pass


def cache_key(block: dict) -> str:
    """Hash of the content block itself, so identical calls are counted once.

    Keyed on content rather than on event uuid deliberately: the same
    ``Read /repo/a.py`` recurs across sessions, and one measurement serves all
    of them. ``sort_keys`` so dict ordering never splits the key.
    """
    payload = json.dumps([block, COUNT_MODEL], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------------------------
# Event -> the content block the API saw
# ----------------------------------------------------------------------

def content_block(event: Event) -> Optional[dict]:
    """Rebuild the API content block this Event came from, or None.

    ``tool_use`` is rebuilt with its ``id``/``name``/``input`` because those are
    what the model emitted and therefore what it was charged for. A user prompt
    is counted as the text block it is — that is what gives a "chatting with
    user" block a real number, which the billed figures cannot (a user line
    carries no ``usage`` at all: 0 of 268 in a checked transcript).
    """
    if event.type == EV_TOOL_USE and event.tool is not None:
        return {
            "type": "tool_use",
            "id": event.tool.id,
            "name": event.tool.name,
            "input": event.tool.input,
        }
    if event.type in (EV_ASSISTANT, EV_USER) and event.text.strip():
        return {"type": "text", "text": event.text}
    return None


def _role_for(event: Event) -> str:
    return "user" if event.type == EV_USER else "assistant"


# ----------------------------------------------------------------------
# Counting
# ----------------------------------------------------------------------

def counting_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    import anthropic

    return anthropic.Anthropic(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=_count_base_url(),
        timeout=LLM_TIMEOUT_S,
        max_retries=1,
    )


# The framing a one-message request costs regardless of content. Measured once
# per process and subtracted, so a 3-token tool call does not come back as 20.
_overhead: Optional[int] = None
_overhead_lock = threading.Lock()


def _framing_overhead(client) -> int:
    global _overhead
    with _overhead_lock:
        if _overhead is not None:
            return _overhead
    probe = client.messages.count_tokens(
        model=COUNT_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": ""}]}],
    ).input_tokens
    with _overhead_lock:
        _overhead = probe
    return probe


def _count_one(client, event: Event, block: dict) -> int:
    total = client.messages.count_tokens(
        model=COUNT_MODEL,
        messages=[{"role": _role_for(event), "content": [block]}],
    ).input_tokens
    return max(0, total - _framing_overhead(client))


def cached_counts(events: list[Event]) -> int:
    """Fill ``content_tokens`` from cache only. Returns how many are still 0.

    Never calls out, so a page render can use it unconditionally.
    """
    cache = _load_cache()
    pending = 0
    for event in events:
        block = content_block(event)
        if block is None:
            continue
        hit = cache.get(cache_key(block))
        if isinstance(hit, int):
            event.content_tokens = hit
        else:
            pending += 1
    return pending


def count_events(events: list[Event], max_workers: int = 8) -> int:
    """Measure every uncounted Event, in parallel. Returns how many were added.

    Concurrency matters: a 421-event session is 421 round trips, and serially
    at ~200ms each that is a minute and a half. Failures are per-event — one
    timeout leaves that block at 0 and the rest still get counted.
    """
    global _dirty

    todo: list[tuple[Event, dict, str]] = []
    cache = _load_cache()
    for event in events:
        block = content_block(event)
        if block is None:
            continue
        key = cache_key(block)
        hit = cache.get(key)
        if isinstance(hit, int):
            event.content_tokens = hit
        else:
            todo.append((event, block, key))

    if not todo or not counting_available():
        return 0

    try:
        client = _client()
    except Exception:
        return 0

    # Distinct content only: a session that reads the same file eight times is
    # eight events but one measurement.
    by_key: dict[str, tuple[Event, dict]] = {}
    for event, block, key in todo:
        by_key.setdefault(key, (event, block))

    counted: dict[str, int] = {}

    def measure(item: tuple[str, tuple[Event, dict]]) -> None:
        key, (event, block) = item
        try:
            counted[key] = _count_one(client, event, block)
        except Exception:
            # Left uncounted; the UI shows the billed estimate instead.
            pass

    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="count") as pool:
        list(pool.map(measure, by_key.items()))

    if counted:
        with _cache_lock:
            _load_cache().update(counted)
            _dirty = True
        save_cache()

    for event, block, key in todo:
        if key in counted:
            event.content_tokens = counted[key]
    return len(counted)
