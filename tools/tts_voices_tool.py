"""``tts_voices``: browse and search ElevenLabs voices so the agent can pick one per
``text_to_speech`` call (``voice_id``).

Registered in the ``tts`` toolset but only exposed while ``tts.provider`` is ElevenLabs with a
key, the same gate as the ``voice_id`` schema field. Two sources share one paginated shape:
``account`` (``GET /v2/voices``: premade + the user's saved, cloned and designed voices) and
``library`` (``GET /v1/shared-voices``: the public Voice Library). ``next_page`` is opaque to the
model: a cursor token for the account and a page number for the library.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from tools.registry import no_cache_check_fn, registry, tool_error

logger = logging.getLogger(__name__)

DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
_SOURCES = ("account", "library")
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100
_DESCRIPTION_MAX_CHARS = 240
_TIMEOUT_SECONDS = 20.0
_FILTERS = ("gender", "age", "accent", "language")


@no_cache_check_fn  # config-backed and local: a provider switch must show/hide it at once
def check_tts_voices_requirements() -> bool:
    """Exposed only when ElevenLabs is the configured TTS provider and its key resolves."""
    from tools import tts_tool
    return (tts_tool.elevenlabs_voice_selection_active()
            and bool(tts_tool._resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")))


def _clip(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _DESCRIPTION_MAX_CHARS else text[: _DESCRIPTION_MAX_CHARS - 1] + "…"


def _voice_row(voice: dict[str, Any]) -> dict[str, Any]:
    """One compact voice: library rows carry the traits at top level, account rows under ``labels``."""
    raw_labels = voice.get("labels")
    labels: dict[str, Any] = raw_labels if isinstance(raw_labels, dict) else {}
    row = {"voice_id": voice.get("voice_id"), "name": voice.get("name")}
    for key in ("gender", "age", "accent", "language", "descriptive", "use_case"):
        value = voice.get(key) or labels.get(key) or labels.get(key.replace("_", " "))
        if value:
            row[key] = value
    if voice.get("category"):
        row["category"] = voice["category"]
    if voice.get("description"):
        row["description"] = _clip(voice["description"])
    if voice.get("preview_url"):
        row["preview_url"] = voice["preview_url"]
    return row


def _request_params(source: str, query: str, filters: dict[str, str], page: str,
                    page_size: int) -> tuple[str, dict[str, Any]]:
    """``(path, params)`` for one source; raises ValueError on a cursor of the wrong kind."""
    params: dict[str, Any] = {"page_size": page_size, **filters}
    if query:
        params["search"] = query
    if source == "account":
        params["include_total_count"] = "false"
        if page:
            params["next_page_token"] = page
        return "/v2/voices", params
    if page and not page.isdigit():
        raise ValueError(f"Invalid library page {page!r}: pass next_page from the previous result.")
    params["page"] = int(page or 0)
    return "/v1/shared-voices", params


def _next_page(source: str, body: dict[str, Any], page: str) -> Optional[str]:
    if not body.get("has_more"):
        return None
    if source == "account":
        return body.get("next_page_token") or None
    return str(int(page or 0) + 1)


def tts_voices_tool(query: str = "", source: str = "account", gender: str = "", age: str = "",
                    accent: str = "", language: str = "", page: str = "",
                    page_size: Optional[int] = None) -> str:
    """Search one ElevenLabs voice source and return a compact, paginated JSON page."""
    from tools import tts_tool
    source = (source or "account").strip().lower()
    if source not in _SOURCES:
        return tool_error(f"source must be one of {', '.join(_SOURCES)}", success=False)
    tts_config = tts_tool._load_tts_config()
    if not tts_tool.elevenlabs_voice_selection_active(tts_config):
        return tool_error("tts_voices needs tts.provider: elevenlabs", success=False)
    api_key = tts_tool._resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
    if not api_key:
        return tool_error("ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/", success=False)
    try:
        size = max(1, min(_MAX_PAGE_SIZE, int(page_size or _DEFAULT_PAGE_SIZE)))
    except (TypeError, ValueError):
        size = _DEFAULT_PAGE_SIZE
    filters = {}
    for name, value in zip(_FILTERS, (gender, age, accent, language)):
        if str(value or "").strip():
            filters[name] = str(value).strip().lower()
    page = str(page or "").strip()
    try:
        path, params = _request_params(source, str(query or "").strip(), filters, page, size)
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    el_config = tts_config.get("elevenlabs") or {}
    base_url = (el_config.get("base_url") or DEFAULT_ELEVENLABS_BASE_URL).rstrip("/")
    try:
        response = httpx.get(base_url + path, params=params, timeout=_TIMEOUT_SECONDS,
                             headers={"xi-api-key": api_key, "accept": "application/json"})
    except httpx.HTTPError as exc:
        return tool_error(f"ElevenLabs voice search failed: {exc}", success=False)
    if response.status_code != 200:
        detail = response.text[:300]
        logger.warning("ElevenLabs %s returned HTTP %s", path, response.status_code)
        return tool_error(f"ElevenLabs voice search failed (HTTP {response.status_code}): {detail}",
                          success=False)
    try:
        body = response.json()
    except ValueError:
        return tool_error("ElevenLabs voice search returned a non-JSON body", success=False)
    voices = [_voice_row(v) for v in body.get("voices") or [] if isinstance(v, dict)]
    result: dict[str, Any] = {
        "success": True, "source": source, "count": len(voices), "voices": voices,
        "next_page": _next_page(source, body, page),
        "configured_voice_id": el_config.get("voice_id") or None,
    }
    if source == "library" and body.get("total_count"):
        result["total"] = body["total_count"]
    return json.dumps(result, ensure_ascii=False)


TTS_VOICES_SCHEMA = {
    "name": "tts_voices",
    "description": (
        "Search ElevenLabs voices to pick a voice_id for text_to_speech. source 'account' = the "
        "user's voices (premade, saved, cloned, designed); 'library' = the public Voice Library "
        "(thousands of community voices; paid plans only). Paginated: pass next_page back as page. "
        "You can't hear voices, so choose from the labels and description, and offer the user "
        "preview_url links when the pick matters."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Free-text search over names, descriptions and labels, e.g. 'gruff old dwarf'."},
            "source": {"type": "string", "enum": list(_SOURCES), "description": "Default account."},
            "gender": {"type": "string", "description": "Filter, e.g. male, female, neutral."},
            "age": {"type": "string", "description": "Filter, e.g. young, middle_aged, old."},
            "accent": {"type": "string", "description": "Filter, e.g. british, american, scottish."},
            "language": {"type": "string", "description": "Filter by ISO 639-1 code, e.g. en."},
            "page": {"type": "string", "description": "next_page from the previous result."},
            "page_size": {"type": "integer", "description": "1-100, default 20."},
        },
        "required": [],
    },
}

registry.register(
    name="tts_voices",
    toolset="tts",
    schema=TTS_VOICES_SCHEMA,
    handler=lambda args, **kw: tts_voices_tool(
        **{k: args.get(k) for k in ("query", "source", "gender", "age", "accent", "language",
                                    "page", "page_size") if args.get(k) is not None}),
    check_fn=check_tts_voices_requirements,
    emoji="🗣️")
