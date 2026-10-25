"""Model-selected ElevenLabs voices: the ``voice_id`` field on ``text_to_speech`` and the
``tts_voices`` browse/search tool exist only while ElevenLabs is the configured provider."""

import json
from unittest.mock import patch

import httpx
import pytest

from tools import tts_tool, tts_voices_tool

_EL_CONFIG = {"provider": "elevenlabs", "elevenlabs": {"voice_id": "configuredVoice001"}}


@pytest.mark.parametrize("config, exposed", [
    (_EL_CONFIG, True), ({"provider": "edge"}, False), ({"provider": "openai"}, False)])
def test_voice_selection_surface_follows_provider(config, exposed):
    with patch("tools.tts_tool._load_tts_config", return_value=config), \
         patch("tools.tts_tool._resolve_provider_key", return_value="el-test-key"):
        schema = tts_tool._tts_schema_overrides()
        assert ("voice_id" in schema["parameters"]["properties"]) is exposed
        assert ("tts_voices" in schema.get("description", "")) is exposed
        assert tts_voices_tool.check_tts_voices_requirements() is exposed
    with patch("tools.tts_tool._load_tts_config", return_value=_EL_CONFIG), \
         patch("tools.tts_tool._resolve_provider_key", return_value=""):
        assert tts_voices_tool.check_tts_voices_requirements() is False


def test_voice_id_reaches_elevenlabs_and_is_ignored_elsewhere(tmp_path):
    seen = []

    class _FakeElevenLabs:  # the SDK client: records the voice the real generator sends
        def __init__(self, **_kwargs):
            self.text_to_speech = self

        def convert(self, *, voice_id, **_kwargs):
            seen.append(voice_id)
            return [b"ID3\x03\x00\x00\x00\x00\x00\x00"]

    with patch("tools.tts_tool._import_elevenlabs", return_value=_FakeElevenLabs), \
         patch("tools.tts_tool._resolve_provider_key", return_value="el-test-key"), \
         patch("tools.tts_tool._load_tts_config", return_value=_EL_CONFIG):
        ok = json.loads(tts_tool.text_to_speech_tool(
            "Hail, traveler.", output_path=str(tmp_path / "a.mp3"), voice_id="pickedVoice0042"))
        default = json.loads(tts_tool.text_to_speech_tool(
            "Hail, traveler.", output_path=str(tmp_path / "b.mp3")))
        bad = json.loads(tts_tool.text_to_speech_tool(
            "Hail.", output_path=str(tmp_path / "c.mp3"), voice_id="../v1/user"))
    assert ok["success"] and default["success"] and not bad["success"]
    assert seen == ["pickedVoice0042", "configuredVoice001"]
    assert _EL_CONFIG["elevenlabs"]["voice_id"] == "configuredVoice001"  # cached config untouched

    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
        config, error = tts_tool._apply_voice_override({"provider": "edge"}, "edge", "pickedVoice0042")
    assert error is None and "elevenlabs" not in config


@pytest.mark.parametrize("source, path, first_page, cursor_param, body, next_page", [
    ("account", "/v2/voices", "", "next_page_token",
     {"voices": [{"voice_id": "acct1", "name": "Bram", "category": "premade",
                  "labels": {"gender": "male", "age": "old", "accent": "scottish",
                             "use case": "characters"},
                  "preview_url": "https://example.test/bram.mp3"}],
      "has_more": True, "next_page_token": "tok-2"}, "tok-2"),
    ("library", "/v1/shared-voices", "3", "page",
     {"voices": [{"voice_id": "lib1", "name": "Gruff", "gender": "male", "age": "old",
                  "accent": "scottish", "descriptive": "gruff", "use_case": "characters",
                  "category": "professional", "description": "x" * 400}],
      "has_more": True, "total_count": 812}, "4"),
])
def test_tts_voices_pages_both_sources(source, path, first_page, cursor_param, body, next_page):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params or {}), headers))
        return httpx.Response(200, json=body)

    with patch("tools.tts_tool._load_tts_config", return_value=_EL_CONFIG), \
         patch("tools.tts_tool._resolve_provider_key", return_value="el-test-key"), \
         patch("tools.tts_voices_tool.httpx.get", side_effect=fake_get):
        result = json.loads(tts_voices_tool.tts_voices_tool(
            query="gruff dwarf", source=source, gender="Male", page=first_page, page_size=500))
    url, params, headers = calls[0]
    assert url == "https://api.elevenlabs.io" + path
    assert headers["xi-api-key"] == "el-test-key"
    assert params["search"] == "gruff dwarf" and params["gender"] == "male"
    assert params["page_size"] == 100
    if first_page:
        assert str(params[cursor_param]) == first_page
    row = result["voices"][0]
    assert result["success"] and result["next_page"] == next_page
    assert row["gender"] == "male" and row["accent"] == "scottish" and row["use_case"] == "characters"
    assert result["configured_voice_id"] == "configuredVoice001"
    if source == "library":
        assert result["total"] == 812 and len(row["description"]) == 240


def test_tts_voices_rejects_a_non_numeric_library_page():
    with patch("tools.tts_tool._load_tts_config", return_value=_EL_CONFIG), \
         patch("tools.tts_tool._resolve_provider_key", return_value="el-test-key"), \
         patch("tools.tts_voices_tool.httpx.get") as get:
        result = json.loads(tts_voices_tool.tts_voices_tool(source="library", page="tok-2"))
    assert not result["success"] and get.call_count == 0
