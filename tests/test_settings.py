import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("INGEST_") or key == "TESSDATA_PREFIX":
            monkeypatch.delenv(key)
    get_ingest_settings.cache_clear()
    yield
    get_ingest_settings.cache_clear()


def test_defaults_match_design():
    s = IngestSettings()
    assert s.tessdata_prefix == Path("/app/models/tessdata")
    assert s.ocr_languages == ["ara", "fra", "eng"]
    assert s.ocr_min_chars_per_page == 32
    assert s.max_file_mb == 25
    assert s.max_file_bytes == 25 * 1024 * 1024
    assert s.max_pages == 300
    assert s.parse_timeout_s == 60
    assert s.ocr_page_timeout_s == 10
    assert s.max_concurrency == 1
    assert s.web_timeout_s == 15
    assert s.web_max_bytes == 5 * 1024 * 1024
    assert s.web_max_redirects == 5
    assert s.web_user_agent.startswith("mcclub-rag/")
    assert s.lang_codes == ["ar", "fr", "en"]
    assert s.lang_min_chars == 20
    assert s.lang_min_prob == 0.5
    assert s.lang_mixed_share == 0.2
    assert s.lang_default == "fr"


def test_prefixed_env_overrides_default(monkeypatch):
    monkeypatch.setenv("INGEST_MAX_PAGES", "5")
    monkeypatch.setenv("INGEST_WEB_TIMEOUT_S", "2.5")
    s = IngestSettings()
    assert s.max_pages == 5
    assert s.web_timeout_s == 2.5


def test_list_env_override_uses_json(monkeypatch):
    monkeypatch.setenv("INGEST_OCR_LANGUAGES", '["eng"]')
    assert IngestSettings().ocr_languages == ["eng"]


def test_unprefixed_name_is_ignored(monkeypatch):
    monkeypatch.setenv("MAX_PAGES", "5")
    assert IngestSettings().max_pages == 300


def test_tessdata_prefix_read_from_standard_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert IngestSettings().tessdata_prefix == tmp_path


def test_tessdata_prefix_settable_by_field_name(tmp_path):
    assert IngestSettings(tessdata_prefix=tmp_path).tessdata_prefix == tmp_path


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("INGEST_MAX_PAGES", "0"),
        ("INGEST_MAX_FILE_MB", "-1"),
        ("INGEST_LANG_DEFAULT", "de"),
        ("INGEST_LANG_MIXED_SHARE", "1.5"),
        ("INGEST_LANG_MIN_PROB", "-0.1"),
    ],
)
def test_invalid_values_rejected(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        IngestSettings()


def test_get_ingest_settings_is_cached():
    assert get_ingest_settings() is get_ingest_settings()


def test_chunk_defaults_match_design(monkeypatch):
    monkeypatch.delenv("MODELS_DIR", raising=False)
    s = IngestSettings()
    assert s.chunk_target_tokens == 300
    assert s.chunk_max_tokens == 400
    assert s.chunk_min_tokens == 80
    assert s.chunk_max_header_tokens == 48
    assert s.chunk_context_header is True
    assert s.chunk_tokenizer_path == Path("/app/models/embedder/tokenizer.json")


def test_chunk_tokenizer_path_follows_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert IngestSettings().chunk_tokenizer_path == tmp_path / "embedder" / "tokenizer.json"


def test_chunk_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_CHUNK_MAX_TOKENS", "500")
    monkeypatch.setenv("INGEST_CHUNK_CONTEXT_HEADER", "false")
    monkeypatch.setenv("INGEST_CHUNK_TOKENIZER_PATH", str(tmp_path / "t.json"))
    s = IngestSettings()
    assert s.chunk_max_tokens == 500
    assert s.chunk_context_header is False
    assert s.chunk_tokenizer_path == tmp_path / "t.json"


@pytest.mark.parametrize(
    "fields",
    [
        {"chunk_min_tokens": 300},  # min == target
        {"chunk_min_tokens": 350},  # min > target
        {"chunk_target_tokens": 450},  # target > max
        {"chunk_max_header_tokens": 100},  # header >= max / 4
        {"chunk_min_tokens": 0},
    ],
)
def test_chunk_size_ordering_enforced(fields):
    with pytest.raises(ValidationError):
        IngestSettings(**fields)
