"""Limits, paths and thresholds for the data-preprocessing phase.

Kept separate from the app-wide ``config.Settings`` so the API side can compose it
without either module editing the other's fields. All variables use the ``INGEST_``
prefix, except ``TESSDATA_PREFIX``, which is Tesseract's own standard name and is
already set in the Dockerfile.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_MB = 1024 * 1024


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        extra="ignore",
        populate_by_name=True,
    )

    # OCR
    tessdata_prefix: Path = Field(
        default=Path("/app/models/tessdata"), validation_alias="TESSDATA_PREFIX"
    )
    ocr_languages: list[str] = ["ara", "fra", "eng"]
    ocr_min_chars_per_page: int = Field(default=32, ge=0)

    # File limits
    max_file_mb: int = Field(default=25, gt=0)
    max_pages: int = Field(default=300, gt=0)
    parse_timeout_s: int = Field(default=60, gt=0)  # hang guard per document
    ocr_page_timeout_s: int = Field(default=10, ge=0)  # extra budget per PDF page / image
    max_concurrency: int = Field(default=1, gt=0)

    # Web fetching
    web_timeout_s: float = Field(default=15, gt=0)
    web_max_bytes: int = Field(default=5 * _MB, gt=0)
    web_max_redirects: int = Field(default=5, ge=0)
    web_user_agent: str = "mcclub-rag/0.1 (+document ingestion)"

    # Language detection
    lang_codes: list[str] = ["ar", "fr", "en"]
    lang_min_chars: int = Field(default=20, ge=0)
    lang_min_prob: float = Field(default=0.5, ge=0, le=1)
    lang_mixed_share: float = Field(default=0.2, gt=0, le=1)
    lang_default: Literal["ar", "fr", "en"] = "fr"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * _MB


@lru_cache
def get_ingest_settings() -> IngestSettings:
    return IngestSettings()
