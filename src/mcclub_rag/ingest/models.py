"""Output contract of the preprocessing phase, plus internal extraction types."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Visibility = Literal["public", "members"]
SourceType = Literal["file", "web"]
SourceKind = Literal["pdf", "docx", "xlsx", "image", "markdown", "text", "html"]


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    heading: str | None = None
    page: int | None = None  # 1-based; None when the format has no pages
    language: str | None = None  # ISO 639-1 or "und"


class ParsedDocument(BaseModel):
    """One preprocessed document, ready for chunking.

    ``doc_id`` is deliberately absent: ``ingest/pipeline.py`` derives it from the topic.
    """

    model_config = ConfigDict(frozen=True)

    content_hash: str  # sha256 hex of raw bytes (files) / normalized text (web)
    title: str
    source: str  # original filename, or final URL after redirects
    source_type: SourceType
    mime_type: str
    visibility: Visibility
    language: str  # "ar" | "fr" | "en" | "mixed" | "und"
    text: str  # normalized, sections joined by blank lines
    sections: tuple[Section, ...]
    ocr_used: bool
    parser: str  # e.g. "xberg 1.3.6", "trafilatura 2.3.1", "text"
    page_count: int | None
    warnings: tuple[str, ...] = ()
    created_at: datetime  # UTC


@dataclass(frozen=True)
class RawPage:
    """One page (PDF/image), sheet (XLSX) or the whole body (DOCX) as extracted."""

    number: int
    content: str
    sheet_name: str | None = None
    tables: tuple[tuple[tuple[str, ...], ...], ...] = ()  # tables -> rows -> cells
    ocr: bool = False


@dataclass
class RawExtraction:
    """Extractor output before sectioning and normalization (internal)."""

    kind: SourceKind
    mime_type: str
    parser: str
    markdown: str
    pages: list[RawPage] = field(default_factory=list)
    metadata_title: str | None = None
    ocr_used: bool = False
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
