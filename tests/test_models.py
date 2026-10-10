from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mcclub_rag.ingest.models import ParsedDocument, Section


def _doc(**overrides) -> ParsedDocument:
    fields = {
        "content_hash": "ab" * 32,
        "title": "Club rules",
        "source": "rules.pdf",
        "source_type": "file",
        "mime_type": "application/pdf",
        "visibility": "public",
        "language": "fr",
        "text": "Bonjour",
        "sections": [Section(text="Bonjour", heading="Intro", page=1, language="fr")],
        "ocr_used": False,
        "parser": "xberg 1.3.6",
        "page_count": 1,
        "created_at": datetime(2026, 10, 9, tzinfo=UTC),
    }
    fields.update(overrides)
    return ParsedDocument(**fields)


@pytest.mark.parametrize("visibility", ["public", "members"])
def test_valid_visibility_is_kept(visibility):
    assert _doc(visibility=visibility).visibility == visibility


@pytest.mark.parametrize("visibility", ["admin", "member", "", "PUBLIC"])
def test_invalid_visibility_rejected(visibility):
    with pytest.raises(ValidationError):
        _doc(visibility=visibility)


def test_invalid_source_type_rejected():
    with pytest.raises(ValidationError):
        _doc(source_type="ftp")


def test_models_are_frozen():
    doc = _doc()
    with pytest.raises(ValidationError):
        doc.title = "changed"
    with pytest.raises(ValidationError):
        doc.sections[0].text = "changed"


def test_warnings_default_to_empty_list():
    assert _doc().warnings == ()


def test_json_round_trip():
    doc = _doc(warnings=["1 OCR line removed"])
    assert ParsedDocument.model_validate_json(doc.model_dump_json()) == doc


def test_sequences_are_immutable_tuples():
    doc = _doc()
    assert isinstance(doc.sections, tuple)
    assert isinstance(doc.warnings, tuple)
