import re

from mcclub_rag.ingest.chunk import _header, chunk_document
from mcclub_rag.ingest.settings import IngestSettings
from tests.chunk_helpers import FakeCounter

COUNTER = FakeCounter()
SETTINGS = IngestSettings(
    chunk_target_tokens=30, chunk_max_tokens=40, chunk_min_tokens=8, chunk_max_header_tokens=8
)
HEADING_LINE = re.compile(r"^#{1,6} .*$", re.M)


def _chunk(document, settings=SETTINGS, counter=COUNTER):
    return chunk_document(document, settings=settings, counter=counter)


class TestFinalize:
    def test_title_equal_to_h1_appears_once(self):
        assert _header("Theming", ("Theming", "Overview"), 8, COUNTER) == "Theming > Overview"

    def test_long_path_keeps_innermost_headings(self):
        header = _header("Club", ("Statuts", "Titre deux", "Article trois"), 5, COUNTER)
        assert header == "Titre deux > Article trois"
