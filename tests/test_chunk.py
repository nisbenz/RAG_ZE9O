from mcclub_rag.ingest.chunk import _outline
from mcclub_rag.ingest.models import Section
from tests.chunk_helpers import md


class TestOutline:
    def test_heading_tree(self):
        outline = _outline(md("# A\nintro", "## B\nb", "### C\nc", "## D\nd", "# E\ne", "### F\nf"))
        assert [o.path for o in outline] == [
            ("A",),
            ("A", "B"),
            ("A", "B", "C"),
            ("A", "D"),
            ("E",),
            ("E", "F"),
        ]
        assert outline[2].heading_line == "### C"
        assert outline[2].body == "c"

    def test_pdf_continuation_inherits_path(self):
        sections = (
            Section(text="# Rules", heading="Rules", page=1),
            Section(text="## Fees\nPay by Friday.", heading="Fees", page=1),
            Section(text="Late fees double.", heading="Fees", page=2),
        )
        outline = _outline(sections)
        assert outline[2].path == ("Rules", "Fees")
        assert outline[2].heading_line is None
        assert outline[2].page == 2

    def test_xlsx_sheets_get_their_own_path(self):
        sections = (
            Section(text="Nom: Ali · Rôle: HR", heading="Membres"),
            Section(text="Date: 12/10 · Lieu: Salle 3", heading="Planning"),
        )
        assert [o.path for o in _outline(sections)] == [("Membres",), ("Planning",)]

    def test_text_before_first_heading_has_empty_path(self):
        outline = _outline(md("Preamble text.", "# A\na"))
        assert outline[0].path == ()
        assert outline[0].body == "Preamble text."

    def test_skipped_level(self):
        assert _outline(md("# A\na", "### X\nx"))[1].path == ("A", "X")

    def test_heading_only_section_has_empty_body(self):
        outline = _outline(md("## Contents"))
        assert outline[0].body == ""
        assert outline[0].index == 0
