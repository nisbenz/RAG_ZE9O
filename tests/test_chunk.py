from mcclub_rag.ingest.chunk import _outline, _parse_blocks
from mcclub_rag.ingest.models import Section
from tests.chunk_helpers import FakeCounter, md

COUNTER = FakeCounter()


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


class TestBlocks:
    def kinds(self, body):
        return [b.kind for b in _parse_blocks(body, COUNTER)]

    def test_each_kind(self):
        body = (
            "Intro paragraph\nstill intro.\n\n"
            "- one\n- two\n  continued\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
            "```python\nx = 1\n\ny = 2\n```\n\n"
            "Closing."
        )
        assert self.kinds(body) == ["para", "list", "table", "code", "para"]

    def test_code_fence_keeps_blank_lines(self):
        blocks = _parse_blocks("```\na\n\nb\n```", COUNTER)
        assert len(blocks) == 1
        assert blocks[0].text == "```\na\n\nb\n```"

    def test_unclosed_fence_runs_to_end(self):
        blocks = _parse_blocks("text\n\n```\ncode\n\nmore code", COUNTER)
        assert [b.kind for b in blocks] == ["para", "code"]
        assert blocks[1].text.endswith("more code")

    def test_numbered_list(self):
        assert self.kinds("1. first\n2) second") == ["list"]

    def test_xlsx_rows_are_one_paragraph(self):
        assert self.kinds("Nom: Ali · Rôle: HR\nNom: Sara · Rôle: OPS") == ["para"]

    def test_token_counts(self):
        blocks = _parse_blocks("un deux trois\n\nquatre", COUNTER)
        assert [b.tokens for b in blocks] == [3, 1]

    def test_lossless_ignoring_whitespace(self):
        body = "a b\n\n- c\n- d\n\n| e |\n|---|\n\n```\nf\n```\n\n\ng"
        joined = "".join(b.text for b in _parse_blocks(body, COUNTER))
        assert "".join(joined.split()) == "".join(body.split())
