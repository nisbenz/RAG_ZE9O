import itertools

from mcclub_rag.ingest.chunk import _outline, _pack, _parse_blocks, _render, _size, _Unit, _units
from mcclub_rag.ingest.models import Section
from tests.chunk_helpers import FakeCounter, md, words

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


def _units_of(body, target=30, budget=40, counter=COUNTER):
    return _units(_parse_blocks(body, counter), target, budget, counter, itertools.count())


class CharCounter(FakeCounter):
    """One token per character, so a single long word can exceed the budget."""

    def count(self, text):
        return len(text)

    def cut(self, text, max_tokens):
        return text[:max_tokens], text[max_tokens:]


class TestSplit:
    def test_long_paragraph_splits_at_sentences(self):
        body = " ".join(f"{words(9)} fin{i}." for i in range(100))  # 1,000 words
        pieces = [_render(p) for p in _pack(_units_of(body), 30, 40, 5, 0)]
        assert len(pieces) > 10
        assert all(p.endswith(".") for p in pieces)
        assert all(COUNTER.count(p) <= 40 for p in pieces)

    def test_oversized_table_repeats_header(self):
        rows = "\n".join(f"| {words(4)} | r{i} |" for i in range(100))
        body = f"| Name | Role |\n|---|---|\n{rows}"
        units = _units_of(body)
        assert len(units) > 1
        assert all(u.text.startswith("| Name | Role |\n|---|---|\n|") for u in units)
        assert all(u.tokens <= 40 for u in units)
        assert sum(u.text.count("| r") for u in units) == 100

    def test_oversized_code_splits_at_newlines(self):
        lines = [f"x{i} = {words(5)}" for i in range(50)]
        body = "```\n" + "\n".join(lines) + "\n```"
        units = _units_of(body)
        assert len(units) > 1
        for unit in units:
            assert all(line in body.split("\n") for line in unit.text.strip().split("\n"))

    def test_single_huge_word_is_cut_losslessly(self):
        word = "x" * 600
        units = _units_of(word, target=80, budget=100, counter=CharCounter())
        assert len(units) == 6
        assert all(u.hard for u in units)
        assert "".join(u.text for u in units) == word


def _u(tokens, block):
    return _Unit(" ".join(["w"] * tokens), tokens, block)


class TestPack:
    def test_block_that_fits_budget_is_never_cut(self):
        table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | x |" for i in range(8))
        units = _units_of(table, target=10, budget=60)
        assert len(units) == 1
        assert _render(units) == table

    def test_no_piece_exceeds_target_or_budget(self):
        units = [_u(n, i) for i, n in enumerate([5, 17, 3, 30, 12, 8, 25, 1, 9, 14])]
        pieces = _pack(units, target=30, budget=40, min_tokens=5, sep=1)
        assert all(_size(p, 1) <= 30 for p in pieces[:-1])
        assert all(_size(p, 1) <= 40 for p in pieces)
        assert [u for p in pieces for u in p] == units

    def test_small_tail_joins_previous(self):
        pieces = _pack([_u(25, 0), _u(10, 1), _u(3, 2)], target=30, budget=40, min_tokens=8, sep=1)
        assert [_size(p, 1) for p in pieces] == [25, 14]

    def test_small_tail_is_rebalanced_when_it_cannot_join(self):
        units = [_u(10, 0), _u(10, 1), _u(10, 2), _u(3, 3)]
        pieces = _pack(units, target=32, budget=32, min_tokens=8, sep=0)
        assert [_size(p, 0) for p in pieces] == [20, 13]

    def test_render_joins_blocks_with_blank_lines(self):
        units = [_Unit("Hello ", 1, 0), _Unit("world.", 1, 0), _Unit("- item", 2, 1)]
        assert _render(units) == "Hello world.\n\n- item"

    def test_arabic_section_splits_at_paragraphs(self):
        paragraph = " ".join(["الصمود في غزة"] * 6) + "."
        body = "\n\n".join([paragraph] * 6)  # 6 paragraphs of 18 words
        pieces = [_render(p) for p in _pack(_units_of(body), 30, 40, 5, 0)]
        assert all(p.count(paragraph) >= 1 for p in pieces)
        assert "".join(pieces).replace("\n", "") == body.replace("\n", "")
