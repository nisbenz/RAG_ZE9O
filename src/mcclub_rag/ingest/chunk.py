"""ParsedDocument -> ordered retrieval chunks (specs/chunking).

Structure first, sentences second, size last: headings are boundaries, oversized sections
split at paragraphs, lines, sentences, words and (last resort) tokens, and runs of tiny
sections under one parent are merged. Every chunk's ``embed_text`` carries a
"title > heading path" header for dense and BM25 retrieval. Pure and deterministic.
"""

import itertools
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.sections import _HEADING
from mcclub_rag.ingest.tokens import TokenCounter
from mcclub_rag.text.sentences import split_sentences, split_words

BlockKind = Literal["para", "list", "table", "code"]

_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


# Finer and finer lossless splitters tried on a block that exceeds the budget; after the
# last one comes a token-level cut. Tables are split by rows separately.
_LEVELS: dict[BlockKind, tuple[Callable[[str], list[str]], ...]] = {
    "para": (_lines, split_sentences, split_words),
    "list": (_lines, split_words),
    "code": (_lines, split_words),
    "table": (_lines, split_words),
}


@dataclass(frozen=True)
class _OutlineSection:
    index: int  # position in ParsedDocument.sections
    path: tuple[str, ...]  # full heading path, own heading included
    heading_line: str | None  # e.g. "## 01"; None when the section continues a heading
    body: str  # section text without its heading line
    page: int | None
    language: str | None


@dataclass(frozen=True)
class _Block:
    kind: BlockKind
    text: str
    tokens: int


@dataclass(frozen=True)
class _Unit:
    """A slice of a block small enough to pack; slices of one block join without a separator."""

    text: str
    tokens: int
    block: int
    hard: bool = False  # produced by a token-level cut


def _outline(sections: Sequence[Section]) -> list[_OutlineSection]:
    """Rebuild the heading tree from the '#' lines kept at the top of each section."""
    stack: list[tuple[int, str]] = []
    outline: list[_OutlineSection] = []
    for index, section in enumerate(sections):
        first, _, rest = section.text.partition("\n")
        match = _HEADING.match(first.strip())
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            heading_line, body = first.strip(), rest
        else:
            heading_line, body = None, section.text
            # An XLSX sheet names its section without a '#' line; a PDF page continuing a
            # heading repeats the heading already on the stack and inherits the path.
            if section.heading is not None and (not stack or stack[-1][1] != section.heading):
                stack = [(1, section.heading)]
        outline.append(
            _OutlineSection(
                index=index,
                path=tuple(heading for _, heading in stack),
                heading_line=heading_line,
                body=body.strip("\n"),
                page=section.page,
                language=section.language,
            )
        )
    return outline


def _parse_blocks(body: str, counter: TokenCounter) -> list[_Block]:
    """Split a section body into paragraphs, lists, tables and fenced code blocks."""
    lines = body.split("\n")
    spans: list[tuple[BlockKind, list[str]]] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            spans.append(("para", paragraph.copy()))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if fence := _FENCE.match(line):
            flush()
            end = i + 1
            while end < len(lines) and not lines[end].lstrip().startswith(fence.group(1)):
                end += 1
            spans.append(("code", lines[i : end + 1]))  # unclosed: runs to the end
            i = end + 1
        elif line.lstrip().startswith("|"):
            flush()
            end = i
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            spans.append(("table", lines[i:end]))
            i = end
        elif _LIST_ITEM.match(line):
            flush()
            end = i + 1
            while end < len(lines) and (
                _LIST_ITEM.match(lines[end])
                or (lines[end][:1] in (" ", "\t") and lines[end].strip())
            ):
                end += 1
            spans.append(("list", lines[i:end]))
            i = end
        elif not line.strip():
            flush()
            i += 1
        else:
            paragraph.append(line)
            i += 1
    flush()
    texts = ["\n".join(span).strip("\n") for _, span in spans]
    counts = counter.count_many(texts)
    return [
        _Block(kind, text, tokens)
        for (kind, _), text, tokens in zip(spans, texts, counts, strict=True)
    ]


def _split_text(
    text: str,
    budget: int,
    counter: TokenCounter,
    levels: Sequence[Callable[[str], list[str]]],
    block: int,
) -> list[_Unit]:
    if not levels:
        units = []
        rest = text
        while rest:
            head, rest = counter.cut(rest, budget)
            units.append(_Unit(head, counter.count(head), block, hard=True))
        return units
    pieces = levels[0](text)
    if len(pieces) <= 1:
        return _split_text(text, budget, counter, levels[1:], block)
    units = []
    for piece, tokens in zip(pieces, counter.count_many(pieces), strict=True):
        if tokens <= budget:
            units.append(_Unit(piece, tokens, block))
        else:
            units.extend(_split_text(piece, budget, counter, levels[1:], block))
    return units


def _split_table(text: str, target: int, counter: TokenCounter) -> list[str]:
    """Row groups of at most ``target`` tokens, each repeating the header (and separator)."""
    rows = text.split("\n")
    width = 2 if len(rows) > 1 and _TABLE_SEPARATOR.match(rows[1]) else 1
    header, body = rows[:width], rows[width:]
    size = header_size = counter.count("\n".join(header))
    groups: list[list[str]] = [[]]
    for row, tokens in zip(body, counter.count_many(body), strict=True):
        if groups[-1] and size + tokens > target:
            groups.append([])
            size = header_size
        groups[-1].append(row)
        size += tokens
    return ["\n".join(header + group) for group in groups]


def _units(
    blocks: Sequence[_Block], target: int, budget: int, counter: TokenCounter, ids: Iterator[int]
) -> list[_Unit]:
    """Blocks -> packable units: whole blocks when they fit, otherwise lossless slices."""
    units: list[_Unit] = []
    for block in blocks:
        if block.tokens <= budget:
            units.append(_Unit(block.text, block.tokens, next(ids)))
            continue
        pieces = (
            _split_table(block.text, target, counter) if block.kind == "table" else [block.text]
        )
        for piece, tokens in zip(pieces, counter.count_many(pieces), strict=True):
            if tokens <= budget:
                units.append(_Unit(piece, tokens, next(ids)))
            else:
                units.extend(_split_text(piece, budget, counter, _LEVELS[block.kind], next(ids)))
    return units


def _size(units: Sequence[_Unit], sep: int) -> int:
    joins = sum(1 for a, b in itertools.pairwise(units) if a.block != b.block)
    return sum(u.tokens for u in units) + joins * sep


def _pack(
    units: Sequence[_Unit], target: int, budget: int, min_tokens: int, sep: int
) -> list[list[_Unit]]:
    """Greedy packing up to ``target`` (units are already <= ``budget``), then fix a small tail."""
    pieces: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and _size([*current, unit], sep) > target:
            pieces.append(current)
            current = []
        current.append(unit)
    if current:
        pieces.append(current)
    if len(pieces) > 1 and _size(pieces[-1], sep) < min_tokens:
        prev, last = pieces[-2], pieces[-1]
        if _size(prev + last, sep) <= budget:
            pieces[-2:] = [prev + last]
        else:  # move whole units back into the tail while both stay reasonable
            while len(prev) > 1 and _size(last, sep) < min_tokens:
                moved = [prev[-1], *last]
                if _size(moved, sep) > budget or _size(prev[:-1], sep) < min_tokens:
                    break
                prev, last = prev[:-1], moved
            pieces[-2:] = [prev, last]
    return pieces


def _render(units: Sequence[_Unit]) -> str:
    groups = [
        "".join(u.text for u in group).strip()
        for _, group in itertools.groupby(units, key=lambda u: u.block)
    ]
    return "\n\n".join(g for g in groups if g)
