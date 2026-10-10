"""ParsedDocument -> ordered retrieval chunks (specs/chunking).

Structure first, sentences second, size last: headings are boundaries, oversized sections
split at paragraphs, lines, sentences, words and (last resort) tokens, and runs of tiny
sections under one parent are merged. Every chunk's ``embed_text`` carries a
"title > heading path" header for dense and BM25 retrieval. Pure and deterministic.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.sections import _HEADING
from mcclub_rag.ingest.tokens import TokenCounter

BlockKind = Literal["para", "list", "table", "code"]

_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")


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
