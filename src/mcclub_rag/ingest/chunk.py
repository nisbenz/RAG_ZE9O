"""ParsedDocument -> ordered retrieval chunks (specs/chunking).

Structure first, sentences second, size last: headings are boundaries, oversized sections
split at paragraphs, lines, sentences, words and (last resort) tokens, and runs of tiny
sections under one parent are merged. Every chunk's ``embed_text`` carries a
"title > heading path" header for dense and BM25 retrieval. Pure and deterministic.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.sections import _HEADING


@dataclass(frozen=True)
class _OutlineSection:
    index: int  # position in ParsedDocument.sections
    path: tuple[str, ...]  # full heading path, own heading included
    heading_line: str | None  # e.g. "## 01"; None when the section continues a heading
    body: str  # section text without its heading line
    page: int | None
    language: str | None


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
