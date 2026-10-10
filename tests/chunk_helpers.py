"""Shared helpers for the chunking tests: a word-count token counter and a document builder."""

from collections.abc import Sequence
from datetime import UTC, datetime

from mcclub_rag.ingest.models import ParsedDocument, Section
from mcclub_rag.text.sentences import split_words


class FakeCounter:
    """One token per whitespace-separated word, so sizes in tests are exact and readable."""

    fingerprint = "fake"

    def count(self, text: str) -> int:
        return len(text.split())

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count(t) for t in texts]

    def cut(self, text: str, max_tokens: int) -> tuple[str, str]:
        words = split_words(text)
        head: list[str] = []
        for word in words:
            if word.strip() and self.count("".join(head)) >= max_tokens:
                break
            head.append(word)
        if not head:  # one oversized "word": cut by characters
            return text[:max_tokens], text[max_tokens:]
        cut = len("".join(head))
        return text[:cut], text[cut:]


def words(n: int, word: str = "mot") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def doc(*sections: Section, title: str = "Doc", language: str = "fr") -> ParsedDocument:
    return ParsedDocument(
        content_hash="ab" * 32,
        title=title,
        source="doc.md",
        source_type="file",
        mime_type="text/markdown",
        visibility="public",
        language=language,
        text="\n\n".join(s.text for s in sections),
        sections=sections,
        ocr_used=False,
        parser="text",
        page_count=None,
        created_at=datetime(2026, 10, 10, tzinfo=UTC),
    )


def md(*texts: str, page: int | None = None, language: str | None = "fr") -> tuple[Section, ...]:
    """Sections from Markdown snippets; the heading is taken from a leading '#' line."""
    out = []
    for text in texts:
        first = text.split("\n", 1)[0]
        heading = first.lstrip("#").strip() if first.startswith("#") else None
        out.append(Section(text=text, heading=heading, page=page, language=language))
    return tuple(out)
