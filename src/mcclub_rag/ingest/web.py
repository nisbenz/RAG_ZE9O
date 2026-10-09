"""Web page fetching (httpx) and main-content extraction (trafilatura), Req 4.

Redirects are followed manually so every hop goes through ``check_url``. The body is
streamed and capped; ``aiter_bytes`` decompresses, so the cap also bounds gzip bombs.
"""

import asyncio
from dataclasses import dataclass

import httpx
import trafilatura

from mcclub_rag.ingest.errors import EmptyDocument, FetchError
from mcclub_rag.ingest.models import RawExtraction
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.ingest.url_guard import Resolver, check_url, default_resolver

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
# trafilatura falls back to whatever text it finds (e.g. a lone "Home" link) on pages with no
# article; fewer letters than this cannot answer anything.
_MIN_CONTENT_LETTERS = 20


@dataclass(frozen=True)
class FetchedResource:
    final_url: str
    content_type: str  # lowercased, parameters stripped ("text/html")
    data: bytes


def _too_large(settings: IngestSettings) -> FetchError:
    return FetchError(f"response too large (limit {settings.web_max_bytes} bytes)")


async def _fetch(
    url: str, settings: IngestSettings, http: httpx.AsyncClient, resolve: Resolver
) -> FetchedResource:
    headers = {"User-Agent": settings.web_user_agent}
    current = url
    for _ in range(settings.web_max_redirects + 1):
        await check_url(current, resolve)
        try:
            async with http.stream("GET", current, headers=headers, follow_redirects=False) as resp:
                if resp.status_code in _REDIRECT_STATUSES:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError(
                            f"HTTP {resp.status_code} redirect without Location header"
                        )
                    current = str(resp.url.join(location))
                    continue
                if not resp.is_success:
                    raise FetchError(f"HTTP {resp.status_code}")
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > settings.web_max_bytes:
                    raise _too_large(settings)
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > settings.web_max_bytes:
                        raise _too_large(settings)
                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                return FetchedResource(str(resp.url), content_type, bytes(body))
        except httpx.TimeoutException as exc:
            raise FetchError("timeout") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"request failed: {type(exc).__name__}") from exc
    raise FetchError(f"too many redirects (limit {settings.web_max_redirects})")


async def fetch(
    url: str,
    settings: IngestSettings,
    *,
    client: httpx.AsyncClient | None = None,
    resolve: Resolver = default_resolver,
) -> FetchedResource:
    http = client or httpx.AsyncClient(timeout=settings.web_timeout_s)
    try:
        # httpx timeouts are per operation; this bounds a slow-drip server overall.
        async with asyncio.timeout(settings.web_timeout_s * 2):
            return await _fetch(url, settings, http, resolve)
    except TimeoutError as exc:
        raise FetchError("timeout") from exc
    finally:
        if client is None:
            await http.aclose()


def extract_html(data: bytes, final_url: str) -> RawExtraction:
    """Main content as Markdown (headings and tables kept). Blocking: run in a thread."""
    markdown = trafilatura.extract(
        data,
        url=final_url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        include_images=False,
        include_links=False,
        favor_precision=True,
    )
    if not markdown or sum(ch.isalpha() for ch in markdown) < _MIN_CONTENT_LETTERS:
        raise EmptyDocument("no main content found on the page")
    metadata = trafilatura.extract_metadata(data)
    title = (metadata.title or "").strip() if metadata else ""
    return RawExtraction(
        kind="html",
        mime_type="text/html",
        parser=f"trafilatura {trafilatura.__version__}",
        markdown=markdown,
        metadata_title=title or None,
    )
