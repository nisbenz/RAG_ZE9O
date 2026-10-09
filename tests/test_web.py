import httpx
import pytest

from mcclub_rag.ingest.errors import EmptyDocument, FetchError, UrlNotAllowed
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.ingest.web import extract_html, fetch

PUBLIC = {
    "club.example": "93.184.216.34",
    "cdn.example": "93.184.216.35",
    "evil.example": "10.0.0.1",
}


async def fake_resolve(host, port):
    return [PUBLIC.get(host, host)]


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def settings():
    return IngestSettings(web_max_bytes=1000, web_max_redirects=5)


async def test_fetch_html(settings):
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(
            200, html="<p>hi</p>", headers={"content-type": "text/html; charset=utf-8"}
        )

    res = await fetch(
        "https://club.example/a", settings, client=_client(handler), resolve=fake_resolve
    )
    assert res.final_url == "https://club.example/a"
    assert res.content_type == "text/html"
    assert res.data == b"<p>hi</p>"
    assert seen["ua"] == settings.web_user_agent


async def test_redirect_sets_final_url_and_rechecks_each_hop(settings):
    checked = []

    async def resolve(host, port):
        checked.append(host)
        return await fake_resolve(host, port)

    def handler(request):
        if request.url.host == "club.example":
            return httpx.Response(301, headers={"location": "https://cdn.example/final"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    res = await fetch(
        "https://club.example/start", settings, client=_client(handler), resolve=resolve
    )
    assert res.final_url == "https://cdn.example/final"
    assert checked == ["club.example", "cdn.example"]


async def test_relative_redirect_is_resolved(settings):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return httpx.Response(200, text="ok")

    res = await fetch(
        "https://club.example/old", settings, client=_client(handler), resolve=fake_resolve
    )
    assert res.final_url == "https://club.example/new"


@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/admin", "https://evil.example/", "file:///etc/passwd"]
)
async def test_redirect_to_internal_address_blocked(settings, target):
    def handler(request):
        return httpx.Response(302, headers={"location": target})

    with pytest.raises(UrlNotAllowed):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_too_many_redirects(settings):
    def handler(request):
        n = int(request.url.path.strip("/") or 0)
        return httpx.Response(302, headers={"location": f"/{n + 1}"})

    with pytest.raises(FetchError, match="too many redirects"):
        await fetch(
            "https://club.example/0", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_redirect_without_location(settings):
    with pytest.raises(FetchError, match="Location"):
        await fetch(
            "https://club.example/",
            settings,
            client=_client(lambda r: httpx.Response(302)),
            resolve=fake_resolve,
        )


@pytest.mark.parametrize("status", [404, 500, 403])
async def test_http_error_status(settings, status):
    with pytest.raises(FetchError, match=f"HTTP {status}"):
        await fetch(
            "https://club.example/",
            settings,
            client=_client(lambda r: httpx.Response(status)),
            resolve=fake_resolve,
        )


async def test_timeout(settings):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FetchError, match="timeout"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_connection_error(settings):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(FetchError, match="request failed"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_declared_length_over_cap_rejected_before_body(settings):
    def handler(request):
        return httpx.Response(200, headers={"content-length": "5000"}, content=b"x")

    with pytest.raises(FetchError, match="too large"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_streamed_body_over_cap_aborted(settings):
    async def chunks():
        for _ in range(50):
            yield b"x" * 100

    def handler(request):
        return httpx.Response(200, content=chunks())  # chunked, no Content-Length

    with pytest.raises(FetchError, match="too large"):
        await fetch(
            "https://club.example/", settings, client=_client(handler), resolve=fake_resolve
        )


async def test_initial_url_checked(settings):
    with pytest.raises(UrlNotAllowed):
        await fetch(
            "http://127.0.0.1/",
            settings,
            client=_client(lambda r: httpx.Response(200)),
            resolve=fake_resolve,
        )


# --- extract_html ------------------------------------------------------------------

ARTICLE = """<html lang="fr"><head><title>Programme de mars - Club MC</title></head><body>
<nav><a href="/">Accueil</a> | <a href="/contact">Contact</a> | <a href="/login">Connexion</a></nav>
<div class="cookie-banner">Nous utilisons des cookies pour améliorer votre expérience.</div>
<article>
<h1>Programme de mars</h1>
<p>La réunion mensuelle du club aura lieu le lundi 3 mars à dix-huit heures dans la salle B12.
Tous les membres sont invités à y participer et à présenter leur carte d'adhérent.</p>
<h2>Tarifs</h2>
<p>Les cotisations restent inchangées cette année pour tous les membres du club.</p>
<table><tr><th>Type</th><th>Prix</th></tr><tr><td>Membre</td><td>50 DT</td></tr>
<tr><td>Étudiant</td><td>30 DT</td></tr></table>
</article>
<footer>© 2026 Club MC - Tous droits réservés - Mentions légales</footer>
</body></html>""".encode()


def test_extract_html_keeps_article_drops_boilerplate():
    raw = extract_html(ARTICLE, "https://club.example/mars")
    assert raw.kind == "html"
    assert raw.parser.startswith("trafilatura ")
    assert "# Programme de mars" in raw.markdown
    assert "## Tarifs" in raw.markdown
    assert "salle B12" in raw.markdown
    assert "| Membre | 50 DT |" in raw.markdown.replace("| \n", "|\n")
    for boilerplate in ["Accueil", "Connexion", "cookies", "Tous droits réservés"]:
        assert boilerplate not in raw.markdown
    assert raw.metadata_title and "Programme de mars" in raw.metadata_title


@pytest.mark.parametrize(
    "page",
    [
        b"<html><body><nav><a href='/'>Home</a> | <a href='/c'>Contact</a></nav>"
        b"<div class='cookie'>We use cookies.</div>"
        b"<footer>(c) 2026 Club MC. All rights reserved.</footer></body></html>",
        # trafilatura's fallback returns just "Home" here
        b"<html><body><nav><a href='/'>Home</a></nav><footer>(c) 2026</footer></body></html>",
        b"<html><body></body></html>",
    ],
)
def test_extract_html_without_main_content_is_empty(page):
    with pytest.raises(EmptyDocument):
        extract_html(page, "https://club.example/empty")
