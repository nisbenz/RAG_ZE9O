import hashlib
import importlib.util
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
_spec = importlib.util.spec_from_file_location("download_models", _SCRIPT)
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

PAYLOAD = {lang: f"{lang}-model-bytes".encode() * 100 for lang in ("ara", "fra", "eng")}
HASHES = {lang: hashlib.sha256(data).hexdigest() for lang, data in PAYLOAD.items()}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve_payload(request: httpx.Request) -> httpx.Response:
    lang = request.url.path.rsplit("/", 1)[-1].removesuffix(".traineddata")
    return httpx.Response(200, content=PAYLOAD[lang])


def test_downloads_and_verifies_all_languages(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _serve_payload(request)

    paths = dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)
    assert sorted(p.name for p in paths) == [
        "ara.traineddata",
        "eng.traineddata",
        "fra.traineddata",
    ]
    for lang, data in PAYLOAD.items():
        assert (tmp_path / f"{lang}.traineddata").read_bytes() == data
    assert all("/tessdata_fast/raw/4.1.0/" in url for url in seen)
    assert not list(tmp_path.glob("*.part"))


def test_hash_mismatch_raises_and_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"tampered")

    with pytest.raises(dm.DownloadError, match="sha256 mismatch for ara"):
        dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)
    assert list(tmp_path.iterdir()) == []


def test_http_error_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)
    assert list(tmp_path.iterdir()) == []


def test_existing_valid_file_is_not_downloaded_again(tmp_path):
    for lang, data in PAYLOAD.items():
        (tmp_path / f"{lang}.traineddata").write_bytes(data)

    def handler(request):
        raise AssertionError(f"unexpected download of {request.url}")

    dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)


def test_existing_invalid_file_is_replaced(tmp_path):
    (tmp_path / "ara.traineddata").write_bytes(b"stale")
    dm.download_tessdata(tmp_path, client=_client(_serve_payload), hashes=HASHES)
    assert (tmp_path / "ara.traineddata").read_bytes() == PAYLOAD["ara"]


def test_best_variant_uses_best_repo_and_pins(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _serve_payload(request)

    dm.download_tessdata(tmp_path, "best", client=_client(handler), hashes=HASHES)
    assert all("/tessdata_best/" in url for url in seen)
    assert set(dm.TESSDATA_SHA256["best"]) == {"ara", "fra", "eng"}


def test_cli_passes_dest_and_variant(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        dm, "download_tessdata", lambda dest, variant: calls.append((dest, variant))
    )
    rc = dm.main(["--only", "tessdata", "--dest", str(tmp_path), "--variant", "best"])
    assert rc == 0
    assert calls == [(tmp_path, "best")]


def test_cli_default_dest_uses_models_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(dm, "download_tessdata", lambda dest, variant: calls.append(dest))
    assert dm.main([]) == 0
    assert calls == [tmp_path / "tessdata"]


def test_cli_reports_failure(monkeypatch, capsys):
    def boom(dest, variant):
        raise dm.DownloadError("sha256 mismatch for ara")

    monkeypatch.setattr(dm, "download_tessdata", boom)
    assert dm.main(["--dest", "/nonexistent"]) == 1
    assert "sha256 mismatch" in capsys.readouterr().err
