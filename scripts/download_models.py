"""Download model assets at image build time (offline at runtime).

Currently: Tesseract traineddata for ara/fra/eng, pinned to tag 4.1.0 and verified by
SHA-256. Embedder/reranker downloads can be added as further ``--only`` targets.

    uv run python scripts/download_models.py --only tessdata --dest models/tessdata
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import httpx

Variant = Literal["fast", "best"]

TESSDATA_TAG = "4.1.0"
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_{variant}/raw/{tag}/{lang}.traineddata"
TESSDATA_LANGUAGES = ("ara", "fra", "eng")
TESSDATA_SHA256: dict[str, dict[str, str]] = {
    "fast": {
        "ara": "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729",
        "fra": "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca",
        "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    },
    "best": {
        "ara": "ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896",
        "fra": "907743d98915c91a3906dfbf6e48b97598346698fe53aaa797e1a064ffcac913",
        "eng": "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
    },
}


class DownloadError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_tessdata(
    dest: Path,
    variant: Variant = "fast",
    *,
    languages: Iterable[str] = TESSDATA_LANGUAGES,
    client: httpx.Client | None = None,
    hashes: dict[str, str] | None = None,
) -> list[Path]:
    """Fetch ``<lang>.traineddata`` into ``dest``; skip files already present and valid.

    Each file is streamed to ``<name>.part``, hash-checked, then renamed into place, so a
    failed or tampered download never leaves a usable-looking file behind.
    """
    expected_hashes = hashes or TESSDATA_SHA256[variant]
    dest.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    written: list[Path] = []
    try:
        for lang in languages:
            target = dest / f"{lang}.traineddata"
            expected = expected_hashes[lang]
            if target.exists() and _sha256_file(target) == expected:
                print(f"tessdata {lang}: already present")
                written.append(target)
                continue
            url = TESSDATA_URL.format(variant=variant, tag=TESSDATA_TAG, lang=lang)
            partial = target.with_name(target.name + ".part")
            digest = hashlib.sha256()
            try:
                with http.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as fh:
                        for chunk in response.iter_bytes():
                            fh.write(chunk)
                            digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise DownloadError(
                        f"sha256 mismatch for {lang}.traineddata ({variant}): "
                        f"expected {expected}, got {digest.hexdigest()}"
                    )
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            print(f"tessdata {lang}: downloaded ({variant}, tag {TESSDATA_TAG})")
            written.append(target)
    finally:
        if owns_client:
            http.close()
    return written


def _default_tessdata_dest() -> Path:
    return Path(os.environ.get("MODELS_DIR", "/app/models")) / "tessdata"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        action="append",
        choices=["tessdata"],
        help="asset group to download (repeatable); default: all",
    )
    parser.add_argument(
        "--dest", type=Path, help="tessdata directory (default: $MODELS_DIR/tessdata)"
    )
    parser.add_argument("--variant", choices=["fast", "best"], default="fast")
    args = parser.parse_args(argv)

    targets = set(args.only or ["tessdata"])
    try:
        if "tessdata" in targets:
            download_tessdata(args.dest or _default_tessdata_dest(), args.variant)
    except (DownloadError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
