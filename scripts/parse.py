"""Run the preprocessing pipeline on a file or URL and print the result.

    uv run python scripts/parse.py ~/Downloads/some.pdf
    uv run python scripts/parse.py https://www.bbc.com/arabic
    uv run python scripts/parse.py some.pdf --summary     # metadata and sections only
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from mcclub_rag.ingest.preprocess import preprocess_file, preprocess_url

LOCAL_TESSDATA = Path(__file__).resolve().parent.parent / "models" / "tessdata"


async def run(target: str, visibility: str):
    if target.startswith(("http://", "https://")):
        return await preprocess_url(target, visibility=visibility)
    path = Path(target).expanduser()
    return await preprocess_file(path.read_bytes(), path.name, visibility=visibility)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="path to a file, or an http(s) URL")
    parser.add_argument("--visibility", choices=("public", "members"), default="members")
    parser.add_argument("--summary", action="store_true", help="skip the full text")
    args = parser.parse_args()

    if LOCAL_TESSDATA.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(LOCAL_TESSDATA))

    doc = asyncio.run(run(args.target, args.visibility))
    print(doc.model_dump(exclude={"text", "sections"}))
    for s in doc.sections:
        print(f"--- page={s.page} [{s.language}] {s.heading!r} len={len(s.text)}")
    if not args.summary:
        print("\n=== TEXT ===\n")
        print(doc.text)


if __name__ == "__main__":
    main()
