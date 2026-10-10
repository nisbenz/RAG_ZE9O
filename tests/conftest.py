import os
from pathlib import Path

import pytest

from mcclub_rag.ingest.settings import IngestSettings

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OCR_LANGUAGES = ("ara", "fra", "eng")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # preprocess_* verifies traineddata before parsing anything, so every test that pulls in
    # the tessdata fixture needs the models even if it never OCRs; keep -m "not ocr" honest.
    # Same for the embedder tokenizer: keep -m "not tokenizer" free of the 33 MB download.
    for item in items:
        fixtures = getattr(item, "fixturenames", ())
        if "tessdata_dir" in fixtures:
            item.add_marker(pytest.mark.ocr)
        if "tokenizer_path" in fixtures:
            item.add_marker(pytest.mark.tokenizer)


def _tessdata_dir() -> Path:
    env = os.environ.get("TESSDATA_PREFIX")
    return Path(env) if env else REPO / "models" / "tessdata"


@pytest.fixture(scope="session")
def tessdata_dir() -> Path:
    """Traineddata directory. Fails (never skips) when language data is missing."""
    directory = _tessdata_dir()
    missing = [lang for lang in OCR_LANGUAGES if not (directory / f"{lang}.traineddata").is_file()]
    if missing:
        pytest.fail(
            f"OCR traineddata missing in {directory}: {', '.join(missing)}. Run:\n"
            "  uv run python scripts/download_models.py --only tessdata --dest models/tessdata",
            pytrace=False,
        )
    return directory


def _tokenizer_path() -> Path:
    env = os.environ.get("INGEST_CHUNK_TOKENIZER_PATH")
    return Path(env) if env else REPO / "models" / "embedder" / "tokenizer.json"


@pytest.fixture(scope="session")
def tokenizer_path() -> Path:
    """Embedder tokenizer.json. Fails (never skips) when it is missing."""
    path = _tokenizer_path()
    if not path.is_file():
        pytest.fail(
            f"Embedder tokenizer missing at {path}. Run:\n"
            "  uv run python scripts/download_models.py --only tokenizer --dest models/embedder",
            pytrace=False,
        )
    return path


@pytest.fixture
def ocr_settings(tessdata_dir: Path) -> IngestSettings:
    return IngestSettings(tessdata_prefix=tessdata_dir)


@pytest.fixture
def fixture_bytes():
    def read(name: str) -> bytes:
        return (FIXTURES / name).read_bytes()

    return read
