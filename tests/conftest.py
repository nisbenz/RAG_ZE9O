import os
from pathlib import Path

import pytest

from mcclub_rag.ingest.settings import IngestSettings

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OCR_LANGUAGES = ("ara", "fra", "eng")


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


@pytest.fixture
def ocr_settings(tessdata_dir: Path) -> IngestSettings:
    return IngestSettings(tessdata_prefix=tessdata_dir)


@pytest.fixture
def fixture_bytes():
    def read(name: str) -> bytes:
        return (FIXTURES / name).read_bytes()

    return read
