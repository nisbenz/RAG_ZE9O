# Implementation Plan: Chunking

Each task is test-first: write the listed tests, see them fail, then implement. Every pushed task
keeps CI green (`uv lock --check`, ruff, `pytest -m "not ocr and not tokenizer"`, plus the model job).
Unit tests use a `FakeCounter` (1 token per whitespace-separated word), so only task 2's tests and
the golden/perf tests in task 8 need the real `tokenizer.json`.

- [ ] 1. Settings, errors and the `Chunk` contract
- [ ] 1.1 Chunking settings in `src/mcclub_rag/ingest/settings.py`
  - Add `chunk_target_tokens=300`, `chunk_max_tokens=400`, `chunk_min_tokens=80`,
    `chunk_max_header_tokens=48`, `chunk_context_header=True`, and `chunk_tokenizer_path` (default
    `$MODELS_DIR/embedder/tokenizer.json`, falling back to `/app/models/embedder/tokenizer.json`).
  - Add a `model_validator(mode="after")` enforcing `min < target <= max` and
    `max_header < max / 4`.
  - Append each variable, with one comment line, to `.env.example`.
  - Tests in `tests/test_settings.py`: the defaults; an env override (`INGEST_CHUNK_MAX_TOKENS=500`);
    `min >= target` and `target > max` both raise `ValidationError`.
  - _Requirements: 6.3, 9.1_
- [ ] 1.2 `TokenizerConfigError` in `src/mcclub_rag/ingest/errors.py`
  - Subclass of `IngestError` with code `tokenizer_config_error`, not HTTP-mapped (same as
    `ocr_config_error`).
  - Extend `tests/test_errors.py`: the code is stable and the message carries the instruction.
  - _Requirements: 6.2_
- [ ] 1.3 `Chunk` model in `src/mcclub_rag/ingest/models.py`
  - Frozen pydantic model with the fields from design "Data Models".
  - Extend `tests/test_models.py`: frozen; `heading_path` is a tuple; round-trips `model_dump()`.
  - _Requirements: 1.1, 1.5, 7.1_

- [ ] 2. Token counting
- [ ] 2.1 Pinned tokenizer download in `scripts/download_models.py`
  - `download_tokenizer(dest)`: fetch `tokenizer.json` from
    `ibm-granite/granite-embedding-311m-multilingual-r2` at revision
    `44399559930365213510b1ee2eb15ded83374f0e`, verify sha256
    `0087c868b33bad550a78a08d19798cfd7f713cde4f020803b8f51f405503e15f` (reuse `_sha256_file`),
    write atomically, and skip if the file is already present and valid.
  - Add `--only tokenizer` to `main()`; the default destination is `$MODELS_DIR/embedder`.
  - Extend `tests/test_download_models.py` (mocked HTTP, as for tessdata): a sha mismatch raises and
    leaves no file; an existing valid file is not re-downloaded; the CLI flag routes correctly.
  - _Requirements: 6.1, 6.2_
- [ ] 2.2 `src/mcclub_rag/ingest/tokens.py`
  - `TokenCounter` protocol (`count`, `count_many`, `cut`, `fingerprint`).
  - `HFTokenCounter(path)`: `Tokenizer.from_file`, then `no_truncation()` and `no_padding()`;
    `add_special_tokens=False`; `cut` splits at `offsets[max_tokens - 1][1]`; `fingerprint` = sha256
    of the file.
  - `verify_tokenizer(settings) -> Path` raises `TokenizerConfigError` with the download command.
  - `get_token_counter(settings)` caches one instance per path.
  - Register the `tokenizer` marker in `pyproject.toml` and auto-apply it to tests using a
    `tokenizer_path` fixture in `tests/conftest.py`. The fixture fails with the instruction when the
    file is missing, mirroring `tessdata_dir`.
  - `tests/test_tokens.py`:
    - A missing file raises `TokenizerConfigError` (no marker needed).
    - `[tokenizer]` a text above 40k tokens counts above 32,768.
    - `[tokenizer]` `count_many` equals `[count(t) ...]`, so padding is off.
    - `[tokenizer]` `cut` is lossless (`head + tail == text`) and `count(head) <= n` on Arabic,
      French and code samples.
    - `[tokenizer]` `fingerprint` is stable.
  - _Requirements: 6.1, 6.2, 6.4_
- [ ] 2.3 CI: model job also covers the tokenizer, in `.github/workflows/ci.yml`
  - In the existing OCR job, add a cache step keyed on `scripts/download_models.py` for
    `models/embedder`, run `download_models.py --only tokenizer --dest models/embedder`, then
    `INGEST_CHUNK_TOKENIZER_PATH=$PWD/models/embedder/tokenizer.json uv run pytest -m "ocr or tokenizer"`.
  - Change the fast job to `pytest -m "not ocr and not tokenizer"`.
  - Update the commands in `AGENTS.md` and the `README.md` to match.
  - _Requirements: 6.2_

- [ ] 3. Lossless sentence and word splitting in `src/mcclub_rag/text/sentences.py`
  - `split_sentences(text)` and `split_words(text)`, using the rules from design `text/sentences.py`.
    Both are pure, and their pieces keep trailing whitespace.
  - `tests/test_sentences.py`:
    - Arabic `؟`, `؛`, `۔` boundaries.
    - French and English `.`, `!`, `?`, `…`.
    - A terminator followed by `»`, `”` or `)`.
    - Non-boundaries: `3.5`, `e.g. this`, `M. Dupont`, `v1.2`, `example.com`, `p. 12`.
    - Mixed Arabic/French text.
    - Lossless join over parametrized samples and every `tests/fixtures/*.md`/`*.txt`.
  - _Requirements: 3.3, 1.3_

- [ ] 4. Outline and blocks in `src/mcclub_rag/ingest/chunk.py` (private helpers)
- [ ] 4.1 `_outline(sections) -> list[_OutlineSection]`
  - A heading stack reusing `sections._HEADING`; XLSX sheet reset; PDF continuation inherits; the
    heading line is removed from `body` and kept in `heading_line`.
  - `tests/test_chunk.py::TestOutline`:
    - An H1 > H2 > H3 path, and an H2 after an H3 popping correctly.
    - A page-2 PDF section without a heading line inherits the path.
    - Two XLSX sheets get paths `(sheet,)` each.
    - Text before the first heading has path `()`.
    - A `### x` with no H2 parent gets path `(H1, x)`.
  - _Requirements: 2.2, 2.3, 2.5_
- [ ] 4.2 `_parse_blocks(body, counter) -> list[_Block]`
  - Line scanner for code fences (including an unclosed fence), tables, lists with continuation
    lines, and blank-line paragraphs. One `count_many` call per section.
  - `TestBlocks`:
    - Each block kind is detected.
    - An unclosed fence runs to the end of the section.
    - XLSX row lines form one paragraph.
    - Joining the blocks reproduces the body, ignoring whitespace.
  - _Requirements: 3.4_

