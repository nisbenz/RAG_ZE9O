# Progress

## 2026-10-10

### Shipped

- Chunking spec (`specs/chunking/`), researched against published benchmarks (sentence-aware
  splitting best for Arabic RAG; no overlap; no semantic chunking; contextual headers).
- `chunk_document()`: heading paths rebuilt from the sections, oversized sections split at
  paragraphs → lines → sentences → words → tokens, tables/lists/code kept whole, small sections
  merged, "title > path" header for retrieval, exact token cap on the embedder tokenizer.
- Embedder `tokenizer.json` downloaded at build (pinned revision, sha256-verified); new `tokenizer`
  test marker, run in the CI models job.
- Real corpus check: VP candidacy PDF 40 sections → 10 chunks, Arabic article 11 → 23 chunks
  (132–340 tokens), all under the 400-token cap.

### Next

- Pipeline spec: embedding (granite ONNX), Qdrant hybrid store, `corpus_version()` using
  `chunking_signature()`, reranker `max_length` ≥ 640.
- Eval: chunk target 200/300/400 × header on/off on `eval/questions.jsonl` (hit@5, MRR, Arabic subset).

## 2026-10-09

### Shipped

- Repo skeleton, project config, Docker setup.
- Data preprocessing spec (`specs/data-preprocessing/`: requirements, design, tasks), all 12 tasks done.
- `preprocess_file()` / `preprocess_url()`: PDF, DOCX, XLSX, MD, TXT, images and web pages → one
  `ParsedDocument` (clean text, sections with page numbers, language, OCR flag, content hash).
- OCR with Tesseract (ara, fra, eng), only on pages without a text layer; startup fails if a language is missing.
- Web ingestion with httpx + trafilatura: boilerplate removed, size cap, SSRF check on every redirect.
- Text cleanup, header/footer removal, Arabic search folding, language detection,
  `detect_language()` for the answer layer.
- Typed ingest errors with stable codes for the API layer.
- Tessdata downloaded at image build, sha256-pinned.
- CI (GitHub Actions): ruff + pytest on every push and PR. 290 tests passing.
- Fixed: xberg dropped the last lines of Arabic OCR (forced page segmentation mode 3, regression-tested).
- 11 edge cases recorded in EDGE_CASES.md.

### Blocked

- Real Arabic scans needed in `tests/fixtures/` (`arabic_scan.pdf`, `arabic_photo.jpg`) to test real-world OCR quality.
- Waiting on eequaled: `ingest_file` / `ingest_url` as `async def`, and whether a bad upload returns
  a 4xx or 200 with `status: "failed"`.

### Next

- Push `nisbenz/data-preprocessing`, check CI on GitHub, open PR into `dev`.
- Spec the pipeline: chunking, embedding (granite ONNX), Qdrant hybrid store, `corpus_version()`.
