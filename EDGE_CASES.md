# Edge cases

| Edge case | Proof it is real | Decision | Why |
|-----------|------------------|----------|-----|
| Junk bytes named `a.pdf` / `.txt` reach the wrong parser | xberg 1.3.6 routed junk named `a.pdf` to its PDF parser and accepted junk as `text/plain` under `content_only` | Own magic-byte sniffing before xberg (`ingest/sniff.py`); extension only picks md vs txt | xberg trusts the extension when bytes are ambiguous |
| OCR language data missing | With only `eng` present, xberg OCR'd `ara+fra+eng` without error; with network it downloaded `ara`/`fra` into `~/.cache/xberg` | `verify_tessdata()` fails fast at startup; `tessdata_path` always passed; tessdata baked into the image, sha256-pinned | Silent degradation of Arabic OCR, and runtime network use |
| Arabic OCR drops lines | xberg's default page segmentation returned 2 of 4 lines of an Arabic notice; `tesseract` CLI read all 4 | `TesseractConfig(psm=3)` + regression test `test_arabic_ocr_keeps_every_line_in_reading_order` | Lost text is invisible without a test |
| OCR results cached on disk | `~/.cache/xberg/ocr/*.msgpack` written even with `ExtractionConfig.use_cache=False` | Also `TesseractConfig.use_cache=False`; test asserts no cache files | Unbounded disk use in the container, stale results |
| OCR invents tables in notices | A plain photographed notice came back as a 5-column Markdown table | `enable_table_detection=False`, Tesseract output `text` | Garbled chunks hurt retrieval |
| Running headers/footers repeated in every chunk | "Club MC - Page N / 4" on every page of a PDF | `strip_repeated_lines`: edge lines repeated on ≥60% of pages (digits masked); never headings or table rows | Noise in BM25 and embeddings |
| Spreadsheet rows lose their column names when chunked | A Markdown table split mid-way leaves rows without the header | XLSX rows rendered as `Header: value · Header: value` lines | Each chunk stays self-explanatory |
| Web page with no article | trafilatura's fallback returned just "Home" for a nav+footer page | Fewer than 20 letters of main content → `EmptyDocument` | Indexing menu text is noise |
| Redirect to an internal address (SSRF) | A public URL can redirect to `http://169.254.169.254/` or `127.0.0.1:6333` (Qdrant) | Manual redirects; every hop resolved and checked (`url_guard.check_url`) | Admin URLs must not reach internal services |
| DNS rebinding between check and connect | TOCTOU: host may resolve differently when httpx connects | **Accepted for now**; would need a transport that connects to the vetted IP | Admin-only feature, low exposure; documented |
| URL tokens in logs | `?token=...` in a submitted URL | Logs strip query string and fragment | Secrets must not land in logs |
