# SEC Simplifier — Single-Company Due-Diligence MVP

This project is intentionally scoped to a single company, single session.
The goal is to answer natural-language questions using filing text that is
actually grounded in EDGAR documents, with visible citations and explicit
"I don't see this disclosed" behavior when a claim is unsupported.

## Scope

- One ticker only: NVCT
- Recent 10-K / 10-Q / 8-K / DEF 14A filings
- Local ingestion and metadata capture
- Section-aware chunking from filing HTML
- grounded answers with citation back to the filing section
- honest failure when the answer is not disclosed

This cuts out:

- multi-company comparison
- broad ticker coverage
- automated alerts on new filings
- heavy orchestration or generic RAG framework complexity

## Current state

The repo now has a minimal working MVP around the grounded Q&A loop:

- [src/ingest_filings.py](src/ingest_filings.py): fetches filings for the chosen ticker
- [src/sec_client.py](src/sec_client.py): SEC API client and filing metadata (accession number, period of report)
- [src/grounded_qa.py](src/grounded_qa.py): chunking, stable chunk IDs, hybrid retrieval, and grounded answer logic
- [src/tables.py](src/tables.py): separates financial tables from layout tables and renders them as readable chunks
- [src/corpus.py](src/corpus.py): turns downloaded filings into chunks with full metadata
- [src/vector_store.py](src/vector_store.py): persists chunks and metadata in a local Chroma store
- [src/build_index.py](src/build_index.py): CLI that builds or refreshes the vector index
- [tests/test_grounded_qa.py](tests/test_grounded_qa.py): chunking, retrieval, grounding, and vector-store contract tests
- [tests/golden_set.json](tests/golden_set.json): reviewed evaluation cases, with known gaps recorded explicitly

## Six-week build roadmap

See [PROGRESS.md](PROGRESS.md) for a pipeline diagram and a full week-by-week
account of what is built, what is pending, and the known weaknesses.

| Week | Focus | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | SEC EDGAR filings | Download a defined company scope: 10-K, 10-Q, 8-K, and DEF 14A filings, with source URLs and filing metadata. | Complete - NVCT filings download locally. |
| 2 | Ingestion and chunking | Parse downloaded filing HTML into section-aware chunks that retain ticker, form, filing date, document name, and SEC source URL. | Core complete - section-aware chunks are loaded into the app. |
| 3 | Vector store and metadata | Persist chunks and metadata in Chroma, with a stable chunk ID, filing accession number, section, period, and source URL. | Complete - `src/build_index.py` writes a persistent Chroma store; every chunk carries a stable ID and full filing metadata. |
| 4 | Hybrid retrieval | Combine lexical/keyword search with vector similarity search, then rerank the strongest evidence chunks. | Complete - table-aware chunking, BM25 + vector search fused with reciprocal rank fusion, two-gate abstention. Live in the app. |
| 5 | Grounded generation | Add Claude orchestration that answers only from retrieved evidence and can abstain when evidence is inadequate. | Next. |
| 6 | Cited answers and evaluation | Show quoted evidence and source links in the UI; build a reviewed golden set and measure citation correctness, support, and abstention quality. | Baseline started - UI evidence cards, and a 13-case golden set scoring 11/13. |

### Current position

The project is at the end of **Week 4** and ready for **Week 5**. The app now
answers through hybrid retrieval over the persisted Chroma index: financial
tables are chunked as tables, lexical and vector search are fused, and abstention
is guarded by two independent gates. What is still missing is a model that
*reads* the retrieved evidence — answers are excerpted, not written, and the two
open evaluation gaps both need a reader rather than a better retriever.

### Still open (carried into Week 5-6)

- Answers are the top chunk truncated to 300 characters behind a template
  sentence. Real synthesis is Week 5.
- Two golden-set cases still answer when they should abstain, both because the
  retrieved evidence genuinely matches the question's words while being about a
  different subject. See [PROGRESS.md](PROGRESS.md).
- Retrieval thresholds are tuned against 13 cases; the golden set needs to grow
  before they can be trusted at scale.

## Week 2: retrieval and grounding baseline

The Week 2 retrieval layer was transparent lexical scoring over filing chunks.
It returned only sections sharing meaningful terms with the question, never
falling back to unrelated text, and produced the explicit unsupported-answer
response with no citations when nothing matched.

It is still the lexical half of hybrid retrieval, and still the fallback when the
vector store is unavailable — exact terms, ticker symbols, and form/item
references are signals embeddings tend to blur. Week 4 raised its bar (see
below) but kept the same evidence-and-abstention contract.

## Week 3: vector store and metadata

Downloaded filings are chunked once and persisted in a local
[Chroma](https://www.trychroma.com/) collection under `data/chroma/`. Each row is:

- **Document**: the section's plain text.
- **ID**: `accession:section-slug:position` — stable across re-ingestion, so a
  rerun updates rows in place instead of duplicating them.
- **Metadata**: ticker, form, filing date, period of report, accession number,
  section heading, source URL, and document name.

Retrieval is cosine similarity over Chroma's default on-device embedding model
(`all-MiniLM-L6-v2`, downloaded once to `~/.cache/chroma`). No filing text or
query leaves the machine.

### Build the vector index

Run this once after downloading filings (and again whenever you re-ingest):

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m src.build_index
```

Add `--reset` after changing how filings are chunked — stable IDs keep
re-ingestion clean, but a chunker that splits differently leaves the previous
run's rows behind. Add `--probe "your question"` to see how similarity search
ranks a question against the persisted chunks:

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m src.build_index --reset --probe "What is NVCT's cash runway?"
```

## Week 4: table-aware chunking and hybrid retrieval

### Tables are chunked as tables

A filing uses `<table>` for both financial data and page layout — the 10-Q has
207 tables, of which 17 carry data. Data tables are now lifted out and rendered
with their caption, one row per line, so a figure keeps the label that explains
it:

```
CONDENSED BALANCE SHEETS (USD in thousands) (unaudited)
June 30, | December 31,
2026 | 2025
Cash and cash equivalents | 22,160 | 31,634
TOTAL ASSETS | 22,376 | 31,709
```

Prose chunks are capped at 3,000 characters and split on sentence boundaries.
Validating this surfaced two chunker defects — text counted once per nesting
level (the largest chunk was 466,558 characters), and the table of contents
being read as real section headings. Both are fixed; see
[PROGRESS.md](PROGRESS.md).

### Retrieval

`retrieve_hybrid()` runs both retrievers, gates each independently, and fuses the
survivors with reciprocal rank fusion. Ranking is BM25 rather than raw term
overlap, which otherwise favours long chunks so heavily that `Item 4. Controls
and Procedures` came back as the best evidence about cash.

Abstention is guarded by two gates, and a chunk must clear at least one:

| Gate | Rule |
| --- | --- |
| Lexical | ≥ 2 matching terms (section headings count double) **and** ≥ 60% of the question's meaningful terms covered |
| Vector | cosine distance ≤ 0.60 |

Neither signal separates on its own — measured against the golden set, the
lexical count is identical for questions that should be answered and questions
that should not. The vector distance nearly separates cleanly, which is where the
0.60 floor comes from.

The app falls back to lexical-only when `chromadb` is absent or the index has not
been built, so the abstention contract holds either way.

## Install dependencies

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
```

`chromadb` pulls in `onnxruntime` and downloads a ~80 MB embedding model on first
use. If you only need the lexical baseline, the app and its tests still run
without it — the golden-set tests skip and the app reports keyword-only mode.

## Load real EDGAR filings

Use a real name and contact email in the User-Agent as requested by the SEC.
This downloads the two most recent 10-K, 10-Q, 8-K, and DEF 14A filings for a
ticker and stores the raw documents locally under `data/`.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m src.ingest_filings --ticker NVCT --limit-per-form 2 --user-agent "Your Name your-email@example.com"
```

After ingesting, rebuild the index with `python -m src.build_index --reset` and
restart `app.py`. The corpus is cached for the life of the process, so a restart
is what picks up new filings. The status line above the question box reports
which mode is live — hybrid search over the index, keyword-only over parsed
filings, or the small demo corpus.

## Run tests

```powershell
cd "C:\Users\yceri\Desktop\SEC Simplifier\sec-simplifier"
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m pytest -q
```

34 tests: contract tests in `tests/test_grounded_qa.py`, plus the reviewed
evaluation cases in [tests/golden_set.json](tests/golden_set.json) run by
`tests/test_golden_set.py`. The golden-set tests skip cleanly until the index has
been built with `python -m src.build_index`.

## Run the demo

`run_demo.py` answers every case in the golden set twice - once with keyword
search alone, once with hybrid retrieval - and prints both verdicts side by side.
It needs the index built first.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\run_demo.py
```

```
  Q  Has the company ever made a profit?
     expect   supported  <- hybrid differs
     keyword  ABSTAINS  MISS
     hybrid   SUPPORTED OK
     cites    Item 7. Management's Discussion and Analysis (10-K 2026-02-11)
     why      keyword alone abstains - 'profit' does not appear as such

==============================================================================
  keyword only : 9/13 correct
  hybrid       : 11/13 correct
==============================================================================
```

Add `--contrast` to show only the cases where the two retrievers disagree.

On this machine, `python` currently resolves to the Microsoft Store launcher,
so use the full interpreter path above until Python is added to your `PATH`.
After adding `C:\Users\yceri\AppData\Local\Programs\Python\Python310\` to
your user `PATH` and opening a new PowerShell window, the shorter command will
work:

```powershell
python .\run_demo.py
```

## Run the local UI

The browser UI runs entirely on your computer. It automatically uses downloaded
local EDGAR filings when available and otherwise falls back to the small demo
corpus used by `run_demo.py`. It is aimed at first-time investors and small
research teams: it provides starter questions, plain-English context on why an
answer matters, and the exact supporting filing excerpt alongside each source
link.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\app.py
```

Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
Use `Ctrl+C` in PowerShell to stop the server.

## Expected behavior

- Supported question: returns answer + filing citations
- Unsupported question: returns "I don't see this disclosed in the filings available for this company."
- This is the trust signal that separates a toy demo from a real analyst tool

## Next step

Use this as the default MVP for interview demos and then expand only if needed
into multi-company compare or filing-diff workflows later. Refer to previous iterations
