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
- [src/grounded_qa.py](src/grounded_qa.py): section-aware chunking, stable chunk IDs, and grounded answer logic
- [src/corpus.py](src/corpus.py): turns downloaded filings into chunks with full metadata
- [src/vector_store.py](src/vector_store.py): persists chunks and metadata in a local Chroma store
- [src/build_index.py](src/build_index.py): CLI that builds or refreshes the vector index
- [tests/test_grounded_qa.py](tests/test_grounded_qa.py): retrieval, grounding, and vector-store contract tests

## Six-week build roadmap

See [PROGRESS.md](PROGRESS.md) for a pipeline diagram and a full week-by-week
account of what is built, what is pending, and the known weaknesses.

| Week | Focus | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | SEC EDGAR filings | Download a defined company scope: 10-K, 10-Q, 8-K, and DEF 14A filings, with source URLs and filing metadata. | Complete - NVCT filings download locally. |
| 2 | Ingestion and chunking | Parse downloaded filing HTML into section-aware chunks that retain ticker, form, filing date, document name, and SEC source URL. | Core complete - section-aware chunks are loaded into the app. |
| 3 | Vector store and metadata | Persist chunks and metadata in Chroma, with a stable chunk ID, filing accession number, section, period, and source URL. | Complete - `src/build_index.py` writes a persistent Chroma store; every chunk carries a stable ID and full filing metadata. |
| 4 | Hybrid retrieval | Combine lexical/keyword search with vector similarity search, then rerank the strongest evidence chunks. | Next - lexical scorer and vector search both exist; still need to merge and rerank them behind the grounded-answer contract. |
| 5 | Grounded generation | Add Claude orchestration that answers only from retrieved evidence and can abstain when evidence is inadequate. | Planned. |
| 6 | Cited answers and evaluation | Show quoted evidence and source links in the UI; build a reviewed golden set and measure citation correctness, support, and abstention quality. | Baseline started - UI evidence cards and regression tests exist. |

### Current position

The project is at the end of **Week 3** and ready for **Week 4**. Downloaded
filings are now persisted as embedded chunks in a local Chroma store, each with a
stable chunk ID and full filing metadata. The app's live answer path still uses
the Week 2 lexical scorer; Week 4 merges the two into hybrid retrieval.

### Done in Week 3

- Stable chunk IDs. Each chunk's ID is `accession:section-slug:position`, so
  re-running ingestion upserts the same rows instead of creating duplicates.
- Accession number and period of report are captured during ingestion
  ([src/sec_client.py](src/sec_client.py)) and stored on every chunk, not only
  the filing date. This is what year-over-year comparison will key on.
- A persistent Chroma collection ([src/vector_store.py](src/vector_store.py))
  with cosine similarity and Chroma's default on-device embedding model. Nothing
  is sent off the machine.
- `store_stats()` and the `--probe` flag on the index builder for inspecting
  what is persisted and how retrieval ranks a question.

### Still open (carried into Week 4-6)

- Add table-aware chunking. Financial statement tables should be extracted as
  structured rows with their table title and reporting period, rather than being
  flattened into paragraph text. (The 10-Q financial-statement chunks currently
  come back as long runs of numbers.)
- Keep the current lexical scorer as part of hybrid retrieval; do not replace it
  outright with vectors. Exact terms, ticker symbols, and form/item references
  are useful financial-research signals.
- Build the evaluation golden set. Each case should record the question, the
  expected support/abstention result, and the correct filing section or quoted
  evidence.

## Week 2: retrieval and grounding baseline

The current retrieval layer uses transparent lexical scoring over filing chunks.
It returns only sections sharing meaningful terms with the question; it never
falls back to unrelated text. When no supporting section is found, the app
returns the explicit unsupported-answer response with no citations.

This is intentionally simple and inspectable. A future semantic retriever or
reranker can replace the scorer, but must preserve the same evidence and
abstention contract.

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

The store is a persistence layer, not yet the live answer path. The app still
answers through the Week 2 lexical scorer; Week 4 combines lexical and vector
retrieval behind the same evidence-and-abstention contract.

### Build the vector index

Run this once after downloading filings (and again whenever you re-ingest):

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m src.build_index
```

Add `--probe "your question"` to see how similarity search ranks a question
against the persisted chunks:

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m src.build_index --probe "What is NVCT's cash runway?"
```

## Install dependencies

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
```

`chromadb` (Week 3) pulls in `onnxruntime` and downloads a ~80 MB embedding
model on first use. If you only need the Week 2 lexical baseline, the app and its
tests still run without it.

## Load real EDGAR filings

Use a real name and contact email in the User-Agent as requested by the SEC.
This downloads the two most recent 10-K, 10-Q, 8-K, and DEF 14A filings for a
ticker and stores the raw documents locally under `data/`.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m src.ingest_filings --ticker NVCT --limit-per-form 2 --user-agent "Your Name your-email@example.com"
```

Restart `app.py` after ingestion. It detects local EDGAR filings automatically
and displays the company and section count above the question box. If none are
available, it continues to use the small demo corpus.

## Run tests

```powershell
cd "C:\Users\yceri\Desktop\SEC Simplifier\sec-simplifier"
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m pytest -q tests/test_grounded_qa.py
```

## Run the demo

From the project directory, run:

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\run_demo.py
```

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
