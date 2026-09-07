# SEC Simplifier — Grounded Due Diligence on SEC Filings

Search any US public company, ask questions in plain English, and get answers
built only from that company's EDGAR filings — with visible citations and an
explicit "I don't see this disclosed" when a claim is unsupported.

Each question is answered within one company: type a ticker, the app downloads
and indexes its recent filings, and every answer is scoped to them.

## Scope

- **Any US company with a ticker in EDGAR's mapping** - searched and indexed on demand
- Recent 10-K / 10-Q / 8-K / DEF 14A filings, two of each per company
- Local ingestion and metadata capture
- Section- and table-aware chunking from filing HTML
- Grounded answers with citation back to the filing section
- Honest failure when the answer is not disclosed

Every question is scoped to one company. This still cuts out:

- multi-company **comparison** in a single answer (period alignment and
  line-item reconciliation are their own problem)
- foreign private issuers, funds, and anything without a ticker in
  `company_tickers.json`
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
- [src/company.py](src/company.py): resolves a ticker and ingests it on demand, with progress reporting
- [src/generate.py](src/generate.py): Claude writes the answer from retrieved evidence, with every quote verified
- [src/generate_gemini.py](src/generate_gemini.py): the same contract on Gemini's free tier
- [src/backend.py](src/backend.py): picks whichever generation backend is configured
- [evaluate.py](evaluate.py): scores the pipeline against the reviewed golden set, with baseline tracking
- [tests/test_grounded_qa.py](tests/test_grounded_qa.py): chunking, retrieval, grounding, and vector-store contract tests
- [tests/test_generate.py](tests/test_generate.py): quote verification and generation wiring, all offline
- [tests/golden_set.json](tests/golden_set.json): 30 reviewed evaluation cases, with known gaps classified and explained
- [tests/test_evaluate.py](tests/test_evaluate.py): the scoring harness's own arithmetic, and golden-set hygiene

## Six-week build roadmap

See [PROGRESS.md](PROGRESS.md) for a pipeline diagram and a full week-by-week
account of what is built, what is pending, and the known weaknesses.

| Week | Focus | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | SEC EDGAR filings | Download a defined company scope: 10-K, 10-Q, 8-K, and DEF 14A filings, with source URLs and filing metadata. | Complete - NVCT filings download locally. |
| 2 | Ingestion and chunking | Parse downloaded filing HTML into section-aware chunks that retain ticker, form, filing date, document name, and SEC source URL. | Core complete - section-aware chunks are loaded into the app. |
| 3 | Vector store and metadata | Persist chunks and metadata in Chroma, with a stable chunk ID, filing accession number, section, period, and source URL. | Complete - `src/build_index.py` writes a persistent Chroma store; every chunk carries a stable ID and full filing metadata. |
| 4 | Hybrid retrieval | Combine lexical/keyword search with vector similarity search, then rerank the strongest evidence chunks. | Complete - table-aware chunking, BM25 + vector search fused with reciprocal rank fusion, two-gate abstention. Live in the app. |
| 5 | Grounded generation | Add Claude orchestration that answers only from retrieved evidence and can abstain when evidence is inadequate. | Complete - two backends sharing one contract. Run on Gemini: closes 3 recorded gaps, specificity 54% -> 77%. |
| 6 | Cited answers and evaluation | Show quoted evidence and source links in the UI; build a reviewed golden set and measure citation correctness, support, and abstention quality. | Complete - `evaluate.py` scores a 30-case set against a tracked baseline. 24/30 on NVCT; portable subset scored across 6 companies. |

### Current position

All six weeks are built, and the app has since been opened up from one
hard-coded ticker to any US company searched on demand. Retrieval over the
Chroma index feeds a generation step that writes the answer from the retrieved
evidence, every quote is checked against the source before it is shown, and
`evaluate.py` scores the whole thing across six companies against a tracked
baseline.

**Measured, in extractive mode: 24/30 on NVCT.** Recall is perfect on every
company tested (90/90 across six), and every single failure is a question that
should have been *declined*.

**Generation has now run** on Gemini's free tier (2026-09-06) and closed three
recorded gaps, including both cases where retrieval finds evidence about a
different subject entirely. Specificity on NVCT went 54% → 77%, with zero
fabricated quotes. It also exposed that the evaluation metric rewards answering
with wrong evidence over honestly declining - see [PROGRESS.md](PROGRESS.md).
The Claude backend remains untested. Without a key the app answers with filing
excerpts and says so in the status line.

### Still open

- Run generation for real and record a second baseline. All seven open gaps are
  exactly the shape it was built to close.
- Quote verification proves a quote is genuine, not that it supports the claim
  built on it. Measuring that needs a judge.
- **The retrieval thresholds do not generalize, and this is now measured.**
  Scoring the portable cases against six companies gives 100% recall every time,
  but specificity of 75% on NVCT against 38-50% on the rest - NVCT being the
  company the thresholds were tuned on. On five of six, no distance floor
  separates the questions that should be answered from those that should not;
  the ranges overlap. See [PROGRESS.md](PROGRESS.md). This makes the unrun
  generation step the only remaining mechanism that could fix abstention.

## Searching any company

Type a ticker and the app resolves it against EDGAR, downloads its two most
recent 10-K, 10-Q, 8-K and DEF 14A filings, chunks them, and indexes them —
about 30–90 seconds, with a progress bar. It is then a company you can pick from
the dropdown, and everything is on disk for next time.

**Set your SEC User-Agent first.** The SEC requires requests to identify who is
making them, so the app refuses to fetch anything without it:

```powershell
$env:SEC_USER_AGENT = "Your Name your-email@example.com"
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\app.py
```

Without it the app still runs and answers questions about companies already
indexed; the Add button is just disabled, with the reason shown.

### How scoping works

Every read is filtered by ticker, in both halves of retrieval:

- the vector query passes `where={"ticker": ...}` to Chroma
- the lexical scorer is handed only that company's chunks

That matters for more than tidiness. BM25 judges how rare a term is *within the
corpus it is given* — so scoping also keeps "rare" meaning rare **in this
company's filings**, rather than across every company you happen to have loaded.

The CLIs take `--ticker` too, and pick the company automatically when only one is
indexed. `evaluate.py` and `run_demo.py` scope themselves to the ticker recorded
in the golden set, so scores stay comparable as you add companies.

### What "any company" does not cover

EDGAR's `company_tickers.json` is the lookup, which means US filers with a listed
symbol. Foreign private issuers filing 20-F, most funds, and private companies
will not resolve — the app says so rather than returning nothing.

Each company costs roughly 640 chunks and a few MB of raw HTML on disk.

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

## Week 5: grounded generation with Claude

Retrieval matches words, not meaning, so it will return real, on-topic-looking
text for a question about a different company or a different kind of fact.
[src/generate.py](src/generate.py) is the step that reads the evidence and
decides.

- **Structured output** — `client.messages.parse()` with a Pydantic
  `GroundedAnswer` model (`supported`, `answer`, `citations`,
  `reason_if_unsupported`), so abstention is a boolean rather than a phrase to
  pattern-match.
- **The model never handles citation metadata.** It returns an evidence block id
  and a quote; section, form, dates, and source URL are looked up locally from
  the id.
- **Every quote is verified before display.** A quote that does not appear
  verbatim in the chunk it cites is dropped, and an answer left with no verified
  citation is downgraded to an abstention. A fabricated quote cannot reach the
  user.
- **The generator can only narrow.** It runs after retrieval and sees only
  chunks retrieval already approved, so it can reject evidence but never
  introduce any.
- **Failure is contained.** An API error, timeout, or refusal falls back to the
  Week 4 extractive answer — degraded, but still cited.

### Two interchangeable backends

Generation runs through either Claude or Gemini. They share the prompt, the
answer schema, and - most importantly - the quote verification, so the grounding
contract cannot drift between them. Only the API call differs.

| | module | key | cost |
| --- | --- | --- | --- |
| Claude | [src/generate.py](src/generate.py) | `ANTHROPIC_API_KEY` | pay-as-you-go, ~2c/question |
| Gemini | [src/generate_gemini.py](src/generate_gemini.py) | `GEMINI_API_KEY` | free tier covers the whole golden set |

[src/backend.py](src/backend.py) picks whichever is configured. A Gemini key is
free from <https://aistudio.google.com/apikey>:

```powershell
$env:GEMINI_API_KEY = "..."
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\app.py
```

The status line changes from `answers excerpted from filings` to
`answers written by gemini (...) from cited evidence`.

Override the model with `SEC_SIMPLIFIER_GEMINI_MODEL` or `SEC_SIMPLIFIER_MODEL`;
force a backend with `SEC_SIMPLIFIER_BACKEND=claude|gemini|none`. With no key
everything runs in extractive mode, which is how every measurement in
[PROGRESS.md](PROGRESS.md) was taken.

> **Not yet verified against the live API.** This path has only been exercised
> with injected fakes and a stub client — see [PROGRESS.md](PROGRESS.md) for what
> is and is not tested, and how to check it.

## Week 6: evaluation, and what it found

Building the evaluation set surfaced a silent data-loss bug. The chunker
recognized only `Item N.` headings — and a proxy statement contains none. Both
DEF 14A filings had been falling through to a whole-document fallback that
truncated at 2,000 characters:

| | before | after |
| --- | --- | --- |
| DEF 14A chunks | 2 | 110 |
| DEF 14A text indexed | ~2% | all of it |
| corpus total | 529 | 637 |

Executive compensation, director biographies, auditor fees and related-party
transactions all live in the proxy, so none of them had been reachable. The fix
falls back to the proxy's ALL-CAPS headings when no `Item` heading is found. It
is a fallback, so it cannot fragment a 10-K on an all-caps table caption — the
10-K and 10-Q chunk counts are unchanged to the chunk.

Adding the proxy also caused a regression, which the harness caught on the next
run: one question stopped citing the section it should. Three candidate fixes
were swept against the whole set before concluding, and none changed any metric,
so it is recorded as a known ranking gap with the diagnosis rather than tuned
around. [PROGRESS.md](PROGRESS.md) has the numbers.

## Install dependencies

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
```

`chromadb` pulls in `onnxruntime` and downloads a ~80 MB embedding model on first
use. If you only need the lexical baseline, the app and its tests still run
without it — the golden-set tests skip and the app reports keyword-only mode.

The two generation SDKs are optional in the same way. Without either - or
without a key for one - the app answers with filing excerpts instead of written
answers, and says so in the status line.

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

77 tests: contract tests in `tests/test_grounded_qa.py`, generation and
quote-verification tests in `tests/test_generate.py`, scoring-harness tests in
`tests/test_evaluate.py`, plus the reviewed evaluation cases in
[tests/golden_set.json](tests/golden_set.json) run by `tests/test_golden_set.py`. The golden-set tests skip cleanly until the index has
been built with `python -m src.build_index`.

None of these call the Claude API. With `ANTHROPIC_API_KEY` set, the golden-set
tests pick the generator up automatically and start costing money — watch
`test_known_gaps_are_still_recorded_as_gaps`, which fails when a known gap starts
passing.

## Ask one question

`ask.py` is the quickest way to try the pipeline. It shows every stage - what
retrieval selected, what the answer was, which quotes survived verification, and
what the call cost. **One question is one API call**, so this is the cheap way to
test generation.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\ask.py "How much cash does the company have on hand?"
```

```
RETRIEVED 3 chunk(s):
  - [prose] Item 7. Management's Discussion and Analysis (10-K 2026-02-11)
  - [table] Item 2. Management's Discussion and Analysis (10-Q 2026-05-05)

ANSWER  [WRITTEN BY CLAUDE]
  The company reports it held $31.6 million in cash and cash equivalents as of
  December 31, 2025.

VERIFIED QUOTES
  Item 7. Management's Discussion and Analysis   10-K filed 2026-02-11
    "As of December 31, 2025, we had $31.6 million of cash and cash equivalents"
    https://www.sec.gov/Archives/edgar/data/.../tmb-20251231x10k.htm

  3,120 in / 210 out tokens on claude-opus-5 - about $0.0209
```

Run it with no question for an interactive prompt, or `--no-generate` for
retrieval only with no API call. Without a key it runs in extractive mode, which
is a useful contrast: for the question above the extractive answer leads with
unrelated text about R&D expenses, and buries the $31.6 million figure in the
middle of a 500-character excerpt.

If a quote the model returns is not found in the evidence it cited, it appears
under `REJECTED QUOTE(S)` and does not count toward the answer.

## Score the pipeline

`evaluate.py` runs the 26 reviewed cases in
[tests/golden_set.json](tests/golden_set.json) and reports how well the system
answers, cites, and declines.

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\evaluate.py
```

```
ANSWER RATE
  answered when it should     13/13     100%
  declined when it should      6/13      46%
  overall                     19/26      73%

CITATIONS
  answers that cite anything  20/20     100%
  cited the expected section   4/5       80%
```

Answering and declining are reported separately because they fail
independently - and here they fail very differently. The system answers
everything it should; it declines less than half of what it should. All seven
failures are questions it should have refused, each recorded in the golden set
with a diagnosis and a class:

- `subject_mismatch` - the evidence is about a different company or person
  ("How many patents does Apple hold?").
- `fact_not_stated` - the evidence is on-topic but the figure is absent
  ("How much did the company spend on advertising?").

`--save-baseline` records a run to `tests/eval_baseline.json`; later runs print
the delta, so a change that helps one metric and quietly costs another shows up.
`--no-generate` scores retrieval only, with no API cost.

### Scoring a different company

Twenty-three of the 30 cases are marked `portable` - their expectation holds for any
US filer - so they can be scored against any loaded company:

```powershell
& "C:\Users\yceri\AppData\Local\Programs\Python\Python310\python.exe" .\evaluate.py --ticker AAPL
```

Doing this is what showed the retrieval thresholds were fit to one company:

| Company | recall | specificity | overall |
| --- | --- | --- | --- |
| NVCT | 100% | 75% | 21/23 |
| AAPL | 100% | 38% | 18/23 |
| QURE | 100% | 50% | 19/23 |
| MSFT | 100% | 38% | 18/23 |
| JPM  | 100% | 50% | 19/23 |
| WMT  | 100% | 38% | 18/23 |

Cases naming this company's drug, and "How many patents does Apple hold?" - a
fair question when the loaded company *is* Apple - are deliberately not portable.

## Run the full comparison

`run_demo.py` answers every case in the golden set twice - once with keyword
search alone, once with hybrid retrieval - and prints both verdicts side by side.
It needs the index built first. With a key set this is 13 API calls; add
`--no-generate` to keep it free.

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
