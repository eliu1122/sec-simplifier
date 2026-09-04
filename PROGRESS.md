# SEC Simplifier — Build Progress

Status of the six-week build as of **2026-09-03** (end of Week 4). The project is a single-company
(ticker **NVCT**) grounded-Q&A MVP over EDGAR filings: it answers natural-language
questions using filing text, shows the supporting excerpt and source link, and
says *"I don't see this disclosed"* when the filings don't support an answer.

- Detailed setup and run instructions: [README.md](README.md)
- Active code lives in this folder (`sec-simplifier/`). `../edgar_copilot/` is a
  stale earlier copy and is not maintained.

---

## The pipeline

```mermaid
flowchart TD
    A["<b>1 · SEC EDGAR filings</b><br/>10-K, 10-Q, 8-K, DEF 14A<br/><i>fetch + filing metadata</i>"]
    B["<b>2 · Ingestion &amp; chunking</b><br/>section-aware and table-aware splits<br/>bounded to 3,000 characters"]
    C["<b>3 · Vector store &amp; metadata</b><br/>Chroma · stable chunk IDs<br/>per-filing citation metadata"]
    D["<b>4 · Hybrid retrieval</b><br/>BM25 + vector search, RRF fusion<br/>two gates guard abstention"]
    E["<b>5 · Claude orchestration</b><br/>grounded generation<br/>answers only from evidence"]
    F["<b>6 · Cited answer</b><br/>quotes the source filing<br/>or flags the disclosure gap"]

    A --> B --> C --> D --> E --> F

    classDef done fill:#e4f7eb,stroke:#146b38,color:#0b3d20;
    classDef partial fill:#fff7e0,stroke:#9a6b00,color:#5a3d00;
    classDef planned fill:#eef2ff,stroke:#3f51b5,color:#22307a;

    class A,B,C,D done;
    class E planned;
    class F partial;
```

**Legend** — 🟢 done · 🟡 partial (baseline in place, work remaining) · 🔵 planned

| Stage | State | Where |
| --- | --- | --- |
| 1 · SEC EDGAR filings | 🟢 done | [src/sec_client.py](src/sec_client.py), [src/ingest_filings.py](src/ingest_filings.py) |
| 2 · Ingestion & chunking | 🟢 done | [src/grounded_qa.py](src/grounded_qa.py), [src/tables.py](src/tables.py), [src/corpus.py](src/corpus.py) |
| 3 · Vector store & metadata | 🟢 done | [src/vector_store.py](src/vector_store.py), [src/build_index.py](src/build_index.py) |
| 4 · Hybrid retrieval | 🟢 done | [src/grounded_qa.py](src/grounded_qa.py), [app.py](app.py) |
| 5 · Grounded generation (Claude) | 🔵 next | — |
| 6 · Cited answers & evaluation | 🟡 UI + abstention live; golden set scaffolded, scoring pending | [tests/golden_set.json](tests/golden_set.json) |

---

## Week 1 — SEC EDGAR filings 🟢

**Goal:** download a defined company scope with source URLs and filing metadata.

Done:

- [src/sec_client.py](src/sec_client.py): ticker → CIK lookup via
  `company_tickers.json`, recent-filings lookup via the `submissions` API, and
  primary-document download. Polite `User-Agent` enforcement and request delay to
  stay within SEC rate guidance.
- [src/ingest_filings.py](src/ingest_filings.py): CLI that pulls the *N* most
  recent 10-K, 10-Q, 8-K, and DEF 14A filings for a ticker.
- Raw HTML saved under `data/raw/`; a metadata record per filing (ticker, CIK,
  form, filing date, period of report, accession number, primary document,
  source URL) saved to `data/metadata/<TICKER>_filings.json`.
- Current corpus: **8 NVCT filings** (2× each form) → **120 chunks**.

---

## Week 2 — Ingestion, chunking, and the grounding baseline 🟡

**Goal:** parse filing HTML into section-aware chunks and answer questions only
from that text, with citations and honest abstention.

Done:

- **Section-aware chunking** ([`build_chunks_from_html`](src/grounded_qa.py)):
  splits filing HTML on `Item N.` headings, keeping ticker, form, filing date,
  section heading, document name, and SEC source URL on every chunk. Falls back
  to a cleaned full-text chunk when a filing has no recognizable item headings.
- **Transparent lexical retrieval**
  ([`retrieve_supporting_chunks`](src/grounded_qa.py)): scores chunks by
  meaningful query-term overlap (plural-normalized, stop-worded, ticker-symbol
  excluded so a company-wide term can't inflate a match). Returns nothing when
  the best score is below a threshold — it never falls back to unrelated text.
- **Grounded answer contract** ([`answer_question`](src/grounded_qa.py)):
  returns an answer plus citations, an evidence list (section, form, date,
  excerpt, source URL), and a plain-English *"why it matters"* note — or the
  explicit *"I don't see this disclosed in the filings available for this
  company."* response with no citations.
- **Local browser UI** ([app.py](app.py)): runs entirely on `127.0.0.1:8000`,
  auto-detects downloaded EDGAR filings (demo corpus fallback), starter
  questions, evidence cards with source links, supported/unsupported badge.
- **Contract tests** ([tests/test_grounded_qa.py](tests/test_grounded_qa.py)):
  supported vs. unsupported answers, retrieval relevance, abstention.

Not yet done (carried forward):

- The *"answer"* string is currently a truncated slice of the top chunk with a
  template prefix — real synthesis is Week 5.

Two defects in this chunker were found and fixed in Week 4 (see below): text was
being counted once per nesting level, and the table of contents was being read as
real section headings.

---

## Week 3 — Vector store and metadata 🟢

**Goal:** persist chunks and metadata in Chroma with a stable chunk ID, filing
accession number, section, period, and source URL.

Done:

- **Stable chunk IDs** ([`make_chunk_id`](src/grounded_qa.py)):
  `accession:section-slug:position`. Re-running ingestion upserts the same rows
  instead of creating duplicates.
- **Fuller citation metadata on every chunk:** added `accession_number` and
  `period_of_report` (from EDGAR's `reportDate`, falling back to the filing date
  when EDGAR leaves it blank, e.g. some 8-Ks) — not just the filing date. This is
  what year-over-year comparison will key on.
- **Persistent Chroma collection** ([src/vector_store.py](src/vector_store.py)):
  `data/chroma/`, cosine similarity, Chroma's default on-device embedding model
  (`all-MiniLM-L6-v2`, cached to `~/.cache/chroma` on first use). No filing text
  or query leaves the machine. `build_vector_store()` upserts by chunk ID;
  `query_vector_store()` returns metadata + text + distance; `store_stats()`
  summarizes what's persisted.
- **Index builder CLI** ([src/build_index.py](src/build_index.py)):
  `python -m src.build_index`, with `--probe "question"` to see how similarity
  search ranks a query against the persisted chunks.
- **Tests:** stable-ID determinism, position-awareness, vector-store round-trip
  (metadata preserved, re-index does not duplicate), empty-store behavior.
  `chromadb` tests `importorskip` so the lexical baseline still tests clean
  without the dependency.

Deliberately **not** wired into the live answer path yet: the app still answers
through the Week 2 lexical scorer. Chroma is persistence only until Week 4.

---

## Week 4 — Hybrid retrieval 🟢

**Goal:** combine lexical and vector search, rerank, and keep the abstention
contract intact.

### Table-aware chunking ([src/tables.py](src/tables.py))

A filing uses `<table>` for two unrelated jobs: real financial data and page
layout. The 10-Q has 207 tables, of which **17 carry data**. Layout tables are
left alone; data tables are lifted out, captioned, and indexed separately.

- **Detection** — at least 3 rows, at least 3 numeric cells, at least 2 columns.
  The table of contents passes that bar (page numbers are numeric), so it is
  excluded explicitly.
- **Captions** — filings rarely mark one up, so the title is the nearest
  preceding prose, skipping text that belongs to another data table.
- **Rendering** — one row per line, cells pipe-separated, so a figure keeps the
  label that explains it:

  ```
  CONDENSED BALANCE SHEETS (USD in thousands) (unaudited)
  June 30, | December 31,
  2026 | 2025
  Cash and cash equivalents | 22,160 | 31,634
  TOTAL ASSETS | 22,376 | 31,709
  ```

  Before this, that content was indexed as an unlabeled run of digits.

### Two chunker defects found while validating

Neither was visible until retrieval started returning junk sections:

- **Text was counted once per nesting level.** The walker visited every
  `div`/`p`/`td`, so a paragraph inside three nested divs was copied three times.
  `Item 1C. Cybersecurity` held 50,217 characters across two near-identical
  chunks; the largest chunk in the corpus was **466,558 characters** — most of
  the document, repeated. Fixed by reading only *leaf* block elements.
- **The table of contents was creating sections.** Its cells read as `Item N.`
  headings, so the chunker opened a section at the TOC entry and filed the front
  matter under it — `Item 6. [Reserved]` ended up holding 4,967 characters of
  Item 7's MD&A. Fixed by dropping TOC tables before the walk.

Chunks are now split on sentence boundaries at 3,000 characters. Effect on the
corpus:

| | before Week 4 | after |
| --- | --- | --- |
| chunks | 120 | 529 (75 tables, 454 prose) |
| largest chunk | 466,558 chars | 3,194 chars |
| `Item 6. [Reserved]` | 4,967 chars of MD&A | 27 chars |
| `Item 1C. Cybersecurity` | 50,217 chars (duplicated) | 15,264 chars |

### The retriever ([src/grounded_qa.py](src/grounded_qa.py))

`retrieve_hybrid()` runs both retrievers, gates each independently, then fuses
the survivors with **reciprocal rank fusion** (`Σ 1/(60 + rank)`), which needs
only rank order — so an integer overlap count never has to be compared against a
cosine distance. Results are de-duplicated by `chunk_id`.

**Ranking is BM25**, not raw overlap count. Raw counts favour long chunks: with
whole-`Item` chunks, almost any question clears the bar by chance, which is how
`Item 4. Controls and Procedures` came back as the best evidence about cash.
BM25 weights rare terms above common ones and normalizes for length. The corpus
index is memoized against a corpus fingerprint, since tokenizing 529 chunks per
question dominated the cost.

**Abstention uses two gates, and a chunk must clear at least one:**

| Gate | Rule |
| --- | --- |
| Lexical | ≥ 2 matching terms (section headings count double) **and** ≥ 60% of the question's meaningful terms covered |
| Vector | cosine distance ≤ 0.60 |

The coverage half of the lexical gate was added after measurement: a bare
two-term bar is trivially cleared, because a long filing contains *some* chunk
mentioning "patents" or "address".

### What the gates actually measure

Against the 13 golden-set cases, neither signal separates on its own:

| | should answer | should abstain |
| --- | --- | --- |
| vector distance | 0.447 – **0.599** | **0.553** – 0.779 |
| lexical gate (count only) | 1, 1, 2, 2, 8 | 1, 1, 1, 2, 3 |

The lexical count is useless as a discriminator — the ranges are identical.
Vector distance nearly separates cleanly, which is why the floor sits at 0.60,
just above the highest on-topic question.

**Current score: 11/13.** Both misses are the same shape and are recorded as
known gaps rather than tuned away:

- *"How many patents does Apple hold?"* — the filings do discuss patents, so
  retrieval finds real, on-topic-looking evidence.
- *"What is the CEO home address?"* — an address is present, just a corporate one.

No retrieval threshold can fix these: the evidence genuinely matches the words.
Distinguishing "Apple's patents" from "our patents" requires **reading** the
retrieved text, which is Week 5. Tuning thresholds until these two passed would
have overfit to thirteen hand-written questions and cost real recall.

### Wired into the app

[app.py](app.py) now answers from the vector store instead of re-parsing filing
HTML on every request, and caches the corpus for the life of the process.
Request latency went from ~2.5 s to **0.15 s**. The status line reports which
mode is live (`Hybrid search over NVCT (529 indexed chunks)`), and the app falls
back to lexical-only when `chromadb` is absent or the index has not been built.

### Evaluation scaffold

[tests/golden_set.json](tests/golden_set.json) holds 13 reviewed cases — the
question, whether the filings should support an answer, the expected section
where unambiguous, and *why*. Several record what the lexical baseline alone
does, which is what makes the hybrid win legible.
[tests/test_golden_set.py](tests/test_golden_set.py) asserts the
supported/abstain contract on non-gap cases, and fails if a known gap starts
passing so the file cannot drift out of date. Scoring is Week 6.

**Tests: 34 passing** (19 unit + 15 golden set). The golden-set module skips
cleanly when the index has not been built.

[run_demo.py](run_demo.py) answers every golden-set case both ways and prints the
verdicts side by side, which is the shortest way to see what Week 4 bought:

```
  keyword only : 9/13 correct
  hybrid       : 11/13 correct
```

---

## Future work

### Week 5 — Grounded generation (Claude) 🔵 (next)

- Claude orchestration that answers **only** from retrieved evidence and abstains
  when the evidence is inadequate.
- Replaces the current truncate-the-top-chunk placeholder with real synthesis
  that still quotes and cites.
- System-prompt guardrails against using outside knowledge; every claim traceable
  to a retrieved chunk.
- **Closes the two known golden-set gaps.** Both need a reader that can tell the
  retrieved evidence is about a different subject than the question — the
  registrant's patents rather than Apple's, a corporate address rather than a
  home one. Retrieval cannot do this; the generator can.
- Also replaces the hard-coded keyword guard in `answer_question` that special-
  cases the words "lawsuit", "settlement", and "exact" — a placeholder standing
  in for judgement the model should be making.

### Week 6 — Cited answers and evaluation 🟡

- Quoted evidence and source links in the UI (baseline exists — evidence cards
  and source links are already rendered).
- Grow the golden set beyond its current 13 cases, and cover the 10-Q and DEF 14A
  as deliberately as the 10-K.
- Score **citation correctness**, **support** (does the cited text actually back
  the answer), and **abstention quality** — turning
  [tests/golden_set.json](tests/golden_set.json) from a pass/fail contract into
  measured metrics with a tracked baseline.

---

## Known weaknesses / open risks

- **Retrieval cannot tell whose facts these are.** The two open golden-set gaps.
  A question about another company's patents, or a person's home address, finds
  evidence that genuinely matches the words. Needs Week 5.
- **Answers are not written, only excerpted.** The response is the top chunk
  truncated to 300 characters behind a template sentence. It reads as an answer
  without being one. Week 5.
- **Thresholds are tuned on 13 cases.** The 0.60 distance floor and 60% coverage
  bar separate the current golden set; they are not validated at any scale.
  Widening the set may move them.
- **Page furniture leaks into chunks.** Running headers and page numbers
  ("`None. 55 Table of Contents`") still ride along in short chunks. Harmless for
  retrieval, ugly as a citation shown to a user.
- **Split cells could drop a minus sign.** Negative figures are rendered as
  `( 13,070 )` in this filer's HTML and survive intact, but a filer that splits
  `(`, `13,070`, `)` into three cells would lose the parentheses, and with them
  the sign. Not yet defended against.
- **`period_of_report` backfill** — the existing `data/metadata` JSON predates the
  `reportDate` capture, so some filings fall back to the filing date until
  re-ingestion.
- **Single company by scope** — multi-company compare and filing-diff are
  explicitly out of scope for the MVP.
