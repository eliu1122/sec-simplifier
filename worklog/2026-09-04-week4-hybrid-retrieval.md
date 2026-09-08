# 2026-09-04 — Week 4: hybrid retrieval and table-aware chunking

Commit: `5dafcf2` · 11 files, +1325/−182

## Built

**Table-aware chunking** (`src/tables.py`). A filing uses `<table>` for two
unrelated jobs: financial data and page layout. The 10-Q has 207 tables, of
which 17 carry data. Data tables are lifted out, captioned from the nearest
preceding prose, and rendered one row per line so a figure keeps the label that
explains it.

**Hybrid retrieval.** Lexical and vector halves are gated independently, then
fused with reciprocal rank fusion — which needs only rank order, so an integer
overlap count never has to be compared against a cosine distance.

**BM25 instead of raw overlap.** Raw counts favour long chunks so heavily that
`Item 4. Controls and Procedures` came back as the best evidence about cash.

## Two defects found while validating

Neither was visible until retrieval started returning junk sections:

- **Text counted once per nesting level.** The walker visited every
  `div`/`p`/`td`, so a paragraph inside three nested divs was copied three times.
  The largest chunk in the corpus was **466,558 characters** — most of the
  document, repeated. Fixed by reading only leaf block elements.
- **The table of contents created sections.** Its cells read as `Item N.`
  headings, so `Item 6. [Reserved]` ended up holding 4,967 characters of Item 7's
  MD&A.

Corpus after: 120 → 529 chunks, largest 466,558 → 3,194 characters.

## What was measured, and what was refused

The golden set was grown to 26 cases with deliberate hard negatives. Adding them
*lowered* the headline score, which was the point — the earlier set was
flattering.

One question (`What stock exchange is the company listed on?`) stopped citing
`Item 5`. Three candidate fixes were swept against the whole set — BM25's length
constant across 0.0–1.0, the vector pool across 5–80, and per-section diversity.
**None changed any metric.** Recorded as a `section_known_gap` with the
diagnosis rather than tuned around.
