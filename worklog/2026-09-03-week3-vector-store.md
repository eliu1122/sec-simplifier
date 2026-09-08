# 2026-09-03 — Week 3: vector store and metadata

Commit: `a8d6cf3` · 12 files, +945/−181

## Goal

Persist chunks in Chroma with a stable chunk ID, accession number, section,
period and source URL.

## Built

- **Stable chunk IDs** — `accession:section-slug:position`. Re-running ingestion
  upserts the same rows rather than duplicating them.
- **Fuller citation metadata** — `accession_number` and `period_of_report`, the
  latter from EDGAR's `reportDate` with a fallback to the filing date, since
  EDGAR leaves it blank on some 8-Ks. This is what year-over-year comparison
  would later key on.
- **Persistent Chroma collection** — cosine similarity, Chroma's default
  on-device embedding model (`all-MiniLM-L6-v2`). Nothing leaves the machine.
- **`src/build_index.py`** with a `--probe` flag to inspect how similarity search
  ranks a question.

## Deliberately not done

The store was persistence only. The app kept answering through the Week 2
lexical scorer until Week 4, so that hybrid retrieval could be introduced as one
change with the abstention contract intact.

## Notes

Chroma's default embedding model downloads ~80 MB on first use. Left as the
default rather than an API-based embedder, because "runs entirely on your
computer" is a property worth keeping for a filings tool.
