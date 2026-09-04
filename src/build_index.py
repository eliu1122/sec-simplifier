"""Build or refresh the local Chroma vector store from downloaded EDGAR filings.

Run this after `python -m src.ingest_filings`. It loads the section-aware chunks,
upserts them into `data/chroma/` keyed by stable chunk ID, and prints what is
now persisted.

    "C:\\Users\\yceri\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m src.build_index
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

# Filing text carries typographic characters (curly quotes, zero-width spaces)
# that the default Windows console codec cannot encode.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

from .corpus import load_local_corpus
from .vector_store import DEFAULT_PERSIST_DIR, build_vector_store, query_vector_store, store_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> object:
    parser = ArgumentParser(description="Persist filing chunks into the Chroma vector store.")
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory for the Chroma store (default: data/chroma).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection first. Use after changing how filings are chunked.",
    )
    parser.add_argument(
        "--probe",
        metavar="QUESTION",
        help="After indexing, run one similarity query and print the top hits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_local_corpus(DATA_DIR)
    if not chunks:
        raise SystemExit(
            "No local filings found. Run `python -m src.ingest_filings` first."
        )

    tables = sum(1 for chunk in chunks if chunk.get("chunk_type") == "table")
    print(f"Loaded {len(chunks)} chunks from local filings ({tables} tables, {len(chunks) - tables} prose).")
    total = build_vector_store(chunks, persist_dir=args.persist_dir, reset=args.reset)
    stats = store_stats(persist_dir=args.persist_dir)
    print(f"Vector store now holds {total} chunks across {stats['filings']} filings.")
    print(f"  Tickers: {', '.join(stats['tickers']) or '-'}")
    print(f"  Forms:   {', '.join(stats['forms']) or '-'}")

    if args.probe:
        print(f"\nTop matches for: {args.probe!r}")
        for hit in query_vector_store(args.probe, persist_dir=args.persist_dir, limit=3):
            print(
                f"  [{hit['distance']:.3f}] {hit['form']} {hit['filing_date']} "
                f"| {hit['section']}"
            )
            print(f"      {hit['text'][:160].strip()}...")


if __name__ == "__main__":
    main()
