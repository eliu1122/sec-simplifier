"""Week 3: persist filing chunks and their metadata in a local Chroma store.

This is the durable retrieval layer. Ingestion writes section-aware chunks here
with a stable chunk ID, accession number, section, period of report, and source
URL, so the app can query by meaning instead of rebuilding a corpus from HTML on
every request.

Chroma runs fully on disk with its default local embedding model; nothing leaves
the machine. `chromadb` is an optional dependency - callers that only need the
lexical baseline can skip installing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "filings"

# Metadata carried on every chunk. Chroma only accepts str/int/float/bool, so
# these are all scalars and missing values become "" rather than None.
METADATA_FIELDS = (
    "chunk_id",
    "chunk_index",
    "chunk_type",
    "table_title",
    "ticker",
    "form",
    "filing_date",
    "period_of_report",
    "accession_number",
    "section",
    "source_url",
    "document_name",
)


def _require_chromadb():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "The vector store needs the 'chromadb' package. Install it with:\n"
            '    "C:\\Users\\yceri\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install chromadb'
        ) from exc
    return chromadb


def get_collection(persist_dir: Path | str = DEFAULT_PERSIST_DIR, reset: bool = False):
    """Return the persistent Chroma collection, creating it if needed.

    `reset` drops the collection first. Use it when chunking changes: stable IDs
    keep re-ingestion clean, but a chunker that splits differently leaves the
    previous run's rows behind as orphans.
    """
    chromadb = _require_chromadb()
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:  # collection may not exist yet
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _metadata_for(chunk: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = chunk.get(field, "")
        if isinstance(value, bool) or isinstance(value, (int, float)):
            meta[field] = value
        else:
            meta[field] = str(value) if value is not None else ""
    return meta


def build_vector_store(
    chunks: list[dict[str, Any]],
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    reset: bool = False,
) -> int:
    """Upsert chunks into the store keyed by their stable chunk ID.

    Re-running this after a fresh ingestion replaces the same rows instead of
    duplicating them. Returns the total row count after the write.
    """
    collection = get_collection(persist_dir, reset=reset)

    # Keep the last chunk seen for any repeated ID; Chroma rejects a batch that
    # contains the same ID twice.
    by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = (chunk.get("text") or "").strip()
        if not chunk_id or not text:
            continue
        by_id[chunk_id] = chunk

    if by_id:
        collection.upsert(
            ids=list(by_id),
            documents=[by_id[cid]["text"] for cid in by_id],
            metadatas=[_metadata_for(by_id[cid]) for cid in by_id],
        )
    return collection.count()


def _ticker_filter(ticker: str | None) -> dict[str, Any] | None:
    """Chroma `where` clause scoping a read to one company, or None for all."""
    return {"ticker": ticker.upper()} if ticker else None


def query_vector_store(
    question: str,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    limit: int = 3,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most similar chunks as plain dicts (metadata + text + distance).

    `ticker` scopes the search to one company. Without it a store holding several
    companies would answer a question about one using another's filings.
    """
    collection = get_collection(persist_dir)
    count = collection.count()
    if not question.strip() or count == 0:
        return []

    result = collection.query(
        query_texts=[question],
        n_results=min(limit, count),
        where=_ticker_filter(ticker),
    )
    hits: list[dict[str, Any]] = []
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        hit = dict(metadata)
        hit["text"] = document
        hit["distance"] = float(distance)
        hits.append(hit)
    return hits


def load_all_chunks(
    persist_dir: Path | str = DEFAULT_PERSIST_DIR, ticker: str | None = None
) -> list[dict[str, Any]]:
    """Return persisted chunks as plain dicts, optionally for one company only.

    The lexical half of hybrid retrieval scores the whole corpus, so it reads
    from the store too - that way the index is the single source of truth and
    the app never re-parses filing HTML to answer a question. Pass `ticker` to
    scope it; the lexical scorer has no metadata filter of its own, so this is
    where company scoping happens for that half.
    """
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return []

    rows = collection.get(include=["documents", "metadatas"], where=_ticker_filter(ticker))
    chunks: list[dict[str, Any]] = []
    for document, metadata in zip(rows["documents"], rows["metadatas"]):
        chunk = dict(metadata)
        chunk["text"] = document
        chunks.append(chunk)
    return chunks


def indexed_companies(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> dict[str, dict[str, Any]]:
    """What is already indexed, keyed by ticker.

    Used to decide whether a searched company needs ingesting, and to populate
    the picker in the UI.
    """
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return {}

    rows = collection.get(include=["metadatas"])
    companies: dict[str, dict[str, Any]] = {}
    for metadata in rows["metadatas"]:
        ticker = metadata.get("ticker")
        if not ticker:
            continue
        entry = companies.setdefault(
            ticker, {"ticker": ticker, "chunks": 0, "filings": set(), "forms": set()}
        )
        entry["chunks"] += 1
        if metadata.get("accession_number"):
            entry["filings"].add(metadata["accession_number"])
        if metadata.get("form"):
            entry["forms"].add(metadata["form"])

    for entry in companies.values():
        entry["filings"] = len(entry["filings"])
        entry["forms"] = sorted(entry["forms"])
    return companies


def store_stats(persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> dict[str, Any]:
    """Small summary of what is persisted, for CLI output and status checks."""
    collection = get_collection(persist_dir)
    count = collection.count()
    if count == 0:
        return {"chunks": 0, "tickers": [], "forms": [], "filings": 0}

    rows = collection.get(include=["metadatas"])
    metadatas = rows["metadatas"]
    return {
        "chunks": count,
        "tickers": sorted({m.get("ticker", "") for m in metadatas if m.get("ticker")}),
        "forms": sorted({m.get("form", "") for m in metadatas if m.get("form")}),
        "filings": len(
            {m.get("accession_number", "") for m in metadatas if m.get("accession_number")}
        ),
    }
