"""Build an answerable corpus from filings previously downloaded from EDGAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .grounded_qa import build_chunks_from_html


def chunks_for_filings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn saved filing metadata into section-aware chunks.

    Records missing their downloaded file are skipped rather than failing the
    batch: one unavailable document should not cost a company its other filings.
    """
    chunks: list[dict[str, Any]] = []
    for record in records:
        local_path = Path(record["local_path"])
        if not local_path.is_file():
            continue
        html_text = local_path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(
            build_chunks_from_html(
                ticker=record["ticker"],
                form=record["form"],
                filing_date=record["filing_date"],
                source_url=record["source_url"],
                html_text=html_text,
                document_name=record["primary_document"],
                accession_number=record.get("accession_number", ""),
                period_of_report=record.get("period_of_report", ""),
            )
        )
    return chunks


def load_local_corpus(data_dir: Path, ticker: str | None = None) -> list[dict[str, Any]]:
    """Load saved filing metadata and convert the corresponding HTML into chunks.

    With no ticker this loads every company that has been ingested, which is what
    the index builder wants. The app scopes reads by ticker instead.
    """
    metadata_dir = data_dir / "metadata"
    pattern = f"{ticker.upper()}_filings.json" if ticker else "*_filings.json"

    chunks: list[dict[str, Any]] = []
    for metadata_path in sorted(metadata_dir.glob(pattern)):
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks.extend(chunks_for_filings(records))
    return chunks
