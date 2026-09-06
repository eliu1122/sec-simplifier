"""Fetch and index a company on demand, so any ticker can be searched.

The CLI path (`src.ingest_filings` then `src.build_index`) stays the way to
prepare companies in bulk. This module is the same work driven from the app: a
user types a ticker, and if it is not indexed yet it is downloaded, chunked and
embedded while they watch.

Progress is reported through a callback rather than printed, because the caller
is a web request, not a terminal.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .corpus import chunks_for_filings
from .sec_client import SecClient
from .vector_store import build_vector_store, indexed_companies

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

FORMS = ("10-K", "10-Q", "8-K", "DEF 14A")
LIMIT_PER_FORM = 2

# EDGAR's ticker file is uppercase alphanumerics with the occasional dash.
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

ProgressFn = Callable[[str, int], None]


class CompanyError(Exception):
    """A ticker could not be resolved or ingested, with a message worth showing."""


def _noop(message: str, percent: int) -> None:
    pass


def user_agent() -> str:
    """The identifying User-Agent the SEC asks for.

    Required for any request to EDGAR, so a missing one is a setup error the
    user needs to see rather than a stack trace.
    """
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not agent:
        raise CompanyError(
            "Set SEC_USER_AGENT to your name and email before loading a company - "
            'the SEC requires it. For example: $env:SEC_USER_AGENT = "Jane Doe jane@example.com"'
        )
    return agent


def normalize_ticker(raw: str) -> str:
    ticker = (raw or "").strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise CompanyError(f"{raw!r} does not look like a ticker symbol.")
    return ticker


def resolve(ticker: str, client: SecClient | None = None) -> dict[str, str]:
    """Look the ticker up in EDGAR's mapping, returning its name and CIK."""
    ticker = normalize_ticker(ticker)
    client = client or SecClient(user_agent=user_agent())
    try:
        cik = client.get_cik_for_ticker(ticker)
    except ValueError:
        raise CompanyError(
            f"{ticker} is not in EDGAR's ticker list. That list covers US filers with a "
            "listed symbol; foreign private issuers and funds often will not appear."
        )
    return {"ticker": ticker, "cik": cik, "name": client.company_name_for_ticker(ticker)}


def is_indexed(ticker: str) -> bool:
    return normalize_ticker(ticker) in indexed_companies()


def ingest(ticker: str, progress: ProgressFn = _noop, client: SecClient | None = None) -> dict[str, Any]:
    """Download, chunk and index one company. Returns a summary.

    Safe to call for a company already indexed: chunk IDs are stable, so the
    rows are replaced rather than duplicated.
    """
    ticker = normalize_ticker(ticker)
    client = client or SecClient(user_agent=user_agent())

    progress(f"Looking up {ticker} in EDGAR", 5)
    company = resolve(ticker, client)

    progress(f"Finding recent filings for {company['name']}", 12)
    records = client.get_filings(ticker, forms=FORMS, limit_per_form=LIMIT_PER_FORM)
    if not records:
        raise CompanyError(
            f"{ticker} resolved to {company['name']}, but it has no recent "
            f"{', '.join(FORMS)} filings to read."
        )

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        # Downloading is most of the wall time, so it owns most of the bar.
        share = 15 + int(55 * position / len(records))
        progress(f"Downloading {record.form} filed {record.filing_date}", share)
        try:
            local_path = client.download_filing(record, RAW_DIR)
        except Exception as error:
            # One unavailable document should not sink the whole company.
            progress(f"Skipped {record.form} filed {record.filing_date}: {error}", share)
            continue
        entry = record.to_dict()
        entry["local_path"] = str(local_path)
        saved.append(entry)

    if not saved:
        raise CompanyError(f"Could not download any filings for {ticker}.")

    (METADATA_DIR / f"{ticker}_filings.json").write_text(
        json.dumps(saved, indent=2), encoding="utf-8"
    )

    progress(f"Reading {len(saved)} filings", 72)
    chunks = chunks_for_filings(saved)
    if not chunks:
        raise CompanyError(f"{ticker}'s filings could not be parsed into readable sections.")

    tables = sum(1 for chunk in chunks if chunk.get("chunk_type") == "table")
    progress(f"Indexing {len(chunks)} sections ({tables} tables)", 82)
    build_vector_store(chunks)

    progress(f"{company['name']} is ready", 100)
    return {
        "ticker": ticker,
        "name": company["name"],
        "cik": company["cik"],
        "filings": len(saved),
        "chunks": len(chunks),
        "tables": tables,
    }
