"""Tests for multi-company support: scoping, ticker handling, and on-demand load.

Nothing here touches the network. The SEC client is stubbed, so the ingestion
flow - including the partial-failure path - is exercised offline.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from src import company
from src.company import CompanyError, normalize_ticker
from src.sec_client import FilingRecord


# --- Ticker handling --------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [("aapl", "AAPL"), ("  msft ", "MSFT"), ("BRK-B", "BRK-B")])
def test_tickers_are_normalized(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not a ticker", "TOOLONGTICKER", "<script>"])
def test_obvious_non_tickers_are_rejected(raw):
    with pytest.raises(CompanyError):
        normalize_ticker(raw)


def test_missing_user_agent_is_a_clear_setup_error(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(CompanyError) as error:
        company.user_agent()

    assert "SEC_USER_AGENT" in str(error.value)


# --- Ingestion, against a stubbed SEC client --------------------------------


FILING_HTML = (
    "<html><body><h1>Item 1. Business</h1>"
    "<p>Example Corp develops industrial sensors for factory automation.</p>"
    "<h2>Item 1A. Risk Factors</h2>"
    "<p>We depend on a small number of large customers for most of our revenue.</p>"
    "</body></html>"
)


class _StubSecClient:
    """Stands in for SecClient. `fail_on` marks a form whose download errors."""

    def __init__(self, tmp_path, forms=("10-K", "10-Q"), fail_on=None):
        self.tmp_path = tmp_path
        self.forms = forms
        self.fail_on = fail_on

    def get_cik_for_ticker(self, ticker):
        if ticker.upper() != "EXMP":
            raise ValueError("not found")
        return "0000000123"

    def company_name_for_ticker(self, ticker):
        return "Example Corp"

    def get_filings(self, ticker, forms=(), limit_per_form=2):
        return [
            FilingRecord(
                ticker=ticker.upper(), cik="0000000123", form=form,
                filing_date="2026-02-11", accession_number=f"0000000123-26-0000{index}",
                primary_document=f"exmp-{form}.htm",
                source_url=f"https://www.sec.gov/Archives/{form}.htm",
                period_of_report="2025-12-31",
            )
            for index, form in enumerate(self.forms)
        ]

    def download_filing(self, record, raw_dir):
        if record.form == self.fail_on:
            raise RuntimeError("404 Not Found")
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{record.ticker}_{record.form}_{record.accession_number}.htm"
        path.write_text(FILING_HTML, encoding="utf-8")
        return path


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Point ingestion at a temp data dir and a temp Chroma store."""
    monkeypatch.setattr(company, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(company, "METADATA_DIR", tmp_path / "metadata")
    indexed: list = []
    monkeypatch.setattr(company, "build_vector_store", lambda chunks: indexed.extend(chunks))
    return tmp_path, indexed


def test_resolve_reports_the_registrant_name(isolated_data, tmp_path):
    resolved = company.resolve("exmp", client=_StubSecClient(tmp_path))

    assert resolved == {"ticker": "EXMP", "cik": "0000000123", "name": "Example Corp"}


def test_unknown_ticker_explains_the_coverage_limit(tmp_path):
    with pytest.raises(CompanyError) as error:
        company.resolve("ZZZZ", client=_StubSecClient(tmp_path))

    assert "EDGAR" in str(error.value)


def test_ingest_downloads_chunks_and_indexes(isolated_data, tmp_path):
    data_dir, indexed = isolated_data
    steps: list[tuple[str, int]] = []

    summary = company.ingest(
        "EXMP",
        progress=lambda message, percent: steps.append((message, percent)),
        client=_StubSecClient(tmp_path),
    )

    assert summary["ticker"] == "EXMP"
    assert summary["name"] == "Example Corp"
    assert summary["filings"] == 2
    assert summary["chunks"] > 0

    # Every indexed chunk is tagged with the company it came from - this is what
    # makes scoped retrieval possible.
    assert {chunk["ticker"] for chunk in indexed} == {"EXMP"}

    # Metadata is written where load_local_corpus expects to find it.
    saved = json.loads((data_dir / "metadata" / "EXMP_filings.json").read_text(encoding="utf-8"))
    assert len(saved) == 2

    # Progress runs forward and finishes.
    assert steps[-1][1] == 100
    assert [percent for _, percent in steps] == sorted(percent for _, percent in steps)


def test_one_unavailable_filing_does_not_lose_the_others(isolated_data, tmp_path):
    _, indexed = isolated_data

    summary = company.ingest(
        "EXMP", client=_StubSecClient(tmp_path, forms=("10-K", "10-Q"), fail_on="10-Q")
    )

    assert summary["filings"] == 1
    assert indexed


def test_ingest_fails_clearly_when_nothing_downloads(isolated_data, tmp_path):
    with pytest.raises(CompanyError) as error:
        company.ingest("EXMP", client=_StubSecClient(tmp_path, forms=("10-K",), fail_on="10-K"))

    assert "Could not download" in str(error.value)


def test_ingest_fails_clearly_when_there_are_no_filings(isolated_data, tmp_path):
    with pytest.raises(CompanyError) as error:
        company.ingest("EXMP", client=_StubSecClient(tmp_path, forms=()))

    assert "no recent" in str(error.value)


# --- Company scoping in the store -------------------------------------------


def test_reads_are_scoped_to_one_company(tmp_path):
    pytest.importorskip("chromadb")
    from src.vector_store import (
        build_vector_store,
        indexed_companies,
        load_all_chunks,
        query_vector_store,
    )

    persist_dir = tmp_path / "chroma"
    build_vector_store(
        [
            {
                "chunk_id": "a:business:0", "ticker": "AAAA", "form": "10-K",
                "filing_date": "2026-01-01", "period_of_report": "2025-12-31",
                "accession_number": "acc-a", "section": "Item 1. Business",
                "text": "AAAA manufactures industrial sensors for factories.",
                "source_url": "https://example.com/a", "document_name": "a.htm",
                "chunk_type": "text", "table_title": "", "chunk_index": 0,
            },
            {
                "chunk_id": "b:business:0", "ticker": "BBBB", "form": "10-K",
                "filing_date": "2026-01-01", "period_of_report": "2025-12-31",
                "accession_number": "acc-b", "section": "Item 1. Business",
                "text": "BBBB manufactures industrial sensors for factories.",
                "source_url": "https://example.com/b", "document_name": "b.htm",
                "chunk_type": "text", "table_title": "", "chunk_index": 0,
            },
        ],
        persist_dir=persist_dir,
    )

    assert sorted(indexed_companies(persist_dir)) == ["AAAA", "BBBB"]
    assert len(load_all_chunks(persist_dir)) == 2
    assert [c["ticker"] for c in load_all_chunks(persist_dir, ticker="AAAA")] == ["AAAA"]

    # The two companies say near-identical things, so an unscoped search would
    # happily return the wrong one.
    hits = query_vector_store(
        "who makes industrial sensors", persist_dir=persist_dir, limit=5, ticker="BBBB"
    )
    assert hits
    assert {hit["ticker"] for hit in hits} == {"BBBB"}


def test_load_runs_on_a_thread_and_reports_progress(monkeypatch):
    """The app's job plumbing, without touching the network."""
    import app

    seen: list[tuple[str, int]] = []

    def fake_ingest(ticker, progress=None, **kwargs):
        for message, percent in [("Looking up", 5), ("Downloading", 50), ("Indexing", 90)]:
            progress(message, percent)
            seen.append((message, percent))
        return {"ticker": ticker}

    monkeypatch.setattr("src.company.ingest", fake_ingest)
    app._set_load_state(ticker=None, message="", percent=0, done=True, error=None)

    app.start_company_load("exmp")
    for _ in range(200):  # the worker is a thread; wait for it to finish
        if app.load_state()["done"]:
            break
        time.sleep(0.01)

    state = app.load_state()
    assert state["done"] is True
    assert state["error"] is None
    assert state["ticker"] == "EXMP"
    assert state["percent"] == 100
    assert seen == [("Looking up", 5), ("Downloading", 50), ("Indexing", 90)]


def test_a_failed_load_surfaces_its_message_rather_than_hanging(monkeypatch):
    import app

    def exploding(ticker, progress=None, **kwargs):
        raise CompanyError("EDGAR is unreachable")

    monkeypatch.setattr("src.company.ingest", exploding)
    app._set_load_state(ticker=None, message="", percent=0, done=True, error=None)

    app.start_company_load("EXMP")
    for _ in range(200):
        if app.load_state()["done"]:
            break
        time.sleep(0.01)

    assert app.load_state()["error"] == "EDGAR is unreachable"


def test_only_one_company_loads_at_a_time(monkeypatch):
    """Two concurrent ingests would write the same Chroma collection."""
    import app

    release = threading.Event()

    def slow(ticker, progress=None, **kwargs):
        release.wait(timeout=5)
        return {"ticker": ticker}

    monkeypatch.setattr("src.company.ingest", slow)
    app._set_load_state(ticker=None, message="", percent=0, done=True, error=None)

    app.start_company_load("EXMP")
    second = app.start_company_load("OTHR")

    assert "error" in second and "one at a time" in second["error"].lower()
    release.set()
    for _ in range(200):
        if app.load_state()["done"]:
            break
        time.sleep(0.01)


def test_lexical_index_keeps_several_companies_cached():
    """Switching companies must not re-tokenize on every question."""
    from src.grounded_qa import _INDEX_CACHE, _lexical_index

    _INDEX_CACHE.clear()
    first = [{"chunk_id": "a:0", "text": "alpha beta gamma delta"}]
    second = [{"chunk_id": "b:0", "text": "epsilon zeta eta theta"}]

    _lexical_index(first)
    _lexical_index(second)
    assert len(_INDEX_CACHE) == 2

    # Returning to the first corpus is a cache hit, not a rebuild.
    before = len(_INDEX_CACHE)
    _lexical_index(first)
    assert len(_INDEX_CACHE) == before
