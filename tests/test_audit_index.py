"""Tests for the index audit: what counts as broken, and what prune is allowed to touch.

The prune path deletes from an index that is not in git, so the tests that matter
most here are the ones proving it leaves things alone: healthy companies, partial
indexes, and missing source files are all recoverable by re-ingesting, and none
of them may trigger a delete.
"""

from __future__ import annotations

import json

import pytest

from src.audit_index import audit, orphan_raw_files, prune


def _chunk(ticker: str, chunk_id: str, accession: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "form": "10-K",
        "filing_date": "2026-02-13",
        "accession_number": accession,
        "section": "Item 1A. Risk Factors",
        "source_url": "https://www.sec.gov/Archives/example",
        "text": text,
    }


def _write_metadata(data_dir, ticker: str, accessions: list[str], raw_exists: bool = True) -> None:
    """Write a metadata file, optionally with the raw files it points at."""
    metadata_dir = data_dir / "metadata"
    raw_dir = data_dir / "raw"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for accession in accessions:
        raw_path = raw_dir / f"{ticker}_10-K_2026-02-13_{accession}.htm"
        if raw_exists:
            raw_path.write_text("<html><body>filing</body></html>", encoding="utf-8")
        records.append(
            {
                "ticker": ticker,
                "form": "10-K",
                "filing_date": "2026-02-13",
                "accession_number": accession,
                "local_path": str(raw_path),
            }
        )
    (metadata_dir / f"{ticker}_filings.json").write_text(
        json.dumps(records), encoding="utf-8"
    )


@pytest.fixture
def store(tmp_path):
    """A data dir and Chroma store the tests can fill however they need."""
    pytest.importorskip("chromadb")
    from src.vector_store import build_vector_store

    data_dir = tmp_path / "data"
    persist_dir = tmp_path / "chroma"

    def index(chunks):
        build_vector_store(chunks, persist_dir=persist_dir)

    return data_dir, persist_dir, index


def test_healthy_company_has_no_problems_or_warnings(store):
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "JPM", ["0001-26-001"])
    index([_chunk("JPM", "jpm-1", "0001-26-001", "Risk factors for the bank.")])

    (report,) = audit(data_dir, persist_dir)
    assert report.ticker == "JPM"
    assert report.healthy
    assert report.problems == []
    assert report.warnings == []
    assert report.chunks == 1
    assert report.indexed_filings == 1
    assert report.metadata_records == 1


def test_metadata_without_chunks_is_broken(store):
    """The EDGAR-resolves-but-files-nothing case: metadata written, nothing indexed."""
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "SHELL", [])
    index([_chunk("JPM", "jpm-1", "0001-26-001", "Unrelated bank filing.")])

    reports = {r.ticker: r for r in audit(data_dir, persist_dir)}
    assert reports["SHELL"].healthy is False
    assert "nothing indexed" in reports["SHELL"].problems
    assert "no filings on record" in reports["SHELL"].problems
    assert reports["JPM"].healthy


def test_chunks_without_metadata_file_warn_but_are_not_broken(store):
    """Still answers questions, so it must survive a prune."""
    data_dir, persist_dir, index = store
    data_dir.mkdir(parents=True, exist_ok=True)
    index([_chunk("NVCT", "nvct-1", "0002-26-002", "Clinical trial risk.")])

    (report,) = audit(data_dir, persist_dir)
    assert report.healthy
    assert report.warnings == ["no metadata file"]
    assert prune([report], data_dir, persist_dir, dry_run=False) == []


def test_partially_indexed_company_warns_and_is_never_pruned(store):
    """Two filings downloaded, one indexed - recoverable, so not a prune target."""
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "WMT", ["0003-26-003", "0003-26-004"])
    index([_chunk("WMT", "wmt-1", "0003-26-003", "Retail segment results.")])

    (report,) = audit(data_dir, persist_dir)
    assert report.healthy
    assert report.unindexed_filings == 1
    assert "1 filing(s) downloaded but not indexed" in report.warnings
    assert prune([report], data_dir, persist_dir, dry_run=False) == []


def test_missing_source_files_warn_but_do_not_break_a_company(store):
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "AAPL", ["0004-26-005"], raw_exists=False)
    index([_chunk("AAPL", "aapl-1", "0004-26-005", "Products and services.")])

    (report,) = audit(data_dir, persist_dir)
    assert report.healthy
    assert report.missing_raw == 1
    assert "1 source file(s) missing from data/raw" in report.warnings


def test_unreadable_metadata_is_broken(store):
    data_dir, persist_dir, index = store
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "JUNK_filings.json").write_text("{not json", encoding="utf-8")

    (report,) = audit(data_dir, persist_dir)
    assert report.ticker == "JUNK"
    assert report.healthy is False
    assert "metadata file unreadable" in report.problems


def test_prune_dry_run_deletes_nothing(store):
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "SHELL", [])
    index([_chunk("JPM", "jpm-1", "0001-26-001", "Bank filing.")])
    metadata_file = data_dir / "metadata" / "SHELL_filings.json"

    reports = audit(data_dir, persist_dir)
    actions = prune(reports, data_dir, persist_dir, dry_run=True)

    assert [a["ticker"] for a in actions] == ["SHELL"]
    assert metadata_file.exists()
    assert {r.ticker for r in audit(data_dir, persist_dir)} == {"JPM", "SHELL"}


def test_prune_removes_only_the_broken_company(store):
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "SHELL", [])
    _write_metadata(data_dir, "JPM", ["0001-26-001"])
    index([_chunk("JPM", "jpm-1", "0001-26-001", "Bank filing.")])

    prune(audit(data_dir, persist_dir), data_dir, persist_dir, dry_run=False)

    assert not (data_dir / "metadata" / "SHELL_filings.json").exists()
    assert (data_dir / "metadata" / "JPM_filings.json").exists()
    remaining = {r.ticker for r in audit(data_dir, persist_dir)}
    assert remaining == {"JPM"}


def test_prune_removes_chunks_but_leaves_downloaded_filings_on_disk(store):
    """Raw filings are the only copy of what was downloaded, so prune must keep them."""
    data_dir, persist_dir, index = store
    _write_metadata(data_dir, "GHOST", ["0005-26-006"])
    index([_chunk("JPM", "jpm-1", "0001-26-001", "Bank filing.")])
    # GHOST has downloaded filings and metadata, but nothing ever made it into the store.

    reports = {r.ticker: r for r in audit(data_dir, persist_dir)}
    assert reports["GHOST"].problems == ["nothing indexed"]

    (action,) = prune([reports["GHOST"]], data_dir, persist_dir, dry_run=False)
    assert action["raw_files_left"] == 1
    assert orphan_raw_files("GHOST", data_dir)


def test_prune_deletes_chunks_of_a_broken_company_without_touching_others(store):
    """A company can hold chunks and still be broken - unreadable metadata."""
    data_dir, persist_dir, index = store
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "BAD_filings.json").write_text("{not json", encoding="utf-8")
    _write_metadata(data_dir, "JPM", ["0001-26-001"])
    index(
        [
            _chunk("BAD", "bad-1", "0006-26-007", "Half-written filing."),
            _chunk("JPM", "jpm-1", "0001-26-001", "Bank filing."),
        ]
    )

    (action,) = prune(
        [r for r in audit(data_dir, persist_dir) if r.ticker == "BAD"],
        data_dir,
        persist_dir,
        dry_run=False,
    )

    assert action["chunks_removed"] == 1
    remaining = {r.ticker: r for r in audit(data_dir, persist_dir)}
    assert set(remaining) == {"JPM"}
    assert remaining["JPM"].chunks == 1


def test_audit_of_empty_everything_is_empty(tmp_path):
    pytest.importorskip("chromadb")
    assert audit(tmp_path / "data", tmp_path / "chroma") == []
