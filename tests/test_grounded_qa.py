import json

import pytest

from src.corpus import load_local_corpus
from src.grounded_qa import (
    answer_question,
    build_chunks_from_html,
    make_chunk_id,
    retrieve_supporting_chunks,
)


def test_build_chunks_from_html_extracts_sections():
    html = """
    <html>
      <body>
        <h1>Item 1. Business</h1>
        <p>We operate a cloud analytics platform.</p>
        <h2>Item 7A. Market Risk</h2>
        <p>We are exposed to interest rate changes.</p>
      </body>
    </html>
    """

    chunks = build_chunks_from_html(
        ticker="NVCT",
        form="10-K",
        filing_date="2024-12-31",
        source_url="https://example.com/filing",
        html_text=html,
        document_name="filing.htm",
    )

    assert len(chunks) >= 2
    assert any("Business" in chunk["section"] for chunk in chunks)
    assert any("platform" in chunk["text"].lower() for chunk in chunks)


def test_answer_question_supports_and_rejects_claims():
    corpus = [
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 1. Business",
            "text": "NVCT operates a cloud analytics platform that enables customers to monitor and analyze operational performance.",
            "source_url": "https://example.com/filing",
            "document_name": "filing.htm",
        },
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 7A. Market Risk",
            "text": "The company is exposed to changes in interest rates and customer concentration risk.",
            "source_url": "https://example.com/filing",
            "document_name": "filing.htm",
        },
    ]

    supported = answer_question("What is NVCT's business?", corpus)
    assert supported["supported"] is True
    assert "Item 1. Business" in supported["citations"][0]
    assert supported["evidence"][0]["excerpt"].startswith("NVCT operates")
    assert "makes money" in supported["why_it_matters"]

    unsupported = answer_question("What is the company's exact lawsuit settlement amount?", corpus)
    assert unsupported["supported"] is False
    assert "I don't see this disclosed" in unsupported["answer"]
    assert unsupported["evidence"] == []


def test_retrieval_returns_only_relevant_filing_sections():
    corpus = [
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 1. Business",
            "text": "NVCT operates a cloud analytics platform.",
            "source_url": "https://example.com/business",
            "document_name": "filing.htm",
        },
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 7A. Market Risk",
            "text": "The company is exposed to interest rate changes.",
            "source_url": "https://example.com/risk",
            "document_name": "filing.htm",
        },
    ]

    matches = retrieve_supporting_chunks("What market risks does NVCT disclose?", corpus)
    assert [match["section"] for match in matches] == ["Item 7A. Market Risk"]


def test_retrieval_normalizes_plural_terms_and_ignores_ticker_matches():
    corpus = [
        {
            "ticker": "NVCT", "form": "10-K", "filing_date": "2024-12-31",
            "section": "Item 5. Market for Common Equity", "text": "NVCT common stock trades on Nasdaq.",
            "source_url": "https://example.com/market", "document_name": "filing.htm",
        },
        {
            "ticker": "NVCT", "form": "10-K", "filing_date": "2024-12-31",
            "section": "Item 7A. Market Risk", "text": "Interest rate risk could affect results.",
            "source_url": "https://example.com/risk", "document_name": "filing.htm",
        },
    ]

    matches = retrieve_supporting_chunks("What market risks does NVCT disclose?", corpus)

    assert matches[0]["section"] == "Item 7A. Market Risk"


def test_answer_abstains_when_no_filing_evidence_matches():
    corpus = [
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 1. Business",
            "text": "NVCT operates a cloud analytics platform.",
            "source_url": "https://example.com/business",
            "document_name": "filing.htm",
        }
    ]

    response = answer_question("Who is the chief executive officer?", corpus)
    assert response["supported"] is False
    assert response["citations"] == []


def test_load_local_corpus_uses_saved_filing_metadata(tmp_path):
    raw_dir = tmp_path / "raw"
    metadata_dir = tmp_path / "metadata"
    raw_dir.mkdir()
    metadata_dir.mkdir()
    filing = raw_dir / "nvct_10k.htm"
    filing.write_text("<h1>Item 1. Business</h1><p>NVCT sells analytics software.</p>")
    (metadata_dir / "NVCT_filings.json").write_text(
        json.dumps(
            [{
                "ticker": "NVCT", "form": "10-K", "filing_date": "2024-12-31",
                "primary_document": "filing.htm", "local_path": str(filing),
                "source_url": "https://www.sec.gov/Archives/example",
            }]
        )
    )

    chunks = load_local_corpus(tmp_path)

    assert chunks[0]["ticker"] == "NVCT"
    assert chunks[0]["section"] == "Item 1. Business"
    assert "analytics software" in chunks[0]["text"]


# --- Week 3: stable chunk IDs and the Chroma vector store -------------------


def _sample_chunks():
    html = (
        "<h1>Item 1. Business</h1><p>NVCT develops targeted oncology therapies.</p>"
        "<h2>Item 1A. Risk Factors</h2><p>The company has a history of operating losses.</p>"
    )
    return build_chunks_from_html(
        ticker="NVCT",
        form="10-K",
        filing_date="2026-02-11",
        source_url="https://www.sec.gov/Archives/example",
        html_text=html,
        document_name="nvct_10k.htm",
        accession_number="0001104659-26-013044",
        period_of_report="2025-12-31",
    )


def test_chunks_carry_stable_id_and_filing_metadata():
    chunks = _sample_chunks()

    assert chunks[0]["chunk_id"] == "0001104659-26-013044:item-1-business:0"
    assert all(c["accession_number"] == "0001104659-26-013044" for c in chunks)
    assert all(c["period_of_report"] == "2025-12-31" for c in chunks)

    # Re-parsing the same filing must produce the same IDs (no duplicates on
    # re-ingestion).
    assert [c["chunk_id"] for c in _sample_chunks()] == [c["chunk_id"] for c in chunks]


def test_make_chunk_id_is_deterministic_and_position_aware():
    section = "Item 7A. Quantitative and Qualitative Disclosures"
    first = make_chunk_id("0000-00-000", section, 0)
    second = make_chunk_id("0000-00-000", section, 1)

    assert first == make_chunk_id("0000-00-000", section, 0)
    assert first != second


def test_vector_store_round_trips_chunks_and_metadata(tmp_path):
    pytest.importorskip("chromadb")
    from src.vector_store import build_vector_store, query_vector_store, store_stats

    persist_dir = tmp_path / "chroma"
    chunks = _sample_chunks()

    count = build_vector_store(chunks, persist_dir=persist_dir)
    assert count == len(chunks)

    # Re-indexing upserts rather than duplicating.
    assert build_vector_store(chunks, persist_dir=persist_dir) == len(chunks)

    hits = query_vector_store("What are the company's risk factors?", persist_dir=persist_dir, limit=1)
    assert hits
    assert hits[0]["section"] == "Item 1A. Risk Factors"
    assert hits[0]["accession_number"] == "0001104659-26-013044"
    assert hits[0]["source_url"] == "https://www.sec.gov/Archives/example"
    assert "distance" in hits[0]

    stats = store_stats(persist_dir=persist_dir)
    assert stats["chunks"] == len(chunks)
    assert stats["tickers"] == ["NVCT"]
    assert stats["filings"] == 1


def test_query_vector_store_empty_when_nothing_indexed(tmp_path):
    pytest.importorskip("chromadb")
    from src.vector_store import query_vector_store

    assert query_vector_store("anything", persist_dir=tmp_path / "chroma") == []
