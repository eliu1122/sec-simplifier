import json

import pytest

from src.corpus import load_local_corpus
from src.grounded_qa import (
    MAX_CHUNK_CHARS,
    answer_question,
    build_chunks_from_html,
    make_chunk_id,
    retrieve_hybrid,
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


# --- Week 4: table-aware chunking and hybrid retrieval ----------------------


FINANCIAL_TABLE_HTML = """
<html><body>
  <p>Item 1. Financial Statements</p>
  <table>
    <tr><td>Item 1.</td><td>Business</td><td>7</td></tr>
    <tr><td>Item 1A.</td><td>Risk Factors</td><td>19</td></tr>
    <tr><td>Item 7.</td><td>Management's Discussion</td><td>48</td></tr>
  </table>
  <p>CONDENSED BALANCE SHEETS (USD in thousands)</p>
  <table>
    <tr><td></td><td>June 30,</td><td>December 31,</td></tr>
    <tr><td></td><td>2026</td><td>2025</td></tr>
    <tr><td>Cash and cash equivalents</td><td>$</td><td>22,160</td><td>$</td><td>31,634</td></tr>
    <tr><td>Total assets</td><td>$</td><td>22,376</td><td>$</td><td>31,709</td></tr>
  </table>
</body></html>
"""


def _financial_chunks():
    return build_chunks_from_html(
        ticker="NVCT",
        form="10-Q",
        filing_date="2026-08-04",
        source_url="https://www.sec.gov/Archives/example",
        html_text=FINANCIAL_TABLE_HTML,
        document_name="nvct_10q.htm",
        accession_number="0001104659-26-090109",
        period_of_report="2026-06-30",
    )


def test_financial_table_becomes_its_own_labelled_chunk():
    tables = [c for c in _financial_chunks() if c["chunk_type"] == "table"]

    assert len(tables) == 1, "the table of contents must not be indexed as data"
    table = tables[0]
    assert "BALANCE SHEETS" in table["table_title"]
    # The figure stays on the same line as the label that explains it, rather
    # than dissolving into a run of numbers.
    assert "Cash and cash equivalents | 22,160 | 31,634" in table["text"]
    assert table["section"] == "Item 1. Financial Statements"
    assert table["accession_number"] == "0001104659-26-090109"


def test_table_of_contents_does_not_create_sections():
    # "Item 1A." and "Item 7." appear only in the dropped TOC, so the filing
    # must yield exactly the one real section.
    assert {c["section"] for c in _financial_chunks()} == {"Item 1. Financial Statements"}


def test_nested_markup_is_not_counted_twice():
    html = (
        "<h1>Item 1. Business</h1>"
        "<div><div><p>We develop targeted oncology therapies.</p></div></div>"
    )
    chunks = build_chunks_from_html(
        ticker="NVCT", form="10-K", filing_date="2026-02-11",
        source_url="https://example.com/f", html_text=html,
        document_name="f.htm", accession_number="acc-1",
    )

    assert chunks[0]["text"].count("targeted oncology therapies") == 1


def test_long_sections_are_split_into_bounded_chunks():
    sentence = "The company reported material progress in its clinical programs. "
    html = f"<h1>Item 1. Business</h1><p>{sentence * 200}</p>"
    chunks = build_chunks_from_html(
        ticker="NVCT", form="10-K", filing_date="2026-02-11",
        source_url="https://example.com/f", html_text=html,
        document_name="f.htm", accession_number="acc-1",
    )

    assert len(chunks) > 1
    assert all(len(c["text"]) <= MAX_CHUNK_CHARS for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def _two_chunk_corpus():
    return [
        {
            "chunk_id": "acc:business:0", "ticker": "NVCT", "form": "10-K",
            "filing_date": "2026-02-11", "section": "Item 1. Business",
            "text": "NVCT operates a cloud analytics platform for enterprise customers.",
            "source_url": "https://example.com/business", "document_name": "f.htm",
        },
        {
            "chunk_id": "acc:liquidity:0", "ticker": "NVCT", "form": "10-K",
            "filing_date": "2026-02-11", "section": "Item 7. Liquidity",
            "text": "We expect our existing funds to finance operations into 2027.",
            "source_url": "https://example.com/liquidity", "document_name": "f.htm",
        },
    ]


def test_hybrid_recovers_a_paraphrase_the_lexical_scorer_misses():
    corpus = _two_chunk_corpus()
    question = "What is the company's cash runway?"

    # No meaningful term overlap: the filing says "funds ... finance", not "cash
    # runway", so the lexical scorer alone abstains.
    assert retrieve_supporting_chunks(question, corpus) == []

    vector_hits = [dict(corpus[1], distance=0.31)]
    hybrid = retrieve_hybrid(question, corpus, vector_hits)

    assert [c["section"] for c in hybrid] == ["Item 7. Liquidity"]


def test_hybrid_still_abstains_when_vector_hits_are_far_away():
    corpus = _two_chunk_corpus()
    distant = [dict(corpus[0], distance=0.94)]

    assert retrieve_hybrid("What is the price of Bitcoin?", corpus, distant) == []


def test_hybrid_without_vector_hits_matches_the_lexical_baseline():
    corpus = _two_chunk_corpus()
    question = "What analytics platform does NVCT operate?"

    assert retrieve_hybrid(question, corpus) == retrieve_supporting_chunks(question, corpus)


def test_hybrid_deduplicates_a_chunk_found_by_both_retrievers():
    corpus = _two_chunk_corpus()
    vector_hits = [dict(corpus[0], distance=0.2)]

    hybrid = retrieve_hybrid("Describe the analytics platform", corpus, vector_hits)

    assert len(hybrid) == 1
    assert hybrid[0]["chunk_id"] == "acc:business:0"


def test_answer_question_uses_vector_hits_and_keeps_the_abstention_contract():
    corpus = _two_chunk_corpus()

    supported = answer_question(
        "What is the company's cash runway?",
        corpus,
        [dict(corpus[1], distance=0.29)],
    )
    assert supported["supported"] is True
    assert "Item 7. Liquidity" in supported["citations"][0]

    unsupported = answer_question(
        "What is the price of Bitcoin?", corpus, [dict(corpus[0], distance=0.97)]
    )
    assert unsupported["supported"] is False
    assert unsupported["citations"] == []
