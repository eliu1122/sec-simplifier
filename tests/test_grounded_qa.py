from src.grounded_qa import answer_question, build_chunks_from_html


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

    unsupported = answer_question("What is the company's exact lawsuit settlement amount?", corpus)
    assert unsupported["supported"] is False
    assert "I don't see this disclosed" in unsupported["answer"]
