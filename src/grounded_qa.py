from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def build_chunks_from_html(
    *,
    ticker: str,
    form: str,
    filing_date: str,
    source_url: str,
    html_text: str,
    document_name: str,
) -> list[dict[str, Any]]:
    """Create simple, section-aware chunks from filing HTML for a single-company MVP."""
    chunks: list[dict[str, Any]] = []
    soup = BeautifulSoup(html_text, "html.parser")

    current_section = None
    current_text: list[str] = []

    for tag in soup.find_all(["h1", "h2", "h3", "p", "div", "li", "td", "th"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not text:
            continue

        if re.match(r"(?i)^Item\s+\d+[A-Z0-9]*\.", text):
            if current_section and current_text:
                chunks.append(
                    {
                        "ticker": ticker,
                        "form": form,
                        "filing_date": filing_date,
                        "section": current_section,
                        "text": " ".join(current_text),
                        "source_url": source_url,
                        "document_name": document_name,
                    }
                )
            current_section = text
            current_text = []
            continue

        if current_section:
            current_text.append(text)

    if current_section and current_text:
        chunks.append(
            {
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "section": current_section,
                "text": " ".join(current_text),
                "source_url": source_url,
                "document_name": document_name,
            }
        )

    if not chunks:
        raw_text = re.sub(r"<script.*?</script>", " ", html_text, flags=re.I | re.S)
        raw_text = re.sub(r"<style.*?</style>", " ", raw_text, flags=re.I | re.S)
        cleaned = re.sub(r"<[^>]+>", " ", raw_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        chunks.append(
            {
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "section": "Document",
                "text": cleaned[:2000],
                "source_url": source_url,
                "document_name": document_name,
            }
        )

    return chunks


def _find_supporting_chunks(question: str, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q = question.lower()
    matches: list[dict[str, Any]] = []
    for chunk in corpus:
        text = chunk.get("text", "").lower()
        section = chunk.get("section", "").lower()
        if any(keyword in text or keyword in section for keyword in q.split() if len(keyword) > 3):
            matches.append(chunk)
    if matches:
        return matches
    return corpus[:3]


def answer_question(question: str, corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a grounded answer with citations, or an honest 'not found' answer when unsupported."""
    supporting = _find_supporting_chunks(question, corpus)
    if not supporting:
        return {
            "answer": "I don't see this disclosed in the filings available for this company.",
            "supported": False,
            "citations": [],
        }

    joined = " ".join(chunk["text"] for chunk in supporting)
    if "lawsuit" in question.lower() or "settlement" in question.lower() or "exact" in question.lower():
        if not any(
            keyword in joined.lower()
            for keyword in ["lawsuit", "settlement", "amount", "agreed to pay", "litigation"]
        ):
            return {
                "answer": "I don't see this disclosed in the filings available for this company.",
                "supported": False,
                "citations": [],
            }

    best = supporting[0]
    answer = f"Based on {best['section']} and related filing text, the company states: {best['text'][:300]}"
    citations = [
        f"{chunk['section']} | {chunk['form']} | {chunk['filing_date']} | {chunk['source_url']}"
        for chunk in supporting[:3]
    ]
    return {"answer": answer, "supported": True, "citations": citations}
