from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def _section_slug(section: str) -> str:
    """Compact, filesystem- and ID-safe form of a section heading."""
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
    return slug or "document"


def make_chunk_id(accession_number: str, section: str, chunk_index: int) -> str:
    """Stable ID for a chunk.

    Keyed on the filing accession number, the section, and the chunk's position
    within that section, so re-running ingestion upserts the same rows instead
    of creating duplicates.
    """
    accession = accession_number or "no-accession"
    return f"{accession}:{_section_slug(section)}:{chunk_index}"


def build_chunks_from_html(
    *,
    ticker: str,
    form: str,
    filing_date: str,
    source_url: str,
    html_text: str,
    document_name: str,
    accession_number: str = "",
    period_of_report: str = "",
) -> list[dict[str, Any]]:
    """Create simple, section-aware chunks from filing HTML for a single-company MVP."""
    raw_sections: list[tuple[str, str]] = []
    soup = BeautifulSoup(html_text, "html.parser")

    current_section = None
    current_text: list[str] = []

    for tag in soup.find_all(["h1", "h2", "h3", "p", "div", "li", "td", "th"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not text:
            continue

        if re.match(r"(?i)^Item\s+\d+[A-Z0-9]*\.", text):
            if current_section and current_text:
                raw_sections.append((current_section, " ".join(current_text)))
            current_section = text
            current_text = []
            continue

        if current_section:
            current_text.append(text)

    if current_section and current_text:
        raw_sections.append((current_section, " ".join(current_text)))

    if not raw_sections:
        raw_text = re.sub(r"<script.*?</script>", " ", html_text, flags=re.I | re.S)
        raw_text = re.sub(r"<style.*?</style>", " ", raw_text, flags=re.I | re.S)
        cleaned = re.sub(r"<[^>]+>", " ", raw_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        raw_sections.append(("Document", cleaned[:2000]))

    chunks: list[dict[str, Any]] = []
    seen_sections: dict[str, int] = {}
    for section, body in raw_sections:
        chunk_index = seen_sections.get(section, 0)
        seen_sections[section] = chunk_index + 1
        chunks.append(
            {
                "chunk_id": make_chunk_id(accession_number, section, chunk_index),
                "chunk_index": chunk_index,
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "period_of_report": period_of_report or filing_date,
                "accession_number": accession_number,
                "section": section,
                "text": body,
                "source_url": source_url,
                "document_name": document_name,
            }
        )

    return chunks


STOP_WORDS = {
    "about", "after", "amount", "and", "are", "company", "does", "exact", "filings",
    "from", "have", "into", "is", "its", "of", "or", "please", "show", "tell", "that",
    "the", "their", "there", "this", "what", "when", "where", "which", "with", "would",
}


def _keywords(text: str) -> set[str]:
    """Return meaningful normalized terms for transparent lexical retrieval."""
    terms: set[str] = set()
    for raw_term in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        term = raw_term[:-1] if raw_term.endswith("s") and not raw_term.endswith("ss") else raw_term
        if len(term) > 2 and raw_term not in STOP_WORDS and term not in STOP_WORDS:
            terms.add(term)
    return terms


def retrieve_supporting_chunks(question: str, corpus: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Return only chunks with explicit query-term overlap.

    This deliberately abstains instead of falling back to unrelated filing text.
    It is a small, inspectable baseline that can later be replaced by semantic
    retrieval without changing the grounded-answer contract.
    """
    query_terms = _keywords(question)
    if not query_terms:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for chunk in corpus:
        # A ticker appears in almost every chunk of a filing. It identifies
        # the company scope, but does not establish that a section answers a
        # question, so it must not affect the relevance score.
        content_query_terms = query_terms - {str(chunk.get("ticker", "")).lower()}
        body_terms = _keywords(chunk.get("text", ""))
        section_terms = _keywords(chunk.get("section", ""))
        matched_terms = content_query_terms & body_terms
        score = len(matched_terms) + 2 * len(content_query_terms & section_terms)
        if score:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    # A ticker name by itself is not evidence. Requiring a score of at least
    # two prevents a company-wide term from pulling in unrelated sections.
    return [chunk for score, chunk in scored[:limit] if score >= 2]


def _why_it_matters(section: str, question: str) -> str:
    """Give first-time investors a plain-English reason to inspect this evidence."""
    text = f"{section} {question}".lower()
    if "risk" in text:
        return "Risks can affect future results, cash needs, and how predictable the business is."
    if "business" in text or "product" in text or "customer" in text:
        return "Understanding how the company makes money is the starting point for judging its prospects."
    if "financial" in text or "revenue" in text or "cash" in text:
        return "This helps you evaluate the company’s financial performance and ability to fund operations."
    return "Read the cited filing section to verify the statement in the company’s own words."


def answer_question(question: str, corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a grounded answer with citations, or an honest 'not found' answer when unsupported."""
    supporting = retrieve_supporting_chunks(question, corpus)
    if not supporting:
        return {
            "answer": "I don't see this disclosed in the filings available for this company.",
            "supported": False,
            "citations": [],
            "evidence": [],
            "why_it_matters": "Try a broader question or add more filings before drawing a conclusion.",
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
                "evidence": [],
                "why_it_matters": "The available filing text does not support a precise answer to this question.",
            }

    best = supporting[0]
    answer = f"Based on {best['section']} and related filing text, the company states: {best['text'][:300]}"
    citations = [
        f"{chunk['section']} | {chunk['form']} | {chunk['filing_date']} | {chunk['source_url']}"
        for chunk in supporting[:3]
    ]
    evidence = [
        {
            "section": chunk["section"],
            "form": chunk["form"],
            "filing_date": chunk["filing_date"],
            "excerpt": chunk["text"][:500],
            "source_url": chunk["source_url"],
        }
        for chunk in supporting[:3]
    ]
    return {
        "answer": answer,
        "supported": True,
        "citations": citations,
        "evidence": evidence,
        "why_it_matters": _why_it_matters(best["section"], question),
    }
