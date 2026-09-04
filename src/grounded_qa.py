from __future__ import annotations

import math
import re
from typing import Any

from bs4 import BeautifulSoup

from .tables import clean_text, drop_table_of_contents, extract_data_tables

# Block-level elements the chunker reads text from. Only *leaf* blocks are read:
# filings nest divs inside divs, so visiting every match would copy the same
# paragraph once per ancestor and inflate a section into a multi-megabyte chunk.
BLOCK_TAGS = ["h1", "h2", "h3", "p", "div", "li", "td", "th"]

# Upper bound on a prose chunk. Oversized chunks make poor evidence - they are
# hard to cite, hard to read in the UI, and they distort length normalization
# during retrieval. Longer sections are split on sentence boundaries.
MAX_CHUNK_CHARS = 3000

# Below this a chunk is leftover padding rather than a disclosure. Kept
# deliberately low: "None." under Item 3 is a real answer about litigation, so
# short chunks are not automatically worthless.
MIN_CHUNK_CHARS = 4

# Sentinel left in the document where a data table was lifted out. The null
# bytes keep it from colliding with anything in real filing text.
TABLE_MARKER = "\x00sec-table:{index}\x00"
_TABLE_MARKER_RE = re.compile(r"^\x00sec-table:(\d+)\x00$")
# A marker also surfaces inside the text of any ancestor element the walker
# visits; strip it there rather than leaking sentinels into chunk text.
_TABLE_MARKER_ANY = re.compile(r"\x00sec-table:\d+\x00")


def _section_slug(section: str) -> str:
    """Compact, filesystem- and ID-safe form of a section heading."""
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
    return slug or "document"


def _split_long_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split oversized prose into windows that end on sentence boundaries."""
    if len(text) <= limit:
        return [text]

    windows: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > limit:
            windows.append(current)
            current = sentence
        elif current:
            current = f"{current} {sentence}"
        else:
            current = sentence
        # A single sentence longer than the limit still has to be broken up.
        while len(current) > limit:
            windows.append(current[:limit])
            current = current[limit:]
    if current:
        windows.append(current)
    return windows


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
    """Create section-aware chunks from filing HTML for a single-company MVP.

    Prose and financial tables become separate chunks. A table keeps its caption
    and one row per line, so a figure stays attached to the label that explains
    it instead of dissolving into a run of numbers.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    drop_table_of_contents(soup)
    extracted_tables = extract_data_tables(soup, TABLE_MARKER)

    # (section, chunk_type, table_title, body) in document order.
    blocks: list[tuple[str, str, str, str]] = []
    current_section = None
    current_text: list[str] = []

    def flush_text() -> None:
        if current_section and current_text:
            body = " ".join(current_text)
            for window in _split_long_text(body):
                blocks.append((current_section, "text", "", window))
        current_text.clear()

    for tag in soup.find_all(BLOCK_TAGS):
        # Skip containers; their block children carry the same text.
        if tag.find(BLOCK_TAGS) is not None:
            continue
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        marker = _TABLE_MARKER_RE.match(text)
        if marker:
            if current_section:
                flush_text()
                table = extracted_tables[int(marker.group(1))]
                blocks.append((current_section, "table", table["title"], table["body"]))
            continue

        text = " ".join(_TABLE_MARKER_ANY.sub(" ", text).split())
        if not text:
            continue

        if re.match(r"(?i)^Item\s+\d+[A-Z0-9]*\.", text):
            flush_text()
            current_section = text
            continue

        if current_section:
            current_text.append(text)

    flush_text()

    if not blocks:
        raw_text = re.sub(r"<script.*?</script>", " ", html_text, flags=re.I | re.S)
        raw_text = re.sub(r"<style.*?</style>", " ", raw_text, flags=re.I | re.S)
        cleaned = re.sub(r"<[^>]+>", " ", raw_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        blocks.append(("Document", "text", "", cleaned[:2000]))

    chunks: list[dict[str, Any]] = []
    seen_sections: dict[str, int] = {}
    for section, chunk_type, title, body in blocks:
        if len(body) < MIN_CHUNK_CHARS:
            continue
        chunk_index = seen_sections.get(section, 0)
        seen_sections[section] = chunk_index + 1
        chunks.append(
            {
                "chunk_id": make_chunk_id(accession_number, section, chunk_index),
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
                "table_title": title,
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "period_of_report": period_of_report or filing_date,
                "accession_number": accession_number,
                "section": section,
                "text": f"{title}\n{body}".strip() if title else body,
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


# A ticker name by itself is not evidence. Requiring a score of at least two
# prevents a company-wide term from pulling in unrelated sections.
MIN_LEXICAL_SCORE = 2

# Fraction of the question's meaningful terms a chunk must account for. The
# count above is not enough on its own: a long filing contains "patents" and
# "address" somewhere, so "How many patents does Apple hold?" clears a bare
# two-term bar. Coverage asks whether the chunk explains most of the question.
MIN_LEXICAL_COVERAGE = 0.6

# Cosine distance above which a vector hit is treated as unrelated. Similarity
# search always returns its nearest neighbours, so without a floor it would
# answer every question and destroy the abstention contract. Measured against
# the golden set: on-topic questions land at 0.45-0.60, off-topic at 0.65+.
MAX_VECTOR_DISTANCE = 0.60

# Reciprocal-rank-fusion damping. The conventional 60 makes fusion depend on
# rank order rather than on two scores that are not on the same scale.
RRF_K = 60


def _term_counts(text: str) -> dict[str, int]:
    """Normalized term frequencies for one document."""
    counts: dict[str, int] = {}
    for raw_term in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        term = raw_term[:-1] if raw_term.endswith("s") and not raw_term.endswith("ss") else raw_term
        if len(term) > 2 and raw_term not in STOP_WORDS and term not in STOP_WORDS:
            counts[term] = counts.get(term, 0) + 1
    return counts


_INDEX_CACHE: dict[tuple, tuple] = {}


def _lexical_index(corpus: list[dict[str, Any]]) -> tuple[list[dict[str, int]], dict[str, int], float]:
    """Term counts per chunk, document frequencies, and mean chunk length.

    Tokenizing the whole corpus on every question is the dominant cost of
    lexical scoring, so the result is memoized against a cheap fingerprint of
    the corpus. Rebuilding the index means restarting the process, which is
    already required after re-ingesting filings.
    """
    fingerprint = (
        len(corpus),
        corpus[0].get("chunk_id", "") if corpus else "",
        corpus[-1].get("chunk_id", "") if corpus else "",
    )
    cached = _INDEX_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    counts = [_term_counts(chunk.get("text", "")) for chunk in corpus]
    document_frequency: dict[str, int] = {}
    for chunk_counts in counts:
        for term in chunk_counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    lengths = [sum(chunk_counts.values()) for chunk_counts in counts]
    average_length = (sum(lengths) / len(lengths)) if lengths else 1.0

    _INDEX_CACHE.clear()  # single-corpus process; do not accumulate
    _INDEX_CACHE[fingerprint] = (counts, document_frequency, average_length or 1.0)
    return _INDEX_CACHE[fingerprint]


# BM25 parameters. k1 damps the benefit of repeating a term; b controls how
# hard long chunks are penalized.
BM25_K1 = 1.5
BM25_B = 0.75

# A section-heading match is strong evidence: "Risk Factors" answering a
# question about risk beats the same words buried in a long prose chunk.
SECTION_MATCH_WEIGHT = 2.0


def score_chunks_lexically(
    question: str, corpus: list[dict[str, Any]]
) -> list[tuple[int, float, dict[str, Any]]]:
    """Rank the corpus by query-term overlap, highest first.

    Three numbers come out of this, doing different jobs. The *gate* counts the
    distinct question terms a chunk matches, weighting a section-heading match
    double. *Coverage* is the fraction of the question those matches account
    for. Together they form the inspectable rule behind abstention. The
    *ranking* is BM25, which weights rare terms above common ones and
    normalizes for length - without that, a whole-Item chunk thousands of words
    long matches almost any question by chance, which is how "Item 4. Controls
    and Procedures" ends up answering a question about cash.

    Returns (gate, coverage, chunk) triples ordered by BM25. No threshold is
    applied here; callers decide what counts as good enough.
    """
    query_terms = _keywords(question)
    if not query_terms or not corpus:
        return []

    counts, document_frequency, average_length = _lexical_index(corpus)
    total_documents = len(corpus)

    ranked: list[tuple[float, int, float, dict[str, Any]]] = []
    for position, chunk in enumerate(corpus):
        # A ticker appears in almost every chunk of a filing. It identifies
        # the company scope, but does not establish that a section answers a
        # question, so it must not affect the relevance score.
        content_query_terms = query_terms - {str(chunk.get("ticker", "")).lower()}
        if not content_query_terms:
            continue
        chunk_counts = counts[position]
        length = sum(chunk_counts.values()) or 1

        matched = content_query_terms & chunk_counts.keys()
        section_terms = _keywords(chunk.get("section", ""))
        section_matches = content_query_terms & section_terms

        gate = len(matched) + 2 * len(section_matches)
        if not gate:
            continue
        coverage = len(matched | section_matches) / len(content_query_terms)

        relevance = 0.0
        for term in matched:
            frequency = chunk_counts[term]
            document_count = document_frequency.get(term, 0)
            idf = math.log(
                1 + (total_documents - document_count + 0.5) / (document_count + 0.5)
            )
            relevance += idf * (frequency * (BM25_K1 + 1)) / (
                frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / average_length)
            )
        relevance += SECTION_MATCH_WEIGHT * len(section_matches)

        ranked.append((relevance, gate, coverage, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(gate, coverage, chunk) for _, gate, coverage, chunk in ranked]


def _passes_lexical_gate(gate: int, coverage: float) -> bool:
    """Enough of the question is accounted for to call this evidence."""
    return gate >= MIN_LEXICAL_SCORE and coverage >= MIN_LEXICAL_COVERAGE


def retrieve_supporting_chunks(question: str, corpus: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Return only chunks with explicit query-term overlap.

    This deliberately abstains instead of falling back to unrelated filing text.
    It stays the lexical half of hybrid retrieval: exact terms, ticker symbols,
    and form/item references are signals embeddings tend to blur.
    """
    scored = score_chunks_lexically(question, corpus)
    return [
        chunk for gate, coverage, chunk in scored[:limit] if _passes_lexical_gate(gate, coverage)
    ]


def _chunk_key(chunk: dict[str, Any]) -> str:
    """Identity for de-duplicating the same chunk arriving from both retrievers."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{chunk.get('source_url', '')}|{chunk.get('section', '')}|{chunk.get('chunk_index', '')}"


def retrieve_hybrid(
    question: str,
    corpus: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]] | tuple = (),
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Fuse lexical ranking with vector-similarity hits.

    Each retriever first applies its own confidence gate - term overlap for
    lexical, a distance floor for vectors - so a chunk has to convince at least
    one of them before it can be cited. Survivors are merged with reciprocal
    rank fusion, which needs only rank order and so avoids comparing an integer
    overlap count against a cosine distance.

    With no vector hits this degrades to the lexical baseline, which keeps the
    abstention contract identical when the vector store is unavailable.
    """
    lexical = [
        chunk
        for gate, coverage, chunk in score_chunks_lexically(question, corpus)
        if _passes_lexical_gate(gate, coverage)
    ]
    vector = [
        hit
        for hit in vector_hits
        if float(hit.get("distance", 1.0)) <= MAX_VECTOR_DISTANCE
    ]

    fused: dict[str, dict[str, Any]] = {}
    ranks: dict[str, float] = {}
    for ranked in (lexical, vector):
        for position, chunk in enumerate(ranked):
            key = _chunk_key(chunk)
            fused.setdefault(key, chunk)
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (RRF_K + position + 1)

    ordered = sorted(fused, key=lambda key: ranks[key], reverse=True)
    return [fused[key] for key in ordered[:limit]]


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


def answer_question(
    question: str,
    corpus: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]] | tuple = (),
) -> dict[str, Any]:
    """Return a grounded answer with citations, or an honest 'not found' answer when unsupported.

    `vector_hits` are similarity results from the Chroma store. Passing none
    falls back to the lexical-only baseline, so the app still answers when
    `chromadb` is not installed or the index has not been built.
    """
    supporting = retrieve_hybrid(question, corpus, vector_hits)
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
