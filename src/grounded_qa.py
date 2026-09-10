from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
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


# A section heading is never a table cell. Layout tables are full of all-caps
# row labels ("ALL CURRENT DIRECTORS AND NEOS AS A GROUP") that read exactly
# like headings and would otherwise carve a filing into fragments.
NON_HEADING_TAGS = {"td", "th"}


def _is_item_heading(text: str, tag_name: str = "") -> bool:
    """`Item 1A.` style headings, used by 10-K, 10-Q and 8-K."""
    return bool(re.match(r"(?i)^Item\s+\d+[A-Z0-9]*\.", text))


# Running headers and cover-page boilerplate that look like headings but are
# page furniture repeated throughout the document.
CAPS_HEADING_NOISE = {
    "table of contents",
    "united states securities and exchange commission",
    "schedule 14a information",
    "your vote is important",
    "your vote is important!",
}

MAX_HEADING_CHARS = 90
MIN_UPPERCASE_RATIO = 0.75


def _is_caps_heading(text: str, tag_name: str = "") -> bool:
    """An ALL-CAPS heading, used by proxy statements.

    A DEF 14A contains no `Item N.` headings at all - it is organised under
    headings like EXECUTIVE COMPENSATION and REPORT OF THE AUDIT COMMITTEE. This
    is only consulted for documents where no Item heading was found, so it
    cannot fragment a 10-K on an all-caps table caption.
    """
    if tag_name in NON_HEADING_TAGS:
        return False
    if len(text) > MAX_HEADING_CHARS or text.strip(" .:").lower() in CAPS_HEADING_NOISE:
        return False
    if len(text.split()) < 2:
        return False
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 6:
        return False
    uppercase = sum(1 for character in letters if character.isupper())
    return uppercase / len(letters) >= MIN_UPPERCASE_RATIO


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

    def walk(is_heading) -> list[tuple[str, str, str, str]]:
        """Collect (section, chunk_type, table_title, body) in document order."""
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

            if is_heading(text, tag.name):
                flush_text()
                current_section = text
                continue

            if current_section:
                current_text.append(text)

        flush_text()
        return blocks

    # 10-K, 10-Q and 8-K are organised by `Item N.` headings. A proxy statement
    # has none, so fall back to its ALL-CAPS headings rather than discarding the
    # document - the DEF 14A is where compensation and governance live.
    blocks = walk(_is_item_heading)
    if not blocks:
        blocks = walk(_is_caps_heading)

    if not blocks:
        raw_text = re.sub(r"<script.*?</script>", " ", html_text, flags=re.I | re.S)
        raw_text = re.sub(r"<style.*?</style>", " ", raw_text, flags=re.I | re.S)
        cleaned = re.sub(r"<[^>]+>", " ", raw_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Window it rather than truncating: a document with no recognizable
        # headings is still evidence.
        blocks = [("Document", "text", "", window) for window in _split_long_text(cleaned)]

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


# Interrogatives and pronouns are the question's scaffolding, not its subject.
# They matter because coverage divides by the number of terms: "How much cash
# does the company have on hand?" scored 'cash' against five terms, four of
# which no filing ever contains, so a balance sheet reached 0.25 coverage and
# was gated out. Stripping them leaves 'cash', which the same table matches
# outright. "hand" is here for the same reason - "on hand" is idiom, not a
# disclosure term.
STOP_WORDS = {
    "about", "after", "amount", "and", "any", "are", "can", "company", "does",
    "exact", "filings", "from", "hand", "have", "how", "into", "is", "its",
    "many", "much", "of", "or", "our", "please", "show", "tell", "that", "the",
    "their", "them", "there", "they", "this", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "your",
}


# People ask in everyday words; filings answer in accounting terms. Mapping the
# question onto the filing's vocabulary has to REPLACE the term rather than add
# to it: coverage is a fraction of the question's terms, so adding 'cash'
# alongside 'money' would demand a chunk contain both and raise the bar it was
# meant to lower. Only applied to questions - chunk text keeps its own words.
QUERY_SYNONYMS = {
    "money": "cash",
    "funds": "cash",
    "fund": "cash",
    "earn": "revenue",
    "earning": "revenue",
    "sale": "revenue",
    "salary": "compensation",
    "pay": "compensation",
    "paid": "compensation",
    "boss": "officer",
    "staff": "employee",
    "worker": "employee",
    "lawsuit": "litigation",
    "sued": "litigation",
}


def _keywords(text: str, expand: bool = False) -> set[str]:
    """Return meaningful normalized terms for transparent lexical retrieval.

    `expand` maps everyday words onto filing vocabulary and is set for questions
    only, so that "how much money do they have?" reaches a balance sheet that
    says "cash and cash equivalents" and never says "money".
    """
    terms: set[str] = set()
    for raw_term in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        term = raw_term[:-1] if raw_term.endswith("s") and not raw_term.endswith("ss") else raw_term
        if len(term) > 2 and raw_term not in STOP_WORDS and term not in STOP_WORDS:
            terms.add(QUERY_SYNONYMS.get(term, term) if expand else term)
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


# A chunk the lexical scorer likes but that vector search never surfaced is
# usually a vocabulary coincidence rather than evidence. "What is the weather
# forecast for New Jersey?" clears the lexical gate on every company tested,
# because "forecast" is financial vocabulary and the state name sits in the
# address block - while the embedding correctly places the same text at 0.78+.
#
# Because the two gates are OR'd, the weaker one sets the floor. Requiring the
# embedding not to actively disagree raised specificity from 35% to 48% across
# six companies with no measured recall cost.
MAX_LEXICAL_VETO_DISTANCE = 0.80

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


_INDEX_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()

# Corpora to keep tokenized at once. One per company in play: a user comparing a
# handful of companies should not re-tokenize on every question, but the cache
# must not grow without bound either.
INDEX_CACHE_SIZE = 8


def _lexical_index(corpus: list[dict[str, Any]]) -> tuple[list[dict[str, int]], dict[str, int], float]:
    """Term counts per chunk, document frequencies, and mean chunk length.

    Tokenizing the whole corpus on every question is the dominant cost of
    lexical scoring, so the result is memoized against a cheap fingerprint of
    the corpus. Several companies can be in play at once, so this is an LRU
    rather than a single slot - switching back to a company you asked about a
    moment ago should not re-tokenize its filings.

    Note the document frequencies are per-corpus, which is what makes BM25
    meaningful here: a term's rarity is judged within one company's filings, not
    across every company that happens to be indexed.
    """
    fingerprint = (
        len(corpus),
        corpus[0].get("chunk_id", "") if corpus else "",
        corpus[-1].get("chunk_id", "") if corpus else "",
    )
    cached = _INDEX_CACHE.get(fingerprint)
    if cached is not None:
        _INDEX_CACHE.move_to_end(fingerprint)
        return cached

    counts = [_term_counts(chunk.get("text", "")) for chunk in corpus]
    document_frequency: dict[str, int] = {}
    for chunk_counts in counts:
        for term in chunk_counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    lengths = [sum(chunk_counts.values()) for chunk_counts in counts]
    average_length = (sum(lengths) / len(lengths)) if lengths else 1.0

    _INDEX_CACHE[fingerprint] = (counts, document_frequency, average_length or 1.0)
    while len(_INDEX_CACHE) > INDEX_CACHE_SIZE:
        _INDEX_CACHE.popitem(last=False)
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
    query_terms = _keywords(question, expand=True)
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


def _passes_lexical_gate(gate: int, coverage: float, query_terms: int = 0) -> bool:
    """Enough of the question is accounted for to call this evidence.

    The two-term floor stops a bare ticker match from counting as evidence, but
    it must never exceed what the question can supply: "How much cash does the
    company have on hand?" reduces to the single term 'cash', and a floor of two
    made every balance sheet unreachable no matter how well it matched. When a
    question has only one meaningful term, matching it is full coverage, and
    coverage is what carries the "explains most of the question" guarantee.
    """
    floor = min(MIN_LEXICAL_SCORE, query_terms) if query_terms else MIN_LEXICAL_SCORE
    return gate >= floor and coverage >= MIN_LEXICAL_COVERAGE


def retrieve_supporting_chunks(question: str, corpus: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Return only chunks with explicit query-term overlap.

    This deliberately abstains instead of falling back to unrelated filing text.
    It stays the lexical half of hybrid retrieval: exact terms, ticker symbols,
    and form/item references are signals embeddings tend to blur.
    """
    scored = score_chunks_lexically(question, corpus)
    terms = len(_keywords(question, expand=True))
    return [
        chunk
        for gate, coverage, chunk in scored[:limit]
        if _passes_lexical_gate(gate, coverage, terms)
    ]


def _chunk_key(chunk: dict[str, Any]) -> str:
    """Identity for de-duplicating the same chunk arriving from both retrievers."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{chunk.get('source_url', '')}|{chunk.get('section', '')}|{chunk.get('chunk_index', '')}"


def _matches_identifier(query_terms: set[str], chunk: dict[str, Any]) -> bool:
    """Does the question name this chunk's section or form outright?

    "What does Item 1A disclose?" and "What was reported on Form 8-K?" name a
    location in the filing rather than describing a topic, which is the one
    thing embeddings are reliably bad at and the reason the lexical half is
    kept. Such a match is an identifier, not a coincidence, so it is exempt from
    the distance veto above - without this, those questions silently break.
    """
    identifiers = _keywords(chunk.get("section", "")) | _keywords(chunk.get("form", ""))
    return bool(query_terms & identifiers)


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
    query_terms = _keywords(question, expand=True)
    distance_by_key = {
        _chunk_key(hit): float(hit.get("distance", 1.0)) for hit in vector_hits
    }
    # The veto says "the embedding disagrees", which requires the embedding to
    # have been consulted. With no vector hits at all - no chromadb, or no index
    # built - there is no disagreement to act on, and applying it anyway would
    # suppress every lexical match and make the app abstain on everything.
    veto_available = bool(vector_hits)

    lexical: list[dict[str, Any]] = []
    for gate, coverage, chunk in score_chunks_lexically(question, corpus):
        if not _passes_lexical_gate(gate, coverage, len(query_terms)):
            continue
        if not veto_available:
            lexical.append(chunk)
            continue
        # Absent from the vector results entirely counts as "placed far away".
        distance = distance_by_key.get(_chunk_key(chunk), 1.0)
        if distance <= MAX_LEXICAL_VETO_DISTANCE or _matches_identifier(query_terms, chunk):
            lexical.append(chunk)

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
    return _diversify([fused[key] for key in ordered], limit)


# Evidence slots are few, so spending them all on one section is wasteful: three
# passages from the same section usually make one point, and crowd out the
# section that actually answers the question. Filings make this worse than most
# corpora, because a mislabelled heading can attach one section name to a long
# run of chunks.
MAX_CHUNKS_PER_SECTION = 2


def _diversify(ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take the top `limit`, but prefer not to fill them from a single section.

    Order is otherwise preserved, and the cap is relaxed rather than returning
    fewer results when there is nothing else to draw on.
    """
    selected: list[dict[str, Any]] = []
    held_back: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for chunk in ranked:
        if len(selected) == limit:
            break
        section = f"{chunk.get('form', '')}|{chunk.get('section', '')}"
        if seen.get(section, 0) >= MAX_CHUNKS_PER_SECTION:
            held_back.append(chunk)
            continue
        seen[section] = seen.get(section, 0) + 1
        selected.append(chunk)

    # Nothing else qualified - fall back to the ranking as it stood.
    for chunk in held_back:
        if len(selected) == limit:
            break
        selected.append(chunk)

    return selected


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


UNSUPPORTED_ANSWER = "I don't see this disclosed in the filings available for this company."


class GenerationUnavailable(Exception):
    """The generator could not be reached. Carries a reason worth showing."""


def describe_generation_failure(error: Exception) -> str:
    """Turn an API error into something a reader can act on.

    The distinction that matters is between "you have run out for today" and
    "something is broken" - the first is expected on a free tier and resolves on
    its own, the second needs attention.
    """
    text = str(error)
    if "PerDay" in text or "per day" in text.lower():
        return "daily free-tier quota reached"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "rate limited"
    if "503" in text or "UNAVAILABLE" in text:
        return "the model is temporarily unavailable"
    if "401" in text or "403" in text or "API key" in text or "PERMISSION" in text:
        return "the API key was rejected"
    if "404" in text or "NOT_FOUND" in text:
        return "the configured model was not found"
    return "the generation service could not be reached"


def _unsupported(why: str) -> dict[str, Any]:
    return {
        "answer": UNSUPPORTED_ANSWER,
        "supported": False,
        "citations": [],
        "evidence": [],
        "why_it_matters": why,
        # Retrieval declined before any model was consulted. Overridden by the
        # caller when a generator made the decision instead.
        "generated": False,
        "generation_error": "",
    }


def _generated_answer(
    question: str,
    supporting: list[dict[str, Any]],
    generator: Any,
) -> dict[str, Any] | None:
    """Let the generator read the evidence and decide. None means fall back.

    A generator that fails is not allowed to take the app down or, worse, to
    silently produce an ungrounded answer - the caller drops back to the
    extractive response, which is still cited.

    The failure is logged rather than swallowed. Falling back silently once hid
    a broken API client through an entire evaluation run, where every answer
    looked merely disappointing instead of never having reached the model.
    """
    try:
        result = generator(question, supporting)
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Generation failed, falling back to the extractive answer: %s: %s",
            type(error).__name__,
            error,
        )
        # The caller attaches this to the extractive answer, so the UI can say
        # which mode actually produced what the reader is looking at.
        raise GenerationUnavailable(describe_generation_failure(error)) from error

    if not result.get("supported"):
        abstained = _unsupported(
            result.get("reason")
            or "The filing text retrieved for this question does not answer it."
        )
        abstained["generated"] = True
        if result.get("usage"):
            abstained["usage"] = result["usage"]
        return abstained

    citations = result.get("citations", [])
    return {
        "answer": result["answer"],
        "supported": True,
        # Quotes here are verified against the filing text, so the citation is
        # the exact sentence backing the claim rather than a section pointer.
        "citations": [
            f"{c['section']} | {c['form']} | {c['filing_date']} | {c['source_url']}"
            for c in citations
        ],
        "evidence": [
            {
                "section": c["section"],
                "form": c["form"],
                "filing_date": c["filing_date"],
                "excerpt": c["quote"],
                "source_url": c["source_url"],
            }
            for c in citations
        ],
        "why_it_matters": _why_it_matters(citations[0]["section"], question),
        "generated": True,
        "usage": result.get("usage", {}),
        # Quotes the model offered that did not survive verification. Empty is
        # the normal case; anything here is worth looking at.
        "rejected_citations": result.get("rejected_citations", []),
    }


def answer_question(
    question: str,
    corpus: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]] | tuple = (),
    generator: Any = None,
) -> dict[str, Any]:
    """Return a grounded answer with citations, or an honest 'not found' answer when unsupported.

    `vector_hits` are similarity results from the Chroma store. Passing none
    falls back to the lexical-only baseline, so the app still answers when
    `chromadb` is not installed or the index has not been built.

    `generator` is called as `generator(question, evidence)` to write the answer
    from the retrieved evidence - see `src.generate`. Without one, the answer is
    the extractive Week 4 response: the top chunk, excerpted. Retrieval and the
    abstention contract are identical either way; the generator can only narrow
    what gets answered, never widen it, because it never sees the filing beyond
    the chunks retrieval already approved.
    """
    supporting = retrieve_hybrid(question, corpus, vector_hits)
    if not supporting:
        return _unsupported(
            "Try a broader question or add more filings before drawing a conclusion."
        )

    # Why generation did not produce this answer, when a generator was offered.
    # Carried on the response so the UI reports the mode that actually ran
    # rather than the mode that was configured.
    generation_error = ""
    if generator is not None:
        try:
            return _generated_answer(question, supporting, generator)
        except GenerationUnavailable as unavailable:
            generation_error = str(unavailable)

    best = supporting[0]
    # Without a generator there is nothing to write a direct answer with, so the
    # honest framing is that this is a passage to read rather than an answer.
    # The old wording ("Based on ... the company states:") presented 300
    # characters of whichever chunk ranked first as though it answered the
    # question, which it did 40% of the time.
    answer = (
        f"No written answer - showing the closest passage from {best['section']}. "
        f"{best['text'][:300]}"
    )
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
        "generated": False,
        "generation_error": generation_error,
    }
