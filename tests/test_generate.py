"""Week 5 tests: quote verification and the generator's effect on the contract.

None of these call the API. The generator is injected, so the parts most likely
to be wrong - whether a fabricated quote is caught, and whether the abstention
contract survives - are testable offline.
"""

from __future__ import annotations

import pytest

from src.generate import (
    Citation,
    build_prompt,
    format_evidence,
    verify_citations,
)
from src.grounded_qa import UNSUPPORTED_ANSWER, answer_question

EVIDENCE = [
    {
        "chunk_id": "acc:liquidity:0",
        "ticker": "NVCT",
        "form": "10-K",
        "filing_date": "2026-02-11",
        "period_of_report": "2025-12-31",
        "section": "Item 7. Liquidity",
        "text": (
            "We expect our existing cash and cash equivalents to fund operations "
            "into the second quarter of 2027."
        ),
        "source_url": "https://example.com/liquidity",
    },
    {
        "chunk_id": "acc:business:0",
        "ticker": "NVCT",
        "form": "10-K",
        "filing_date": "2026-02-11",
        "period_of_report": "2025-12-31",
        "section": "Item 1. Business",
        "text": "The company develops targeted oncology therapies.",
        "source_url": "https://example.com/business",
    },
]


# --- Quote verification ----------------------------------------------------


def test_verified_citation_is_enriched_with_filing_metadata():
    verified, rejected = verify_citations(
        [Citation(chunk_id="acc:liquidity:0", quote="fund operations into the second quarter of 2027")],
        EVIDENCE,
    )

    assert rejected == []
    assert len(verified) == 1
    # The model supplies only an id and a quote; everything a reader needs to
    # check the source is looked up locally.
    assert verified[0]["section"] == "Item 7. Liquidity"
    assert verified[0]["form"] == "10-K"
    assert verified[0]["source_url"] == "https://example.com/liquidity"


def test_fabricated_quote_is_rejected():
    verified, rejected = verify_citations(
        [Citation(chunk_id="acc:liquidity:0", quote="cash runway extends into 2029")],
        EVIDENCE,
    )

    assert verified == []
    assert rejected[0]["reason"] == "quote does not appear in the cited evidence"


def test_quote_attributed_to_the_wrong_chunk_is_rejected():
    # The text is real, but it is not in the chunk the model named.
    verified, rejected = verify_citations(
        [Citation(chunk_id="acc:business:0", quote="fund operations into the second quarter")],
        EVIDENCE,
    )

    assert verified == []
    assert rejected


def test_citation_of_an_unknown_chunk_id_is_rejected():
    verified, rejected = verify_citations(
        [Citation(chunk_id="acc:nonexistent:9", quote="anything at all")], EVIDENCE
    )

    assert verified == []
    assert rejected[0]["reason"] == "cited an evidence id that was not provided"


@pytest.mark.parametrize(
    "quote",
    [
        "We expect our existing cash and cash equivalents",  # exact
        "we expect our existing cash",  # case differs
        "existing   cash  and cash equivalents",  # whitespace differs
    ],
)
def test_quotes_match_through_cosmetic_differences(quote):
    verified, _ = verify_citations([Citation(chunk_id="acc:liquidity:0", quote=quote)], EVIDENCE)
    assert len(verified) == 1


def test_typographic_quotes_do_not_break_matching():
    evidence = [dict(EVIDENCE[0], text="The company’s board approved the plan—unanimously.")]
    verified, _ = verify_citations(
        [Citation(chunk_id="acc:liquidity:0", quote="The company's board approved the plan-unanimously.")],
        evidence,
    )
    assert len(verified) == 1


# --- Prompt construction ---------------------------------------------------


def test_evidence_blocks_carry_the_ids_the_model_must_cite():
    rendered = format_evidence(EVIDENCE)

    assert 'id="acc:liquidity:0"' in rendered
    assert 'section="Item 7. Liquidity"' in rendered
    assert "fund operations into the second quarter of 2027" in rendered


def test_prompt_contains_the_question_and_the_evidence():
    prompt = build_prompt("What is the cash runway?", EVIDENCE)

    assert "What is the cash runway?" in prompt
    assert "acc:liquidity:0" in prompt


# --- The generator's effect on answer_question ------------------------------


def _corpus():
    return [dict(chunk) for chunk in EVIDENCE]


def _vector_hits():
    """Close similarity hits, so retrieval passes and the generator is consulted."""
    return [dict(EVIDENCE[0], distance=0.28), dict(EVIDENCE[1], distance=0.44)]


def _generator(result):
    """A stand-in for src.generate.generate_grounded_answer."""
    return lambda question, evidence: result


def test_generated_answer_replaces_the_extractive_one():
    verified = {
        "chunk_id": "acc:liquidity:0",
        "quote": "fund operations into the second quarter of 2027",
        "section": "Item 7. Liquidity",
        "form": "10-K",
        "filing_date": "2026-02-11",
        "period_of_report": "2025-12-31",
        "source_url": "https://example.com/liquidity",
    }
    generator = _generator(
        {
            "supported": True,
            "answer": "The company reports it expects its cash to last into Q2 2027.",
            "citations": [verified],
            "rejected_citations": [],
            "reason": "",
        }
    )

    result = answer_question("What is the cash runway?", _corpus(), _vector_hits(), generator=generator)

    assert result["supported"] is True
    assert result["answer"].startswith("The company reports")
    assert result["generated"] is True
    # The excerpt is now the verified quote, not the first 500 characters.
    assert result["evidence"][0]["excerpt"] == verified["quote"]
    assert "Item 7. Liquidity" in result["citations"][0]


def test_generator_can_abstain_on_evidence_retrieval_approved():
    """The point of Week 5: a reader can reject evidence retrieval let through."""
    generator = _generator(
        {
            "supported": False,
            "answer": "",
            "citations": [],
            "rejected_citations": [],
            "reason": "The evidence describes this company's patents, not Apple's.",
        }
    )

    result = answer_question("How many patents does Apple hold?", _corpus(), _vector_hits(), generator=generator)

    assert result["supported"] is False
    assert result["answer"] == UNSUPPORTED_ANSWER
    assert result["citations"] == []
    assert "not Apple" in result["why_it_matters"]


def test_generator_cannot_answer_when_retrieval_found_nothing():
    """The generator narrows; it never widens. It is not consulted at all here."""
    calls = []

    def generator(question, evidence):
        calls.append(question)
        return {"supported": True, "answer": "Bitcoin trades at $60,000.",
                "citations": [], "rejected_citations": [], "reason": ""}

    result = answer_question("What is the price of Bitcoin?", _corpus(), generator=generator)

    assert calls == [], "the generator must not see questions retrieval already rejected"
    assert result["supported"] is False


# --- Reporting which mode actually produced an answer -----------------------
#
# The status line used to report what was *configured* rather than what ran, so
# it kept claiming answers were model-written after the daily quota was gone.


@pytest.mark.parametrize(
    "message,expected",
    [
        ("429 RESOURCE_EXHAUSTED ... GenerateRequestsPerDayPerProjectPerModel-FreeTier", "daily free-tier quota reached"),
        ("429 RESOURCE_EXHAUSTED quota per minute", "rate limited"),
        ("503 UNAVAILABLE model is overloaded", "the model is temporarily unavailable"),
        ("401 Unauthorized: invalid API key", "the API key was rejected"),
        ("404 NOT_FOUND: model not available", "the configured model was not found"),
        ("something else entirely", "the generation service could not be reached"),
    ],
)
def test_failures_are_described_in_terms_a_reader_can_act_on(message, expected):
    from src.grounded_qa import describe_generation_failure

    assert describe_generation_failure(RuntimeError(message)) == expected


def test_a_generated_answer_says_so():
    corpus = _corpus()
    verified = {
        "chunk_id": "acc:liquidity:0", "quote": "fund operations into the second quarter of 2027",
        "section": "Item 7. Liquidity", "form": "10-K", "filing_date": "2026-02-11",
        "period_of_report": "2025-12-31", "source_url": "https://example.com/liquidity",
    }
    generator = _generator({
        "supported": True, "answer": "The company expects cash into Q2 2027.",
        "citations": [verified], "rejected_citations": [], "reason": "",
    })

    result = answer_question("What is the cash runway?", corpus, _vector_hits(), generator=generator)

    assert result["generated"] is True


def test_quota_exhaustion_is_reported_rather_than_hidden():
    """The whole point: an excerpt must not be presented as a written answer."""
    def exhausted(question, evidence):
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )

    result = answer_question("What is the cash runway?", _corpus(), _vector_hits(), generator=exhausted)

    assert result["supported"] is True          # still answered, extractively
    assert result["generated"] is False         # but not by a model
    assert result["generation_error"] == "daily free-tier quota reached"


def test_no_generator_reports_no_error():
    """Extractive by choice is not a failure, and must not read like one."""
    result = answer_question("What is the cash runway?", _corpus(), _vector_hits())

    assert result["generated"] is False
    assert result["generation_error"] == ""


def test_retrieval_abstention_is_not_attributed_to_a_model():
    result = answer_question("What is the price of Bitcoin?", _corpus(), generator=_generator({}))

    assert result["supported"] is False
    assert result["generated"] is False
    assert result["generation_error"] == ""


def test_generator_failure_falls_back_to_the_extractive_answer():
    def exploding(question, evidence):
        raise RuntimeError("API unavailable")

    result = answer_question("What is the cash runway?", _corpus(), _vector_hits(), generator=exploding)

    # Degraded, but still a cited answer rather than an error page.
    assert result["supported"] is True
    assert result.get("generated") is not True
    assert result["citations"]


# --- generate_grounded_answer, against a stub client ------------------------
#
# The API itself is never called. These pin the request shape and the handling
# of what comes back, which is otherwise only exercisable with credentials.


class _StubUsage:
    input_tokens = 2500
    output_tokens = 300
    cache_read_input_tokens = 700
    cache_creation_input_tokens = 0


class _StubResponse:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = _StubUsage()


class _StubClient:
    """Records the request and returns a canned response."""

    def __init__(self, parsed, stop_reason="end_turn"):
        self._parsed = parsed
        self._stop_reason = stop_reason
        self.request = None
        self.messages = self

    def parse(self, **kwargs):
        self.request = kwargs
        return _StubResponse(self._parsed, self._stop_reason)


def _answer(**kwargs):
    from src.generate import GroundedAnswer

    return GroundedAnswer(**kwargs)


def test_request_is_shaped_for_structured_output():
    from src.generate import DEFAULT_MODEL, GroundedAnswer, generate_grounded_answer

    client = _StubClient(_answer(supported=False, answer="", citations=[], reason_if_unsupported="no"))
    generate_grounded_answer("What is the cash runway?", EVIDENCE, client=client)

    assert client.request["model"] == DEFAULT_MODEL
    assert client.request["output_format"] is GroundedAnswer
    assert client.request["thinking"] == {"type": "adaptive"}
    assert "acc:liquidity:0" in client.request["messages"][0]["content"]

    # The system prompt is identical every request, so it carries a cache
    # breakpoint; the per-question evidence goes after it in `messages`.
    system = client.request["system"]
    assert "SEC filings" in system[0]["text"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_usage_is_reported_back():
    from src.generate import generate_grounded_answer

    client = _StubClient(_answer(supported=False, answer="", citations=[], reason_if_unsupported="no"))
    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["usage"]["cache_read_tokens"] == 700


def test_supported_answer_returns_only_verified_citations():
    from src.generate import generate_grounded_answer

    client = _StubClient(
        _answer(
            supported=True,
            answer="The company reports cash into Q2 2027.",
            citations=[
                Citation(chunk_id="acc:liquidity:0", quote="fund operations into the second quarter of 2027"),
                Citation(chunk_id="acc:liquidity:0", quote="a sentence never written in the filing"),
            ],
            reason_if_unsupported="",
        )
    )

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is True
    assert len(result["citations"]) == 1
    assert len(result["rejected_citations"]) == 1
    # Usage is reported back so callers can show what a question cost.
    assert result["usage"]["input_tokens"] == 2500


def test_answer_with_every_quote_fabricated_is_downgraded():
    """A confident answer that cannot be traced to the filing is withheld."""
    from src.generate import generate_grounded_answer

    client = _StubClient(
        _answer(
            supported=True,
            answer="The company earned $50 million in revenue.",
            citations=[Citation(chunk_id="acc:liquidity:0", quote="revenue of $50 million")],
            reason_if_unsupported="",
        )
    )

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is False
    assert result["answer"] == ""
    assert result["citations"] == []
    assert "could not be traced" in result["reason"]


def test_refusal_becomes_an_abstention():
    from src.generate import generate_grounded_answer

    client = _StubClient(
        _answer(supported=True, answer="x", citations=[], reason_if_unsupported=""),
        stop_reason="refusal",
    )

    result = generate_grounded_answer("q", EVIDENCE, client=client)

    assert result["supported"] is False
    assert result["citations"] == []


def test_answer_question_without_a_generator_is_unchanged():
    with_none = answer_question("What is the cash runway?", _corpus(), _vector_hits())
    assert with_none["supported"] is True
    assert with_none.get("generated") is not True
