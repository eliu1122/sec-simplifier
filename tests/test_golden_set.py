"""Run the reviewed golden set against the built index.

This is the Week 4 scaffold for the Week 6 evaluation. It asserts only the
supported/abstain contract, and only on cases not marked as known gaps -
scoring citation correctness and support quality comes later.

The whole module skips when the vector store has not been built, so a fresh
checkout still runs green:

    python -m src.ingest_filings --ticker NVCT ...
    python -m src.build_index --reset
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grounded_qa import answer_question

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"

pytest.importorskip("chromadb")


def _golden() -> dict:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def _cases():
    return _golden()["cases"]


TICKER = _golden().get("corpus", {}).get("ticker")


@pytest.fixture(scope="module")
def indexed_corpus():
    """Only this golden set's own company.

    Reading unscoped would score these cases against every company that happens
    to be indexed, which is exactly what ticker scoping exists to prevent.
    """
    from src.vector_store import load_all_chunks

    corpus = load_all_chunks(ticker=TICKER)
    if not corpus:
        pytest.skip(f"{TICKER} is not indexed - run `python -m src.build_index` first")
    return corpus


@pytest.fixture(scope="module")
def generator():
    """The Claude generator when credentials exist, else None (extractive mode).

    With no generator these cases measure retrieval alone - the Week 4 result.
    With one they measure the full pipeline, which is what should finally close
    the two known gaps.
    """
    from src import backend

    return backend.get_generator()


def _answer(question: str, corpus: list[dict], generator=None) -> dict:
    from src.vector_store import query_vector_store

    return answer_question(
        question,
        corpus,
        query_vector_store(question, limit=10, ticker=TICKER),
        generator=generator,
    )


def _is_gap(case: dict, generating: bool) -> bool:
    """Is this case expected to fail in the mode we are running in?

    Some cases retrieval cannot get right but a reading model can. They are
    gaps without a generator and must pass with one - that difference is what
    the generation step is worth.
    """
    if case.get("known_gap"):
        return True
    return bool(case.get("known_gap_without_generation")) and not generating


@pytest.mark.parametrize(
    "case",
    [case for case in _cases() if not case.get("known_gap")],
    ids=lambda case: case["question"][:48],
)
def test_golden_case_matches_expected_support(case, indexed_corpus, generator):
    if _is_gap(case, generator is not None):
        pytest.skip("known to need generation; not configured")
    result = _answer(case["question"], indexed_corpus, generator)
    expected_supported = case["expect"] == "supported"

    assert result["supported"] is expected_supported, (
        f"{case['question']!r} should {case['expect']} - {case['why']}"
    )

    if expected_supported:
        assert result["citations"], "a supported answer must cite filing text"
    else:
        assert result["citations"] == [], "an abstention must not cite anything"


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in _cases()
        if case.get("expect_section_contains") and not case.get("section_known_gap")
    ],
    ids=lambda case: case["question"][:48],
)
def test_golden_case_cites_the_expected_section(case, indexed_corpus, generator):
    result = _answer(case["question"], indexed_corpus, generator)
    sections = [item["section"] for item in result["evidence"]]

    assert any(case["expect_section_contains"] in section for section in sections), (
        f"expected a citation from {case['expect_section_contains']!r}, got {sections}"
    )


def test_known_gaps_are_still_recorded_as_gaps(indexed_corpus, generator):
    """Fail loudly once a known gap starts passing, so the file gets updated.

    A gap that quietly fixes itself is a golden set drifting out of date.
    """
    fixed = [
        case["question"]
        for case in _cases()
        if _is_gap(case, generator is not None)
        and _answer(case["question"], indexed_corpus, generator)["supported"]
        is (case["expect"] == "supported")
    ]

    assert not fixed, (
        "these known gaps now behave correctly - remove their 'known_gap' flag "
        f"in tests/golden_set.json: {fixed}"
    )
