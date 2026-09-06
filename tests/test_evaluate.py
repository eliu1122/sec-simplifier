"""Week 6 tests: the scoring harness and evidence diversity.

The harness is what the project's claims about itself rest on, so its arithmetic
is tested directly rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grounded_qa import MAX_CHUNKS_PER_SECTION, _diversify

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


# --- Evidence diversity ----------------------------------------------------


def _chunk(section: str, form: str = "10-K", text: str = "body"):
    return {"section": section, "form": form, "text": text}


def test_one_section_cannot_fill_every_evidence_slot():
    ranked = [_chunk("Item 1A. Risk Factors") for _ in range(5)] + [_chunk("Item 5. Market")]

    picked = _diversify(ranked, limit=3)

    sections = [c["section"] for c in picked]
    assert sections.count("Item 1A. Risk Factors") == MAX_CHUNKS_PER_SECTION
    assert "Item 5. Market" in sections


def test_diversity_preserves_rank_order():
    ranked = [_chunk("A"), _chunk("B"), _chunk("C")]
    assert [c["section"] for c in _diversify(ranked, limit=3)] == ["A", "B", "C"]


def test_the_cap_relaxes_rather_than_returning_too_few():
    """When one section is all there is, returning two results would be worse."""
    ranked = [_chunk("Item 1A. Risk Factors") for _ in range(4)]

    assert len(_diversify(ranked, limit=3)) == 3


def test_same_section_name_in_different_forms_is_not_conflated():
    # "Item 1A. Risk Factors" appears in both the 10-K and the 10-Q and they are
    # different disclosures, so they should not share a quota.
    ranked = [
        _chunk("Item 1A. Risk Factors", "10-K"),
        _chunk("Item 1A. Risk Factors", "10-K"),
        _chunk("Item 1A. Risk Factors", "10-Q"),
    ]

    picked = _diversify(ranked, limit=3)

    assert len(picked) == 3
    assert [c["form"] for c in picked] == ["10-K", "10-K", "10-Q"]


# --- Scoring arithmetic -----------------------------------------------------


def _row(expected, answered, **kwargs):
    base = {
        "question": "q",
        "expected": expected,
        "answered": answered,
        "correct": answered is expected,
        "known_gap": False,
        "gap_class": "",
        "cited": answered,
        "sections": [],
        "wanted_section": None,
        "section_hit": None,
        "section_known_gap": False,
        "rejected_quotes": 0,
        "generated": False,
        "usage": {},
    }
    base.update(kwargs)
    return base


def test_summarize_separates_answering_from_declining():
    import evaluate

    rows = [
        _row(True, True),    # should answer, did
        _row(True, False),   # should answer, did not
        _row(False, False),  # should decline, did
        _row(False, True),   # should decline, did not
    ]

    stats = evaluate.summarize(rows)

    assert stats["answered_when_expected"] == 1
    assert stats["answered_total"] == 2
    assert stats["declined_when_expected"] == 1
    assert stats["declined_total"] == 2
    assert stats["correct"] == 2
    assert stats["cases"] == 4


def test_summarize_counts_section_hits_only_where_a_section_was_expected():
    import evaluate

    rows = [
        _row(True, True, section_hit=True),
        _row(True, True, section_hit=False),
        _row(True, True, section_hit=None),  # no expectation recorded
    ]

    stats = evaluate.summarize(rows)

    assert stats["section_checked"] == 2
    assert stats["section_hits"] == 1


def test_summarize_tracks_open_versus_recorded_gaps():
    import evaluate

    rows = [
        _row(False, True, known_gap=True),    # still failing
        _row(False, False, known_gap=True),   # now passing - gap closed
        _row(False, False),
    ]

    stats = evaluate.summarize(rows)

    assert stats["gaps_recorded"] == 2
    assert stats["open_gaps"] == 1


# --- The golden set file itself ---------------------------------------------


def _cases():
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))["cases"]


def test_every_case_is_well_formed():
    for case in _cases():
        assert case["expect"] in {"supported", "abstain"}, case["question"]
        assert case.get("why"), f"{case['question']} needs a 'why'"


def test_every_known_gap_explains_itself():
    """A gap without a reason is just a failing test someone silenced."""
    for case in _cases():
        if case.get("known_gap"):
            assert case.get("gap_reason"), f"{case['question']} needs a 'gap_reason'"
            assert case.get("gap_class") in {"subject_mismatch", "fact_not_stated"}, (
                f"{case['question']} needs a recognised gap_class"
            )
        if case.get("section_known_gap"):
            assert case.get("section_gap_reason"), (
                f"{case['question']} needs a 'section_gap_reason'"
            )


def test_the_set_covers_every_form_and_both_expectations():
    cases = _cases()
    expectations = {case["expect"] for case in cases}
    assert expectations == {"supported", "abstain"}

    # The proxy statement was silently reduced to 2 chunks until Week 6; keep
    # cases that would catch that regression.
    assert any(case.get("form_hint") == "DEF 14A" for case in cases)
