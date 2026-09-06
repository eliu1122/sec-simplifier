"""Score the pipeline against the reviewed golden set.

    python evaluate.py                  # score, compare to the baseline
    python evaluate.py --no-generate    # retrieval only, no API cost
    python evaluate.py --save-baseline  # record this run as the new baseline

Reports three things, because they fail independently:

  Answer rate    - does it answer what it should, and decline what it should?
  Citations      - when it answers, does it cite, and from the right section?
  Grounding      - are the cited quotes actually present in the filing?

Baselines are stored in tests/eval_baseline.json and compared on every run, so a
change that helps one metric and quietly costs another is visible.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from src.grounded_qa import answer_question

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

GOLDEN_SET_PATH = Path(__file__).parent / "tests" / "golden_set.json"
BASELINE_PATH = Path(__file__).parent / "tests" / "eval_baseline.json"
VECTOR_CANDIDATES = 10
WIDTH = 74


def load_corpus_and_generator(use_generator: bool, ticker: str | None = None):
    try:
        from src.vector_store import load_all_chunks
    except ImportError:
        raise SystemExit("Install dependencies:  pip install -r requirements.txt")

    # The golden set is written against one company's filings, so scoring reads
    # only that company - otherwise another indexed company's text would compete
    # for evidence the cases were never reviewed against.
    corpus = load_all_chunks(ticker=ticker)
    if not corpus:
        raise SystemExit(
            f"No indexed chunks for {ticker or 'any company'}. Build the index first:\n"
            "  python -m src.build_index --reset"
        )

    generator = None
    if use_generator:
        from src import generate

        if generate.is_available():
            generator = lambda q, e: generate.generate_grounded_answer(q, e)  # noqa: E731
    return corpus, generator


def score_case(case: dict, corpus: list[dict], generator, ticker: str | None = None) -> dict:
    """Run one case and record what happened, without judging it yet."""
    from src.vector_store import query_vector_store

    question = case["question"]
    result = answer_question(
        question,
        corpus,
        query_vector_store(question, limit=VECTOR_CANDIDATES, ticker=ticker),
        generator=generator,
    )

    expected = case["expect"] == "supported"
    sections = [item["section"] for item in result["evidence"]]
    wanted_section = case.get("expect_section_contains")

    return {
        "question": question,
        "expected": expected,
        "answered": result["supported"],
        "correct": result["supported"] is expected,
        "known_gap": bool(case.get("known_gap")),
        "gap_class": case.get("gap_class", ""),
        "cited": bool(result["citations"]),
        "sections": sections,
        "wanted_section": wanted_section,
        "section_hit": (
            any(wanted_section in section for section in sections)
            if wanted_section and result["supported"]
            else None
        ),
        "section_known_gap": bool(case.get("section_known_gap")),
        # Only the generator produces verifiable quotes; the extractive path
        # returns a slice of the chunk, which is grounded by construction.
        "rejected_quotes": len(result.get("rejected_citations") or []),
        "generated": bool(result.get("generated")),
        "usage": result.get("usage") or {},
    }


def percent(numerator: int, denominator: int) -> str:
    if not denominator:
        return "   n/a"
    return f"{numerator / denominator * 100:5.0f}%"


def summarize(rows: list[dict]) -> dict:
    should_answer = [r for r in rows if r["expected"]]
    should_decline = [r for r in rows if not r["expected"]]
    answered = [r for r in rows if r["answered"]]
    with_wanted_section = [r for r in rows if r["section_hit"] is not None]

    return {
        "cases": len(rows),
        "answered_when_expected": sum(1 for r in should_answer if r["correct"]),
        "answered_total": len(should_answer),
        "declined_when_expected": sum(1 for r in should_decline if r["correct"]),
        "declined_total": len(should_decline),
        "correct": sum(1 for r in rows if r["correct"]),
        "cited_when_answered": sum(1 for r in answered if r["cited"]),
        "answers": len(answered),
        "section_hits": sum(1 for r in with_wanted_section if r["section_hit"]),
        "section_checked": len(with_wanted_section),
        "rejected_quotes": sum(r["rejected_quotes"] for r in rows),
        "open_gaps": sum(1 for r in rows if r["known_gap"] and not r["correct"]),
        "gaps_recorded": sum(1 for r in rows if r["known_gap"]),
    }


def report(rows: list[dict], stats: dict, baseline: dict | None, mode: str) -> None:
    print()
    print("=" * WIDTH)
    print("  SEC Simplifier - golden set evaluation")
    print(f"  Mode: {mode}")
    print("=" * WIDTH)

    print("\nANSWER RATE")
    print(f"  answered when it should    {stats['answered_when_expected']:>3}/{stats['answered_total']:<3}  "
          f"{percent(stats['answered_when_expected'], stats['answered_total'])}")
    print(f"  declined when it should    {stats['declined_when_expected']:>3}/{stats['declined_total']:<3}  "
          f"{percent(stats['declined_when_expected'], stats['declined_total'])}")
    print(f"  overall                    {stats['correct']:>3}/{stats['cases']:<3}  "
          f"{percent(stats['correct'], stats['cases'])}")

    print("\nCITATIONS")
    print(f"  answers that cite anything {stats['cited_when_answered']:>3}/{stats['answers']:<3}  "
          f"{percent(stats['cited_when_answered'], stats['answers'])}")
    print(f"  cited the expected section {stats['section_hits']:>3}/{stats['section_checked']:<3}  "
          f"{percent(stats['section_hits'], stats['section_checked'])}")
    section_gaps = [r for r in rows if r["section_known_gap"] and r["section_hit"] is False]
    for row in section_gaps:
        print(f"    known ranking gap: {row['question'][:48]}")

    print("\nGROUNDING")
    if any(r["generated"] for r in rows):
        print(f"  quotes rejected as not found in the cited evidence: {stats['rejected_quotes']}")
        print("  (each one is a fabricated quote that never reached the user)")
    else:
        print("  n/a - extractive mode quotes the chunk directly, so nothing to verify")

    failures = [r for r in rows if not r["correct"]]
    if failures:
        print(f"\nFAILURES ({len(failures)})")
        for row in failures:
            tag = f"[{row['gap_class']}]" if row["gap_class"] else "[NEW]"
            want = "answer" if row["expected"] else "decline"
            print(f"  {tag:<20} should {want}: {row['question'][:44]}")

    unexpected = [r for r in failures if not r["known_gap"]]
    if unexpected:
        print(f"\n  {len(unexpected)} failure(s) are NOT recorded as known gaps - investigate.")

    fixed = [r for r in rows if r["known_gap"] and r["correct"]]
    if fixed:
        print(f"\nCLOSED GAPS ({len(fixed)}) - clear their 'known_gap' flag in golden_set.json")
        for row in fixed:
            print(f"  {row['question'][:60]}")

    if baseline:
        print(f"\nVS BASELINE ({baseline.get('recorded')}, {baseline.get('mode')})")
        previous = baseline.get("stats", {})
        for label, key, total_key in [
            ("overall", "correct", "cases"),
            ("declined when it should", "declined_when_expected", "declined_total"),
            ("cited expected section", "section_hits", "section_checked"),
        ]:
            was, now = previous.get(key), stats[key]
            if was is None:
                continue
            delta = now - was
            arrow = "same" if delta == 0 else (f"+{delta}" if delta > 0 else str(delta))
            print(f"  {label:<26} {was:>3} -> {now:<3}  {arrow}")
    else:
        print("\nNo baseline recorded yet. Run with --save-baseline to set one.")

    spent = sum(r["usage"].get("input_tokens", 0) for r in rows)
    produced = sum(r["usage"].get("output_tokens", 0) for r in rows)
    if spent:
        cost = spent * 5.00 / 1_000_000 + produced * 25.00 / 1_000_000
        print(f"\n  {spent:,} in / {produced:,} out tokens - about ${cost:.2f}")
    print()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--no-generate", action="store_true",
                        help="Retrieval only - no API calls, no cost.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Record this run as the baseline future runs compare against.")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    cases = golden["cases"]
    ticker = golden.get("corpus", {}).get("ticker")
    corpus, generator = load_corpus_and_generator(not args.no_generate, ticker)

    if generator:
        from src.generate import DEFAULT_MODEL

        mode = f"{ticker} - hybrid retrieval + {DEFAULT_MODEL}"
    else:
        mode = f"{ticker} - hybrid retrieval, extractive answers (no generation)"

    rows = [score_case(case, corpus, generator, ticker) for case in cases]
    stats = summarize(rows)

    baseline = None
    if BASELINE_PATH.is_file():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    report(rows, stats, baseline, mode)

    if args.save_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {"recorded": date.today().isoformat(), "mode": mode, "stats": stats},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Baseline written to {BASELINE_PATH.name}\n")


if __name__ == "__main__":
    main()
