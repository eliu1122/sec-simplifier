"""Score the pipeline against the reviewed golden set.

    python evaluate.py                  # score, compare to the baseline
    python evaluate.py --no-generate    # retrieval only, no API cost
    python evaluate.py --save-baseline  # record this run as the new baseline

Reports three things, because they fail independently:

  Answer rate    - does it answer what it should, and decline what it should?
  Citations      - when it answers, does it cite, and from the right section?
  Correctness    - does the answer actually state the fact that was asked for?
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
# Candidates pulled before fusion. Wide enough that a chunk missing from
# this list really is far away: the lexical veto treats an absent chunk as
# distance 1.0, so a short window vetoed good lexical matches for being
# outside the window rather than for being unrelated.
VECTOR_CANDIDATES = 60
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
        from src import backend

        generator = backend.get_generator()
    return corpus, generator


def answer_states_the_fact(case: dict, result: dict, home: bool) -> bool | None:
    """Does the answer actually contain the fact the case says it should?

    None when the question is not checkable this way - no fact recorded, the
    system declined, or we are scoring a company the facts were not written
    for. The facts are NVCT's, so they are only asserted against NVCT.

    This is the check whose absence let a confident wrong answer outscore an
    honest refusal: without it, scoring asked only whether the system answered
    and whether the section matched.
    """
    wanted = case.get("expect_answer_contains")
    if not wanted or not home or not result["supported"]:
        return None
    answer = (result.get("answer") or "").lower()
    return any(str(fact).lower() in answer for fact in wanted)


def score_case(
    case: dict, corpus: list[dict], generator, ticker: str | None = None, home: bool = True
) -> dict:
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
        "fact_hit": answer_states_the_fact(case, result, home),
        "generation_error": result.get("generation_error", ""),
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
        "fact_hits": sum(1 for r in rows if r.get("fact_hit")),
        "fact_checked": sum(1 for r in rows if r.get("fact_hit") is not None),
        "reached_model": sum(1 for r in rows if r["generated"]),
        "rejected_quotes": sum(r["rejected_quotes"] for r in rows),
        "open_gaps": sum(1 for r in rows if r["known_gap"] and not r["correct"]),
        "gaps_recorded": sum(1 for r in rows if r["known_gap"]),
    }


def report(rows: list[dict], stats: dict, baseline: dict | None, mode: str) -> None:
    print()
    print("=" * WIDTH)
    print("  Footnote - golden set evaluation")
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

    if stats["fact_checked"]:
        print("\nCORRECTNESS")
        print(f"  answer states the expected fact {stats['fact_hits']:>3}/{stats['fact_checked']:<3}  "
              f"{percent(stats['fact_hits'], stats['fact_checked'])}")
        for row in rows:
            if row.get("fact_hit") is False:
                print(f"    answered without the fact: {row['question'][:44]}")

    # A run where some calls never reached the model is not a measurement of
    # it. Free tiers cap requests per day and the service returns 503 under
    # load; the fallback is deliberately quiet in the UI, so the harness has
    # to say so itself rather than publishing a blend of two modes as one.
    attempted = stats["reached_model"] or any(r.get("generation_error") for r in rows)
    if attempted and stats["reached_model"] < stats["cases"]:
        missed = stats["cases"] - stats["reached_model"]
        print("")
        print(f"  PARTIAL RUN: generation reached {stats['reached_model']}/{stats['cases']} cases;")
        print(f"  {missed} fell back to extractive answers. These figures mix both modes")
        print("  and measure neither. Re-run before trusting them.")
        reasons = sorted({r['generation_error'] for r in rows
                          if not r['generated'] and r.get('generation_error')})
        for reason in reasons:
            print(f"    cause: {reason}")

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
            ("answer states the fact", "fact_hits", "fact_checked"),
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
        from src import backend

        # Priced by whichever backend actually ran. Hardcoding one
        # backend's rates reported a charge for calls that were free.
        model = next((r["usage"].get("model", "") for r in rows if r["usage"]), "")
        print("")
        print("  " + backend.format_cost(
            {"input_tokens": spent, "output_tokens": produced, "model": model}))
    print()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--no-generate", action="store_true",
                        help="Retrieval only - no API calls, no cost.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Record this run as the baseline future runs compare against.")
    parser.add_argument(
        "--only-checkable",
        action="store_true",
        help=(
            "Score only the cases with a recorded fact. Ten calls rather than "
            "thirty, which matters on a free tier capped at 20 per day."
        ),
    )
    parser.add_argument(
        "--ticker",
        help=(
            "Score a different company using only the portable cases - those whose "
            "expectation holds for any filer. Reveals whether thresholds generalize."
        ),
    )
    args = parser.parse_args()

    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    home_ticker = golden.get("corpus", {}).get("ticker")
    ticker = (args.ticker or home_ticker or "").upper()

    if args.ticker and ticker != home_ticker:
        # Company-specific expectations (this filer's drug, its section names)
        # would be wrong against another company.
        cases = [case for case in golden["cases"] if case.get("portable")]
        if not cases:
            raise SystemExit("No cases are marked portable.")
    else:
        cases = golden["cases"]

    corpus, generator = load_corpus_and_generator(not args.no_generate, ticker)

    if generator:
        from src import backend

        name, module = backend.active_backend()
        mode = f"{ticker} - hybrid retrieval + {name} ({module.DEFAULT_MODEL})"
    else:
        mode = f"{ticker} - hybrid retrieval, extractive answers (no generation)"

    if args.only_checkable:
        cases = [case for case in cases if case.get("expect_answer_contains")]
        if not cases:
            raise SystemExit("No cases carry expect_answer_contains.")

    home = ticker == home_ticker
    rows = [score_case(case, corpus, generator, ticker, home) for case in cases]
    stats = summarize(rows)

    baseline = None
    if BASELINE_PATH.is_file() and ticker == home_ticker and len(cases) == len(golden["cases"]):
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    report(rows, stats, baseline, mode)

    if args.save_baseline and (args.only_checkable or (args.ticker and ticker != home_ticker)):
        raise SystemExit(
            "Baselines are only recorded for a full run against the golden set's own company."
        )

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
