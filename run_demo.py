"""Side-by-side demo: what keyword search alone answers, versus hybrid retrieval.

Runs the reviewed cases in tests/golden_set.json through both retrievers and
prints what each one does, so the difference the vector half makes is visible in
one command.

    python run_demo.py            # all cases
    python run_demo.py --contrast # only the cases where the two disagree

Requires a built index:

    python -m src.ingest_filings --ticker NVCT --user-agent "Name email"
    python -m src.build_index --reset
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from src.grounded_qa import answer_question, retrieve_supporting_chunks

# Filing text carries curly quotes and zero-width spaces the default Windows
# console codec cannot encode.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

GOLDEN_SET_PATH = Path(__file__).parent / "tests" / "golden_set.json"
VECTOR_CANDIDATES = 10
WIDTH = 78


def load_corpus() -> tuple[list[dict], str]:
    """Return the indexed corpus, or exit with instructions if it is missing."""
    try:
        from src.vector_store import load_all_chunks, store_stats
    except ImportError:
        raise SystemExit(
            "This demo needs the vector store.\n"
            '  python -m pip install -r requirements.txt'
        )

    corpus = load_all_chunks()
    if not corpus:
        raise SystemExit(
            "The vector store is empty. Build it first:\n"
            "  python -m src.ingest_filings --ticker NVCT --user-agent \"Name email\"\n"
            "  python -m src.build_index --reset"
        )

    stats = store_stats()
    tables = sum(1 for chunk in corpus if chunk.get("chunk_type") == "table")
    label = (
        f"{len(corpus)} chunks ({tables} tables, {len(corpus) - tables} prose) "
        f"from {stats['filings']} {'/'.join(stats['tickers'])} filings"
    )
    return corpus, label


def verdict(supported: bool) -> str:
    return "SUPPORTED" if supported else "ABSTAINS "


def evaluate(case: dict, corpus: list[dict]) -> dict:
    """Answer one case both ways."""
    from src.vector_store import query_vector_store

    question = case["question"]
    expected = case["expect"] == "supported"

    # The lexical baseline is answer_question with no vector hits supplied.
    lexical = answer_question(question, corpus)
    hybrid = answer_question(
        question, corpus, query_vector_store(question, limit=VECTOR_CANDIDATES)
    )
    return {
        "case": case,
        "lexical": lexical,
        "hybrid": hybrid,
        "lexical_ok": lexical["supported"] is expected,
        "hybrid_ok": hybrid["supported"] is expected,
        "disagree": lexical["supported"] != hybrid["supported"],
    }


def report(result: dict) -> None:
    """Print one evaluated case."""
    case, lexical, hybrid = result["case"], result["lexical"], result["hybrid"]
    lexical_ok, hybrid_ok = result["lexical_ok"], result["hybrid_ok"]
    question = case["question"]
    flag = "  <- hybrid differs" if result["disagree"] else ""

    print()
    print(f"  Q  {question}")
    print(f"     expect   {case['expect']}{flag}")
    print(f"     keyword  {verdict(lexical['supported'])} {'OK' if lexical_ok else 'MISS'}")
    print(f"     hybrid   {verdict(hybrid['supported'])} {'OK' if hybrid_ok else 'MISS'}")

    if hybrid["evidence"]:
        top = hybrid["evidence"][0]
        excerpt = " ".join(top["excerpt"].split())[:110]
        print(f"     cites    {top['section']} ({top['form']} {top['filing_date']})")
        print(f'              "{excerpt}..."')

    if case.get("known_gap"):
        print(f"     KNOWN GAP: {case['gap_reason'][:150]}")
    elif not lexical_ok and hybrid_ok and case.get("lexical_alone"):
        print(f"     why      keyword alone {case['lexical_alone']}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="Show only the cases where keyword and hybrid retrieval disagree.",
    )
    args = parser.parse_args()

    corpus, label = load_corpus()
    cases = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))["cases"]

    print("=" * WIDTH)
    print("SEC Simplifier - keyword search vs hybrid retrieval")
    print(f"Corpus: {label}")
    print("=" * WIDTH)

    results = [evaluate(case, corpus) for case in cases]
    shown = [r for r in results if r["disagree"]] if args.contrast else results

    if not shown:
        print("\nNo cases where the two retrievers disagree.")
        return

    for result in shown:
        report(result)

    lexical_score = sum(1 for r in shown if r["lexical_ok"])
    hybrid_score = sum(1 for r in shown if r["hybrid_ok"])
    total = len(shown)

    print()
    print("=" * WIDTH)
    print(f"  keyword only : {lexical_score}/{total} correct")
    print(f"  hybrid       : {hybrid_score}/{total} correct")
    print("=" * WIDTH)

    if hybrid_score < total:
        print(
            "\nRemaining misses are recorded as known gaps in tests/golden_set.json:\n"
            "the retrieved text genuinely matches the question's words while being\n"
            "about a different subject. Resolving that needs a model to read the\n"
            "evidence, which is Week 5."
        )


if __name__ == "__main__":
    main()
