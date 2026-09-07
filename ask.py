"""Ask one question and see every stage of the answer.

    python ask.py "How much cash does the company have?"
    python ask.py                      # interactive, ask as many as you like
    python ask.py --no-generate "..."  # retrieval only, no API call

Shows what retrieval found, what Claude did with it, which quotes survived
verification, and what the call cost. One question is one API call, so this is
the cheap way to try the pipeline - run_demo.py runs all 13 golden-set cases.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser

from src.grounded_qa import answer_question

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

VECTOR_CANDIDATES = 10
WIDTH = 78

# Published Opus 5 rates, for a rough per-question figure.
COST_PER_INPUT_TOKEN = 5.00 / 1_000_000
COST_PER_OUTPUT_TOKEN = 25.00 / 1_000_000


def wrap(text: str, indent: str = "  ") -> str:
    import textwrap

    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)


def resolve_ticker(requested: str | None) -> str:
    """Pick the company to ask about, or explain the choices."""
    from src.vector_store import indexed_companies

    available = indexed_companies()
    if not available:
        raise SystemExit(
            "The vector store is empty. Load a company first:\n"
            "  python -m src.ingest_filings --ticker NVCT --user-agent \"Name email\"\n"
            "  python -m src.build_index --reset"
        )
    if requested:
        ticker = requested.upper()
        if ticker not in available:
            raise SystemExit(
                f"{ticker} is not indexed. Available: {', '.join(sorted(available))}"
            )
        return ticker
    if len(available) == 1:
        return next(iter(available))
    raise SystemExit(
        f"Several companies are indexed - pass --ticker. Available: {', '.join(sorted(available))}"
    )


def load_everything(use_generator: bool, ticker: str):
    try:
        from src.vector_store import load_all_chunks
    except ImportError:
        raise SystemExit("Install dependencies first:  pip install -r requirements.txt")

    corpus = load_all_chunks(ticker=ticker)
    if not corpus:
        raise SystemExit(f"No indexed chunks for {ticker}.")

    generator = None
    if use_generator:
        from src import backend

        generator = backend.get_generator()
    return corpus, generator


def ask(question: str, corpus: list[dict], generator, ticker: str = "") -> None:
    from src.grounded_qa import retrieve_hybrid
    from src.vector_store import query_vector_store

    vector_hits = query_vector_store(question, limit=VECTOR_CANDIDATES, ticker=ticker or None)

    print()
    print("=" * WIDTH)
    print(f"  {question}")
    print("=" * WIDTH)

    # Stage 1 - what retrieval selected, before anything reads it.
    retrieved = retrieve_hybrid(question, corpus, vector_hits)
    print(f"\nRETRIEVED {len(retrieved)} chunk(s):")
    if not retrieved:
        print("  (nothing cleared the confidence gates)")
    for chunk in retrieved:
        kind = "table" if chunk.get("chunk_type") == "table" else "prose"
        print(f"  - [{kind}] {chunk['section'][:58]} ({chunk['form']} {chunk['filing_date']})")

    # Stage 2 - the answer.
    result = answer_question(question, corpus, vector_hits, generator=generator)

    print()
    print("-" * WIDTH)
    if result["supported"]:
        mode = "WRITTEN BY CLAUDE" if result.get("generated") else "EXCERPTED (no generation)"
        print(f"ANSWER  [{mode}]")
        print()
        print(wrap(result["answer"]))
    else:
        print("ABSTAINED")
        print()
        print(wrap(result["answer"]))
        print()
        print(wrap(f"Why: {result['why_it_matters']}"))

    if result["evidence"]:
        label = "VERIFIED QUOTES" if result.get("generated") else "SUPPORTING EXCERPTS"
        print()
        print("-" * WIDTH)
        print(label)
        for item in result["evidence"]:
            print()
            print(f"  {item['section'][:66]}")
            print(f"  {item['form']} filed {item['filing_date']}")
            print(wrap(f'"{item["excerpt"]}"', indent="    "))
            print(f"    {item['source_url']}")

    # Quotes the model offered that could not be found in the cited chunk. This
    # is the fabrication check doing its job - normally empty.
    rejected = result.get("rejected_citations") or []
    if rejected:
        print()
        print("-" * WIDTH)
        print(f"REJECTED {len(rejected)} QUOTE(S) - not found in the cited evidence")
        for item in rejected:
            print()
            print(f"  {item['chunk_id']}: {item['reason']}")
            print(wrap(f'"{item["quote"][:200]}"', indent="    "))

    usage = result.get("usage") or {}
    if usage:
        cost = (
            usage.get("input_tokens", 0) * COST_PER_INPUT_TOKEN
            + usage.get("output_tokens", 0) * COST_PER_OUTPUT_TOKEN
        )
        print()
        print("-" * WIDTH)
        line = (
            f"  {usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out "
            f"tokens on {usage.get('model', '?')} - about ${cost:.4f}"
        )
        cached = usage.get("cache_read_tokens", 0)
        if cached:
            line += f"  ({cached:,} read from cache)"
        print(line)
    print()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Retrieval only - no API call, no cost.",
    )
    parser.add_argument(
        "--ticker",
        help="Company to ask about. Optional when only one is indexed.",
    )
    args = parser.parse_args()

    ticker = resolve_ticker(args.ticker)
    corpus, generator = load_everything(not args.no_generate, ticker)

    tables = sum(1 for c in corpus if c.get("chunk_type") == "table")
    print(f"Company: {ticker} - {len(corpus)} chunks ({tables} tables) from local EDGAR filings")
    if generator:
        from src import backend

        print(f"Answers: {backend.describe()} (each question is one API call)")
    elif args.no_generate:
        print("Answers: excerpted - generation disabled with --no-generate")
    else:
        print("Answers: excerpted - set ANTHROPIC_API_KEY to have Claude write them")

    if args.question:
        ask(" ".join(args.question), corpus, generator, ticker)
        return

    print("\nType a question, or blank to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        try:
            ask(question, corpus, generator, ticker)
        except Exception as error:  # keep the session alive on a bad call
            print(f"\n  Failed: {error}")


if __name__ == "__main__":
    main()
