"""Download recent filings for the single-company SEC due-diligence MVP."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

from .sec_client import SecClient

# --- Config ---------------------------------------------------------------

FORMS = ("10-K", "10-Q", "8-K", "DEF 14A")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

# ---------------------------------------------------------------------------


def ingest_ticker(client: SecClient, ticker: str, limit_per_form: int) -> list[dict]:
    print(f"\n=== {ticker} ===")
    records = client.get_filings(ticker, forms=FORMS, limit_per_form=limit_per_form)
    print(f"Found {len(records)} filings across forms {FORMS}")

    saved_metadata = []
    for record in records:
        print(f"  Downloading {record.form} filed {record.filing_date} ({record.accession_number})...")
        try:
            local_path = client.download_filing(record, RAW_DIR)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        meta = record.to_dict()
        meta["local_path"] = str(local_path)
        saved_metadata.append(meta)
        print(f"    Saved -> {local_path}")

    return saved_metadata


def parse_args() -> object:
    parser = ArgumentParser(description="Download recent EDGAR filings for one ticker.")
    parser.add_argument("--ticker", default="NVCT", help="Company ticker to ingest (default: NVCT).")
    parser.add_argument("--limit-per-form", type=int, default=2, help="Recent filings to download per form.")
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help="Your name and contact email, or set the SEC_USER_AGENT environment variable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.user_agent:
        raise SystemExit(
            "Provide --user-agent 'Your Name your-email@example.com' or set SEC_USER_AGENT."
        )
    if args.limit_per_form < 1:
        raise SystemExit("--limit-per-form must be at least 1.")

    client = SecClient(user_agent=args.user_agent)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    ticker = args.ticker.upper()
    metadata = ingest_ticker(client, ticker, args.limit_per_form)
    out_path = METADATA_DIR / f"{ticker}_filings.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote metadata for {len(metadata)} filings -> {out_path}")


if __name__ == "__main__":
    main()
