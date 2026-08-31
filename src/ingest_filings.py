"""Download recent filings for the single-company SEC due-diligence MVP."""

from __future__ import annotations

import json
from pathlib import Path

from sec_client import FilingRecord, SecClient

# --- Config ---------------------------------------------------------------

USER_AGENT = "Eric Liu EDGAR Copilot Project (replace-with-your-email@example.com)"

TICKERS = ["NVCT"]  # start with one company you know well; add peers once this works

FORMS = ("10-K", "10-Q", "8-K", "DEF 14A")
LIMIT_PER_FORM = 5  # most recent N filings of each type

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

# ---------------------------------------------------------------------------


def ingest_ticker(client: SecClient, ticker: str) -> list[dict]:
    print(f"\n=== {ticker} ===")
    records = client.get_filings(ticker, forms=FORMS, limit_per_form=LIMIT_PER_FORM)
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


def main():
    client = SecClient(user_agent=USER_AGENT)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        metadata = ingest_ticker(client, ticker)

        out_path = METADATA_DIR / f"{ticker}_filings.json"
        out_path.write_text(json.dumps(metadata, indent=2))
        print(f"\nWrote metadata for {len(metadata)} filings -> {out_path}")


if __name__ == "__main__":
    main()
