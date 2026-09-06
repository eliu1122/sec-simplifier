"""Thin SEC EDGAR client for the single-company due-diligence MVP."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_BASE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"

# Be a polite citizen of a free public API: SEC asks for a real identifying UA.
# Replace with your own name + contact email before running.
DEFAULT_USER_AGENT = "Eric Liu EDGAR Copilot Project (replace-with-your-email@example.com)"

REQUEST_DELAY_SECONDS = 0.15  # keeps us comfortably under SEC's rate guidance


@dataclass
class FilingRecord:
    ticker: str
    cik: str
    form: str
    filing_date: str
    accession_number: str
    primary_document: str
    source_url: str
    period_of_report: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "form": self.form,
            "filing_date": self.filing_date,
            "period_of_report": self.period_of_report,
            "accession_number": self.accession_number,
            "primary_document": self.primary_document,
            "source_url": self.source_url,
        }


class SecClient:
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        if "replace-with-your-email" in user_agent:
            raise ValueError(
                "Set a real User-Agent (your name + contact email) before hitting "
                "SEC EDGAR. Pass user_agent= to SecClient(...) or edit DEFAULT_USER_AGENT."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._ticker_map_cache: dict | None = None

    def _get(self, url: str) -> requests.Response:
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return resp

    def _ticker_entry(self, ticker: str) -> dict:
        """Look one ticker up in EDGAR's ticker/CIK/name mapping."""
        if self._ticker_map_cache is None:
            resp = self._get(TICKER_MAP_URL)
            self._ticker_map_cache = resp.json()  # {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}

        ticker_upper = ticker.upper()
        for entry in self._ticker_map_cache.values():
            if entry["ticker"].upper() == ticker_upper:
                return entry

        raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")

    def get_cik_for_ticker(self, ticker: str) -> str:
        """Return a 10-digit, zero-padded CIK string for a given ticker."""
        return str(self._ticker_entry(ticker)["cik_str"]).zfill(10)

    def company_name_for_ticker(self, ticker: str) -> str:
        """Registrant name as EDGAR spells it, for labelling the UI."""
        return self._ticker_entry(ticker).get("title") or ticker.upper()

    def get_filings(
        self,
        ticker: str,
        forms: Iterable[str] = ("10-K", "10-Q", "8-K", "DEF 14A"),
        limit_per_form: int = 5,
    ) -> list[FilingRecord]:
        """Fetch recent filings of the given forms for a ticker."""
        cik10 = self.get_cik_for_ticker(ticker)
        cik_int = int(cik10)

        resp = self._get(SUBMISSIONS_URL.format(cik10=cik10))
        data = resp.json()

        recent = data["filings"]["recent"]
        forms_wanted = set(forms)
        counts: dict[str, int] = {f: 0 for f in forms_wanted}
        records: list[FilingRecord] = []

        for i, form in enumerate(recent["form"]):
            if form not in forms_wanted:
                continue
            if counts[form] >= limit_per_form:
                continue

            accession_raw = recent["accessionNumber"][i]  # e.g. "0001193125-24-012345"
            accession_nodash = accession_raw.replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]
            # Period the filing reports on (fiscal year / quarter end). EDGAR
            # leaves this blank for some 8-Ks; fall back to the filing date so
            # every chunk still has a usable period for year-over-year work.
            period_of_report = (recent.get("reportDate") or [])[i:i + 1]
            period_of_report = period_of_report[0] if period_of_report else ""
            if not period_of_report:
                period_of_report = filing_date

            source_url = ARCHIVE_BASE_URL.format(
                cik_int=cik_int, accession_nodash=accession_nodash, filename=primary_doc
            )

            records.append(
                FilingRecord(
                    ticker=ticker.upper(),
                    cik=cik10,
                    form=form,
                    filing_date=filing_date,
                    accession_number=accession_raw,
                    primary_document=primary_doc,
                    source_url=source_url,
                    period_of_report=period_of_report,
                )
            )
            counts[form] += 1

        return records

    def download_filing(self, record: FilingRecord, raw_dir: Path) -> Path:
        """Download a filing's primary document to raw_dir and return the saved path."""
        raw_dir.mkdir(parents=True, exist_ok=True)
        resp = self._get(record.source_url)

        suffix = Path(record.primary_document).suffix or ".html"
        out_name = f"{record.ticker}_{record.form.replace(' ', '')}_{record.filing_date}_{record.accession_number}{suffix}"
        out_path = raw_dir / out_name
        out_path.write_bytes(resp.content)
        return out_path
