"""Check that every indexed company is actually usable, and prune the ones that are not.

The picker in the app lists whatever the vector store holds, so a half-ingested
company shows up looking normal and then answers nothing. Ingestion can leave
that state behind in a few ways: a ticker that resolves in EDGAR but files none
of the four forms writes metadata and no chunks, and a run interrupted partway
indexes some filings and not others.

    "C:\\Users\\yceri\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m src.audit_index
    "C:\\Users\\yceri\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m src.audit_index --prune
    "C:\\Users\\yceri\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m src.audit_index --prune --yes

`--prune` prints what it would remove and changes nothing. Only `--prune --yes`
deletes, and it only ever touches companies that are unusable: no chunks in the
store, or no filings on record. Partial indexes and missing source files are
reported as warnings and never deleted, because both are recoverable by
re-running ingestion and neither is worth risking an index that is not in git.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

# Filing text carries typographic characters (curly quotes, zero-width spaces)
# that the default Windows console codec cannot encode.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

from .vector_store import DEFAULT_PERSIST_DIR, delete_company, get_collection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class TickerReport:
    """What one company looks like across the store, its metadata, and disk."""

    ticker: str
    chunks: int = 0
    indexed_filings: int = 0
    metadata_records: int = 0
    missing_raw: int = 0
    unindexed_filings: int = 0
    # Disqualifying: the company cannot answer anything. `--prune --yes` removes these.
    problems: list[str] = field(default_factory=list)
    # Worth knowing, but recoverable by re-ingesting. Never pruned.
    warnings: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.problems


def _chroma_by_ticker(persist_dir: Path | str) -> dict[str, dict]:
    """Chunk counts and accession numbers per ticker, straight from the store."""
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return {}

    found: dict[str, dict] = {}
    for metadata in collection.get(include=["metadatas"])["metadatas"]:
        ticker = metadata.get("ticker")
        if not ticker:
            continue
        entry = found.setdefault(ticker, {"chunks": 0, "accessions": set()})
        entry["chunks"] += 1
        if metadata.get("accession_number"):
            entry["accessions"].add(metadata["accession_number"])
    return found


def _metadata_by_ticker(data_dir: Path) -> dict[str, dict]:
    """Filing records per ticker, plus which of their source files are gone."""
    metadata_dir = Path(data_dir) / "metadata"
    if not metadata_dir.is_dir():
        return {}

    found: dict[str, dict] = {}
    for path in sorted(metadata_dir.glob("*_filings.json")):
        ticker = path.name.split("_")[0]
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            records = []
            found.setdefault(ticker, {})["unreadable"] = True
        entry = found.setdefault(ticker, {})
        entry["path"] = path
        entry["records"] = records
        entry["accessions"] = {
            r.get("accession_number") for r in records if r.get("accession_number")
        }
        entry["missing_raw"] = [
            r for r in records if not Path(r.get("local_path", "")).exists()
        ]
    return found


def audit(
    data_dir: Path | str = DATA_DIR, persist_dir: Path | str = DEFAULT_PERSIST_DIR
) -> list[TickerReport]:
    """Cross-check every known ticker against the store, its metadata, and disk."""
    chroma = _chroma_by_ticker(persist_dir)
    metadata = _metadata_by_ticker(Path(data_dir))

    reports = []
    for ticker in sorted(set(chroma) | set(metadata)):
        indexed = chroma.get(ticker, {"chunks": 0, "accessions": set()})
        meta = metadata.get(ticker)
        records = meta.get("records", []) if meta else []

        report = TickerReport(
            ticker=ticker,
            chunks=indexed["chunks"],
            indexed_filings=len(indexed["accessions"]),
            metadata_records=len(records),
            missing_raw=len(meta.get("missing_raw", [])) if meta else 0,
        )

        if report.chunks == 0:
            report.problems.append("nothing indexed")
        if meta is None:
            # Chunks with no metadata file still answer questions, so this is not
            # disqualifying - but re-indexing would silently drop the company.
            report.warnings.append("no metadata file")
        elif meta.get("unreadable"):
            report.problems.append("metadata file unreadable")
        elif not records:
            report.problems.append("no filings on record")

        if meta and indexed["accessions"]:
            unindexed = meta["accessions"] - indexed["accessions"]
            report.unindexed_filings = len(unindexed)
            if unindexed:
                report.warnings.append(f"{len(unindexed)} filing(s) downloaded but not indexed")
        if report.missing_raw:
            report.warnings.append(f"{report.missing_raw} source file(s) missing from data/raw")

        reports.append(report)
    return reports


def orphan_raw_files(ticker: str, data_dir: Path | str = DATA_DIR) -> list[Path]:
    """Downloaded filings on disk for one ticker, left behind after a prune."""
    raw_dir = Path(data_dir) / "raw"
    if not raw_dir.is_dir():
        return []
    return sorted(raw_dir.glob(f"{ticker}_*"))


def prune(
    reports: list[TickerReport],
    data_dir: Path | str = DATA_DIR,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    dry_run: bool = True,
) -> list[dict]:
    """Remove unusable companies from the store and drop their metadata file.

    Source files under data/raw are deliberately left alone - they are the only
    copy of what was downloaded, and re-indexing needs them. The returned
    actions name them so the caller can report what is still on disk.
    """
    metadata = _metadata_by_ticker(Path(data_dir))
    actions = []
    for report in reports:
        if report.healthy:
            continue
        meta_path = metadata.get(report.ticker, {}).get("path")
        action = {
            "ticker": report.ticker,
            "reason": "; ".join(report.problems),
            "chunks_removed": report.chunks,
            "metadata_file": meta_path,
            "raw_files_left": len(orphan_raw_files(report.ticker, data_dir)),
        }
        if not dry_run:
            action["chunks_removed"] = delete_company(report.ticker, persist_dir=persist_dir)
            if meta_path and Path(meta_path).exists():
                Path(meta_path).unlink()
        actions.append(action)
    return actions


def _print_table(reports: list[TickerReport]) -> None:
    header = f"{'ticker':<8}{'chunks':>8}{'filings':>9}{'on record':>11}  status"
    print(header)
    print("-" * len(header))
    for r in reports:
        if r.problems:
            status = "BROKEN - " + "; ".join(r.problems)
        elif r.warnings:
            status = "ok, but " + "; ".join(r.warnings)
        else:
            status = "ok"
        print(
            f"{r.ticker:<8}{r.chunks:>8}{r.indexed_filings:>9}"
            f"{r.metadata_records:>11}  {status}"
        )


def main() -> None:
    parser = ArgumentParser(
        description="Check that every indexed company is usable, and optionally prune the ones that are not."
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory holding metadata/ and raw/.")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR), help="Chroma store directory.")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Show which unusable companies would be removed. Deletes nothing on its own.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="With --prune, actually delete. Without it, --prune is a dry run.",
    )
    args = parser.parse_args()

    reports = audit(args.data_dir, args.persist_dir)
    if not reports:
        print("Nothing indexed yet. Run `python -m src.build_index` first.")
        return

    _print_table(reports)
    broken = [r for r in reports if not r.healthy]
    warned = [r for r in reports if r.healthy and r.warnings]
    print(
        f"\n{len(reports)} companies: {len(reports) - len(broken)} usable, "
        f"{len(broken)} broken, {len(warned)} with warnings."
    )

    if not broken:
        print("Nothing to prune.")
        return

    if not args.prune:
        print("Re-run with --prune to see what would be removed.")
        return

    actions = prune(reports, args.data_dir, args.persist_dir, dry_run=not args.yes)
    verb = "Removed" if args.yes else "Would remove"
    print()
    for action in actions:
        print(f"{verb} {action['ticker']}: {action['reason']}")
        print(f"    {action['chunks_removed']} chunks from the vector store")
        if action["metadata_file"]:
            print(f"    metadata file {Path(action['metadata_file']).name}")
        if action["raw_files_left"]:
            print(
                f"    leaving {action['raw_files_left']} downloaded filing(s) in data/raw "
                "(delete by hand if you want the disk space)"
            )
    if not args.yes:
        print("\nDry run - nothing was deleted. Add --yes to apply.")


if __name__ == "__main__":
    main()
