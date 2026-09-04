"""Week 4: pull financial tables out of filing HTML as their own readable chunks.

EDGAR filings use `<table>` for two very different things: real financial data
(balance sheets, compensation tables) and pure page layout (cover-page boxes,
checkbox grids, spacing). Flattening both into paragraph text - what the Week 2
chunker did - turned the financial statements into unlabeled runs of numbers.

This module separates the two, renders a data table as one row per line with its
title attached, and leaves layout tables alone.
"""

from __future__ import annotations

import re
from typing import Any

# A cell that is a figure: 1,234 / (1,234) / $1,234.56 / 12.5% / -3
NUMERIC_CELL = re.compile(r"^\(?\$?\s*-?\d[\d,]*(\.\d+)?\s*\)?%?$")

# Cells EDGAR emits purely for alignment.
FILLER_CELLS = {"$", "(", ")", "%", "*", "—", "-", "|"}

MIN_ROWS = 3
MIN_NUMERIC_CELLS = 3
TITLE_CHAR_BUDGET = 180
TITLE_MAX_CHARS = 250


def clean_text(value: str) -> str:
    """Collapse whitespace and drop the zero-width spaces EDGAR pads cells with."""
    return " ".join(value.replace("​", " ").replace("\xa0", " ").split())


def table_rows(table: Any) -> list[list[str]]:
    """Return the table as rows of non-empty, non-filler cell text."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["td", "th"])]
        cells = [cell for cell in cells if cell and cell not in FILLER_CELLS]
        if cells:
            rows.append(cells)
    return rows


def _is_table_of_contents(rows: list[list[str]]) -> bool:
    """A filing's TOC is numeric and wide enough to pass as data. Exclude it."""
    flattened = " ".join(" ".join(row) for row in rows).lower()
    if "table of contents" in flattened:
        return True
    if any(row and row[0].lower() == "page" for row in rows):
        return True
    item_rows = sum(1 for row in rows if row and re.match(r"(?i)^(item|part)\b", row[0]))
    return item_rows >= max(3, len(rows) // 2)


def is_data_table(table: Any) -> bool:
    """True when a `<table>` carries real tabular data rather than page layout."""
    rows = table_rows(table)
    if len(rows) < MIN_ROWS:
        return False
    if sum(1 for row in rows for cell in row if NUMERIC_CELL.match(cell)) < MIN_NUMERIC_CELLS:
        return False
    if max(len(row) for row in rows) < 2:
        return False
    return not _is_table_of_contents(rows)


def table_title(table: Any, data_table_ids: set[int]) -> str:
    """Nearest preceding prose, used as the table's caption.

    Filings rarely mark up a caption; the title sits in whatever paragraph runs
    just above the table ("CONDENSED BALANCE SHEETS (USD in thousands)", or
    "The following table sets forth ..."). Text belonging to another data table
    is skipped so captions do not bleed between adjacent statements.
    """
    pieces: list[str] = []
    budget = 0
    for node in table.find_all_previous(string=True):
        if budget > TITLE_CHAR_BUDGET:
            break
        if any(id(parent) in data_table_ids for parent in node.find_parents("table")):
            continue
        text = clean_text(str(node))
        if not text or text in FILLER_CELLS:
            continue
        pieces.append(text)
        budget += len(text)
    return " ".join(reversed(pieces))[-TITLE_MAX_CHARS:].strip()


def render_table(rows: list[list[str]]) -> str:
    """One row per line, cells pipe-separated, so figures keep their row label."""
    return "\n".join(" | ".join(row) for row in rows)


def drop_table_of_contents(soup: Any) -> int:
    """Remove TOC tables outright and return how many were dropped.

    A filing's table of contents lists every `Item N.` heading. Left in the
    document those cells read as real section headings, so the chunker opens a
    section at the TOC entry and files the front matter under it. Removing them
    is what keeps section boundaries anchored to the actual body.
    """
    dropped = 0
    for table in soup.find_all("table"):
        if _is_table_of_contents(table_rows(table)):
            table.decompose()
            dropped += 1
    return dropped


def extract_data_tables(soup: Any, marker_template: str) -> list[dict[str, str]]:
    """Replace each data table with a marker paragraph and return what was removed.

    Returns one record per extracted table, in document order, with its title and
    rendered rows. The caller walks the modified soup and swaps each marker back
    for a table chunk, which keeps tables positioned inside the right section.
    """
    tables = soup.find_all("table")
    data_tables = [table for table in tables if is_data_table(table)]
    data_table_ids = {id(table) for table in data_tables}

    # Titles are read from the surrounding document, so resolve them all before
    # mutating the tree.
    extracted = [
        {
            "title": table_title(table, data_table_ids),
            "body": render_table(table_rows(table)),
        }
        for table in data_tables
    ]

    for index, table in enumerate(data_tables):
        marker = soup.new_tag("p")
        marker.string = marker_template.format(index=index)
        table.replace_with(marker)

    return extracted
