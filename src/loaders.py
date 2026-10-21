"""Read source files without applying Q1–Q5 business rules.

CSV cells remain strings (including empty strings) to preserve identifiers,
localized numbers and dates. No rows are deduplicated or filtered.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _csv(path: str | Path, encoding: str, first_column: str) -> pd.DataFrame:
    # Locate the actual header; service lines are not data records.
    with Path(path).open(encoding=encoding, newline="") as source:
        for index, line in enumerate(source):
            if line.rstrip("\r\n").split(";")[0].strip('"') == first_column:
                break
        else:
            raise ValueError(f"CSV header {first_column!r} not found in {path}")
    return pd.read_csv(path, sep=";", encoding=encoding, skiprows=index,
                       dtype=str, keep_default_na=False)


def load_crm(path: str | Path = DATA_DIR / "01_crm_orders_export.csv") -> pd.DataFrame:
    """Read CRM using UTF-8 BOM and preserve raw cell values."""
    return _csv(path, "utf-8-sig", "Номер заказа")


def load_bank(path: str | Path = DATA_DIR / "02_bank_statement_rub.csv") -> pd.DataFrame:
    """Read cp1251 bank export, skipping service lines before the header."""
    return _csv(path, "cp1251", "Дата операции")


def load_catalog(path: str | Path = DATA_DIR / "03_catalog_prices_reference.xlsx") -> dict[str, pd.DataFrame]:
    """Read every sheet, including hidden sheets, with Excel-native cell types.

    Sheet visibility and the number of skipped rows are recorded in attrs.
    Formula text is preserved rather than replaced by potentially stale caches.
    """
    workbook = load_workbook(path, read_only=True, data_only=False)
    result = {}
    try:
        for sheet in workbook:
            rows = list(sheet.values)
            # A title row contains one cell; table headers contain several.
            start = next((i for i, row in enumerate(rows)
                          if sum(value is not None for value in row) > 1), None)
            if start is None:
                raise ValueError(f"No table header in sheet {sheet.title!r}")
            frame = pd.DataFrame(rows[start + 1:], columns=rows[start])
            frame.attrs.update(sheet_state=sheet.sheet_state, header_row=start + 1)
            result[sheet.title] = frame
    finally:
        workbook.close()
    return result


@dataclass(frozen=True)
class Message:
    timestamp: datetime
    author: str
    text: str


@dataclass(frozen=True)
class Chat:
    chat_id: str
    client_id: str
    order_id: str
    manager: str
    messages: tuple[Message, ...]
    source_name: str
    raw_text: str


_HEADER = re.compile(r"Чат ([^|]+) \| клиент ([^|]+) \| заказ ([^|]+) \| менеджер (.+)")
_MESSAGE = re.compile(r"\[(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})\] ([^:]+):\s?(.*)")


def load_chats(path: str | Path = DATA_DIR / "04_client_chats.zip") -> list[Chat]:
    """Read UTF-8 conversations directly from ZIP, without extracting files.

    Unexpected nonempty lines fail explicitly instead of silently losing text.
    Archive and message order are preserved.
    """
    chats = []
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            raw = archive.read(entry).decode("utf-8-sig")
            lines = raw.splitlines()
            header = _HEADER.fullmatch(lines[0]) if lines else None
            if header is None:
                raise ValueError(f"Invalid chat header: {entry.filename}")
            messages = []
            for number, line in enumerate(lines[1:], 2):
                if not line.strip():
                    continue
                match = _MESSAGE.fullmatch(line)
                if match is None:
                    raise ValueError(f"Invalid message: {entry.filename}:{number}")
                stamp, author, text = match.groups()
                messages.append(Message(datetime.strptime(stamp, "%d.%m.%Y %H:%M"), author, text))
            chats.append(Chat(*header.groups(), tuple(messages), entry.filename, raw))
    return chats


def load_answers_template(path: str | Path = DATA_DIR / "answers_template.json") -> dict:
    """Read the template unchanged; placeholder values are not answers."""
    with Path(path).open(encoding="utf-8-sig") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise ValueError("Answers template must be a JSON object")
    return result
