"""Structural integration checks and isolated loader edge cases; no Q answers."""
import hashlib
import json
from datetime import datetime
from zipfile import ZipFile

import pytest
from openpyxl import Workbook

from src.loaders import (
    DATA_DIR, load_answers_template, load_bank, load_catalog, load_chats, load_crm,
)


@pytest.fixture(scope="module", autouse=True)
def sources_unchanged():
    def hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in DATA_DIR.iterdir() if p.is_file()}
    before = hashes()
    yield
    assert hashes() == before


def test_actual_csv_structure():
    crm, bank = load_crm(), load_bank()
    assert not crm.empty and len(crm.columns) == 14
    assert not bank.empty and len(bank.columns) == 7
    assert crm.columns[0] == "Номер заказа"
    assert bank.columns[0] == "Дата операции"
    assert all(isinstance(v, str) for frame in (crm, bank)
               for v in frame.to_numpy().flat)
    assert "" in crm["Скидка"].values


def test_actual_workbook():
    sheets = load_catalog()
    assert set(sheets) == {"Каталог", "Переименования", "Прайс_история", "Менеджеры"}
    assert all(not frame.empty for frame in sheets.values())
    assert sheets["Прайс_история"].attrs["sheet_state"] == "hidden"
    assert sheets["Каталог"].attrs["header_row"] == 2


def test_actual_chats():
    chats = load_chats()
    assert len(chats) == 150
    assert all(c.chat_id and c.raw_text and c.messages for c in chats)
    assert all(isinstance(m.timestamp, datetime) for c in chats for m in c.messages)
    assert all(c.source_name.endswith(".txt") for c in chats)


def test_actual_template():
    result = load_answers_template()
    assert result == json.loads((DATA_DIR / "answers_template.json").read_text(encoding="utf-8-sig"))
    assert "q1_unique_orders" in result and "run_command" in result


@pytest.mark.parametrize("loader,encoding,header", [
    (load_crm, "utf-8-sig", "Номер заказа"),
    (load_bank, "cp1251", "Дата операции"),
])
def test_csv_preserves_values_and_duplicates(tmp_path, loader, encoding, header):
    path = tmp_path / "source.csv"
    path.write_text(f'Служебная строка\n{header};Поле\n0001;"текст; внутри"\n0001;"текст; внутри"\nNA;\n', encoding=encoding)
    frame = loader(path)
    assert frame.iloc[:, 0].tolist() == ["0001", "0001", "NA"]
    assert frame.iloc[:, 1].tolist() == ["текст; внутри", "текст; внутри", ""]


def test_missing_csv_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("wrong;columns", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_crm(path)


def test_hidden_sheet_and_formula(tmp_path):
    path = tmp_path / "book.xlsx"
    book = Workbook()
    book.active.append(["Название"])
    book.active.append(["Артикул", "Цена"])
    book.active.append(["001", "=1+2"])
    hidden = book.create_sheet("Скрытый")
    hidden.sheet_state = "hidden"
    hidden.append(["Код", "Дата"])
    hidden.append(["002", datetime(2025, 1, 1)])
    book.save(path)
    book.close()
    result = load_catalog(path)
    assert result["Sheet"].iloc[0].tolist() == ["001", "=1+2"]
    assert result["Скрытый"].attrs["sheet_state"] == "hidden"


def test_chat_spaces_order_and_raw_text(tmp_path):
    path = tmp_path / "chats.zip"
    raw = "Чат C9999 | клиент CL 123 | заказ  a-123 | менеджер Имя\n[01.02.2025 03:04] Клиент: Цена: 10 ₽\n"
    with ZipFile(path, "w") as archive:
        archive.writestr("chats/", "")
        archive.writestr("chats/example.txt", raw.encode("utf-8-sig"))
    chat, = load_chats(path)
    assert chat.client_id == "CL 123"
    assert chat.order_id == " a-123"
    assert chat.raw_text == raw
    assert chat.messages[0].text == "Цена: 10 ₽"
    assert chat.messages[0].timestamp == datetime(2025, 2, 1, 3, 4)


@pytest.mark.parametrize("raw", ["", "invalid", "Чат C1 | клиент X | заказ Y | менеджер Z\ninvalid"])
def test_malformed_chat_rejected(tmp_path, raw):
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("bad.txt", raw)
    with pytest.raises(ValueError):
        load_chats(path)


def test_invalid_template(tmp_path):
    path = tmp_path / "template.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_answers_template(path)
