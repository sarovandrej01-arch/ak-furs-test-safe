import hashlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.crm import clean_crm, load_clean_crm, summarize_q1
from src.loaders import DATA_DIR, load_crm


def record(order="A-75", line="1", exported="2025-01-01 09:00", **updates):
    row = {"Номер заказа": order, "№ строки": line,
           "Выгружено": exported, "Дата создания": "45658",
           "Статус": "PAID", "Менеджер": "исходный менеджер",
           "Цена за ед.": "17,000.00", "Скидка": "7%", "Комментарий": ""}
    row.update(updates)
    return row


def test_latest_snapshot_and_multiple_lines():
    raw = pd.DataFrame([
        record(exported="2025-01-02 09:00", **{"Комментарий": "new"}),
        record(order="А-00075", exported="45658.5", **{"Комментарий": "old"}),
        record(order="a75", line="2", **{"Статус": "возврат"}),
        record(order="A76"),
    ])
    before = raw.copy(deep=True)
    result = clean_crm(raw)
    assert_frame_equal(raw, before)
    assert summarize_q1(result) == {"q1_unique_orders": 2, "q1_unique_order_lines": 3}
    assert result["order_id"].tolist() == [75, 75, 76]
    assert result["line_id"].tolist() == [1, 2, 1]
    assert result["Комментарий"].tolist() == ["new", "", ""]
    assert result["status"].tolist() == ["Оплачен", "Возврат", "Оплачен"]
    assert result["Цена за ед."].tolist() == ["17,000.00"] * 3
    assert result["Скидка"].tolist() == ["7%"] * 3
    assert result["source_row"].tolist() == [1, 3, 4]


def test_equal_dates_last_input_row_wins():
    raw = pd.DataFrame([record(exported="45658.5", **{"Статус": "PAID"}),
                        record(order="А75", line="01", exported="01/01/2025 12:00",
                               **{"Статус": "CANCELLED"})], index=[10, 10])
    result = clean_crm(raw)
    assert len(result) == 1
    assert result.iloc[0]["status"] == "Отменён"
    assert result.iloc[0]["source_row"] == 2
    assert_frame_equal(result, clean_crm(raw))
    assert_frame_equal(result, clean_crm(result))


def test_invalid_key_or_date_is_not_silently_dropped():
    for updates in ({"Номер заказа": "bad75"}, {"№ строки": ""},
                    {"Выгружено": ""}, {"Дата создания": "bad"}):
        with pytest.raises(ValueError, match="CRM record 1, field"):
            clean_crm(pd.DataFrame([record(**updates)]))


def test_missing_required_column():
    with pytest.raises(ValueError, match="Missing required CRM column"):
        clean_crm(pd.DataFrame([record()]).drop(columns="Выгружено"))


def test_empty_crm():
    result = clean_crm(pd.DataFrame(columns=record().keys()))
    assert summarize_q1(result) == {"q1_unique_orders": 0, "q1_unique_order_lines": 0}


def test_actual_crm_repeatability_latest_rows_and_source_integrity():
    before_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in DATA_DIR.iterdir() if p.is_file()}
    raw = load_crm()
    result = clean_crm(raw)
    assert_frame_equal(result, load_clean_crm())
    assert_frame_equal(result, clean_crm(result))
    assert not result.duplicated(["order_id", "line_id"]).any()
    assert set(result["status"]) <= {"Оплачен", "Отменён", "Возврат", "В работе"}
    # Independently retain the original rows chosen by latest export timestamp.
    from src.normalizers import normalize_datetime, normalize_order_id, normalize_line_id
    expected = {}
    for position, (_, row) in enumerate(raw.iterrows(), 1):
        key = (normalize_order_id(row["Номер заказа"]), normalize_line_id(row["№ строки"]))
        candidate = (normalize_datetime(row["Выгружено"]), position)
        expected[key] = max(expected.get(key, candidate), candidate)
    assert len(result) == len(expected)
    for _, row in result.iterrows():
        assert (row["exported_at"], row["source_row"]) == expected[(row["order_id"], row["line_id"])]
    selected_raw = raw.iloc[result["source_row"].to_numpy() - 1].reset_index(drop=True)
    assert_frame_equal(result[raw.columns], selected_raw)
    assert before_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in DATA_DIR.iterdir() if p.is_file()}
