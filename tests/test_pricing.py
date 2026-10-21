"""Проверки Q3: граничные условия, диагностика и локальные источники."""
from datetime import date
from decimal import Decimal
from fractions import Fraction
import hashlib

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.crm import clean_crm, load_clean_crm, summarize_q1
from src.bank import reconcile_bank
from src.loaders import DATA_DIR, load_bank, load_catalog
from src.pricing import (
    build_manager_map, build_price_history, build_sku_map, calculate_discount,
    effective_unit_price, find_price_violations, historical_price,
    normalize_manager, normalize_sku, parse_decimal,
)


@pytest.fixture
def sheets():
    return {
        "Каталог": pd.DataFrame({"Артикул": ["NEW"]}),
        "Переименования": pd.DataFrame([
            ["OLD", "MID", "2025-03-01"], ["MID", "NEW", "2025-06-01"]
        ], columns=["Старый артикул", "Новый артикул", "Дата перехода"]),
        "Прайс_история": pd.DataFrame([
            ["NEW", "2025-01-01", "2025-06-30", 90],
            ["NEW", "2025-07-01", None, 100],
        ], columns=["Артикул", "Действует с", "Действует по", "Минимально допустимая цена, ₽"]),
        "Менеджеры": pd.DataFrame({"Каноническое имя": ["Анна Иванова"],
                                  "Варианты написания в CRM": ["Аня | Ivanova A | Иванова А."]}),
    }


def crm(*updates):
    rows = []
    for i, changes in enumerate(updates, 1):
        row = {"Номер заказа": f"A{i}", "№ строки": "1", "Дата создания": "2025-02-01",
               "Выгружено": "2026-10-01 09:00", "Статус": "PAID", "Артикул": "OLD",
               "Цена за ед.": "100", "Кол-во": "2", "Скидка": "", "Менеджер": "Аня"}
        rows.append({**row, **changes})
    return clean_crm(pd.DataFrame(rows))


@pytest.mark.parametrize("raw,expected", [
    ("2,48 млн", "2480000"), ("2.48 млн", "2480000"), ("1,5 тыс.", "1500"),
    ("17,000.00", "17000"), ("17.000,00", "17000"), ("50 000", "50000"),
    ("50\u00a0000,25", "50000.25"), ("50\u202f000.25", "50000.25"),
    ("6700,00", "6700"), ("0,01", "0.01"), (123, "123"), (Decimal("0.1"), "0.1"),
])
def test_decimal_formats(raw, expected):
    assert parse_decimal(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["1,234", "12 34", "17,00.00", "", None, True, 0.1, "NaN", Decimal("Infinity")])
def test_invalid_decimal(raw):
    with pytest.raises(ValueError):
        parse_decimal(raw)


@pytest.mark.parametrize("discount,expected_amount,expected_unit", [
    ("10%", "20", "90"), ("30", "30", "85"),
    ("", "0", "100"), (None, "0", "100"), (" 5,5 % ", "11", "94.5"),
])
def test_discounts_apply_to_entire_line(discount, expected_amount, expected_unit):
    price, qty = Decimal(100), Decimal(2)
    assert calculate_discount(price, qty, discount) == Decimal(expected_amount)
    assert effective_unit_price(price, qty, discount) == Decimal(expected_unit)


def test_decimal_exact_cents_and_no_kopeck_rounding():
    assert effective_unit_price(Decimal("0.3"), Decimal(2), "0.20") == Decimal("0.2")
    assert effective_unit_price(Decimal("0.1"), Decimal(2), "1%") == Decimal("0.099")


@pytest.mark.parametrize("price,qty,discount", [
    ("100", "0", ""), ("100", "-1", ""), ("-1", "1", ""),
    ("100", "1", "101%"), ("100", "1", "-1%"),
    ("100", "1", "101"), ("100", "1", "-1"),
])
def test_invalid_discount_or_quantity(price, qty, discount):
    with pytest.raises(ValueError):
        effective_unit_price(Decimal(price), Decimal(qty), discount)


def test_sku_chain_preserves_old_codes(sheets):
    mapping = build_sku_map(sheets["Каталог"], sheets["Переименования"])
    assert mapping == {"OLD": "NEW", "MID": "NEW", "NEW": "NEW"}
    assert normalize_sku(" old ") == "OLD"


@pytest.mark.parametrize("edges", [
    [("OLD", "MID"), ("MID", "OLD")],
    [("OLD", "MID"), ("OLD", "NEW")],
    [("OLD", "MISSING")], [("OLD", "OLD")],
])
def test_invalid_rename_graph(sheets, edges):
    renames = pd.DataFrame([(a, b, "2025-01-01") for a, b in edges],
                           columns=sheets["Переименования"].columns)
    with pytest.raises(ValueError):
        build_sku_map(sheets["Каталог"], renames)


@pytest.mark.parametrize("day,minimum", [
    ("2024-12-31", None), ("2025-01-01", 90), (45658, 90),
    ("2025-06-30 23:59", 90), ("2025-07-01", 100), ("2027-01-01", 100),
])
def test_price_period_boundaries(sheets, day, minimum):
    mapping = build_sku_map(sheets["Каталог"], sheets["Переименования"])
    history = build_price_history(sheets["Прайс_история"], mapping)
    period = historical_price("NEW", day, history)
    assert (period.minimum if period else None) == minimum


def test_price_gap_and_expired_period(sheets):
    history = sheets["Прайс_история"].iloc[:1].copy()
    prices = build_price_history(history, {"NEW": "NEW"})
    assert historical_price("NEW", "2025-07-01", prices) is None
    assert historical_price("UNKNOWN", "2025-01-01", prices) is None


def test_price_overlap_rejected(sheets):
    history = sheets["Прайс_история"].copy()
    history.loc[1, "Действует с"] = "2025-06-30"
    with pytest.raises(ValueError, match="Пересекающиеся"):
        build_price_history(history, {"NEW": "NEW"})


def test_missing_period_start_rejected(sheets):
    history = sheets["Прайс_история"].copy()
    history.loc[0, "Действует с"] = None
    with pytest.raises(ValueError, match="Отсутствует начало"):
        build_price_history(history, {"NEW": "NEW"})


def test_missing_creation_date_diagnosed(sheets):
    rows = crm({})
    rows.loc[0, "created_at"] = pd.NaT
    result = find_price_violations(rows, sheets)
    assert result.diagnostics.issue.tolist() == ["invalid_value"]


def test_managers(sheets):
    mapping = build_manager_map(sheets["Менеджеры"])
    for alias in [" АНЯ ", "ivanova a", "Анна   Иванова", "Иванова А."]:
        assert normalize_manager(alias, mapping) == "Анна Иванова"
    assert normalize_manager("неизвестно", mapping) is None
    conflict = pd.concat([sheets["Менеджеры"], pd.DataFrame({
        "Каноническое имя": ["Другая Анна"], "Варианты написания в CRM": ["Аня"]})])
    with pytest.raises(ValueError, match="Неоднозначный"):
        build_manager_map(conflict)


def test_violation_equal_minimum_cancelled_and_date_of_creation(sheets):
    rows = crm({"Скидка": "10%"}, {"Скидка": "30"},
               {"Дата создания": "2025-07-01", "Цена за ед.": "95"},
               {"Статус": "CANCELLED", "Цена за ед.": "1"},
               {"Статус": "REFUND", "Скидка": "30"},
               {"Статус": "NEW", "Скидка": "30"})
    result = find_price_violations(rows, sheets)
    assert result.cancelled_count == 1
    assert len(result.checked) == result.eligible_count == 5
    assert result.violations.order_id.tolist() == [2, 3, 5, 6]
    assert result.checked.iloc[0].minimum_price == Decimal(90)
    assert result.answers() == {"q3_price_violation_lines_total": 4,
                                "q3_price_violation_lines_by_manager": {"Анна Иванова": 4}}


def test_excel_dates_in_reference_and_order(sheets):
    sheets["Прайс_история"].loc[0, "Действует с"] = "45658"
    result = find_price_violations(crm({"Дата создания": "45658", "Скидка": "30"}), sheets)
    assert result.answers()["q3_price_violation_lines_total"] == 1
    assert result.checked.iloc[0].price_from == date(2025, 1, 1)


@pytest.mark.parametrize("changes,issue", [
    ({"Артикул": "MISSING"}, "unknown_sku"),
    ({"Дата создания": "2024-01-01"}, "missing_price"),
    ({"Кол-во": "0"}, "invalid_value"),
    ({"Цена за ед.": "bad"}, "invalid_value"),
    ({"Скидка": "999%"}, "invalid_value"),
])
def test_explicit_diagnostics(sheets, changes, issue):
    result = find_price_violations(crm(changes), sheets)
    assert result.checked.empty
    assert result.diagnostics.issue.tolist() == [issue]
    assert result.missing_price_count == int(issue in {"unknown_sku", "missing_price"})


def test_unknown_manager_not_lost_and_zero_manager_included(sheets):
    result = find_price_violations(crm({"Менеджер": "Неизвестный", "Скидка": "30"}), sheets)
    assert result.diagnostics.issue.tolist() == ["unknown_manager"]
    assert result.answers()["q3_price_violation_lines_by_manager"] == {
        "Анна Иванова": 0, "Неизвестный менеджер: Неизвестный": 1}


def test_duplicate_cleaned_keys_rejected(sheets):
    rows = crm({})
    with pytest.raises(ValueError, match="повторяющихся"):
        find_price_violations(pd.concat([rows, rows]), sheets)


def test_empty_crm(sheets):
    result = find_price_violations(crm({}).iloc[:0], sheets)
    assert result.answers()["q3_price_violation_lines_total"] == 0
    assert result.cancelled_count == result.eligible_count == 0


def test_real_sources_repeatability_integrity_and_regressions():
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DATA_DIR.iterdir() if p.is_file()}
    rows, sheets = load_clean_crm(), load_catalog()
    original = rows.copy(deep=True)
    original_sheets = {name: frame.copy(deep=True) for name, frame in sheets.items()}
    q1 = summarize_q1(rows)
    q2 = reconcile_bank(rows, load_bank())
    result = find_price_violations(rows, sheets)
    repeated = find_price_violations(load_clean_crm(), load_catalog())
    assert_frame_equal(result.checked, repeated.checked)
    assert_frame_equal(result.diagnostics, repeated.diagnostics)
    assert result.answers() == repeated.answers()
    assert result.diagnostics.empty
    assert result.eligible_count + result.cancelled_count == len(rows)
    assert len(result.checked) == result.eligible_count
    assert sum(result.answers()["q3_price_violation_lines_by_manager"].values()) == len(result.violations)
    assert not result.checked.duplicated(["order_id", "line_id"]).any()
    assert not result.checked.status.eq("Отменён").any()
    for _, row in result.checked.iterrows():
        assert isinstance(row.effective_unit_price, Decimal)
        assert row.is_violation == (row.net_line_total < row.minimum_price * row.quantity)
        assert row.price_from <= row.created_at.date()
        assert row.price_to is None or row.created_at.date() <= row.price_to
        # Независимая проверка точной арифметикой рациональных чисел.
        raw_discount = str(row["Скидка"]).strip()
        gross = Fraction(row.unit_price) * Fraction(row.quantity)
        if raw_discount.endswith("%"):
            discount = gross * Fraction(parse_decimal(raw_discount[:-1])) / 100
        else:
            discount = Fraction(parse_decimal(raw_discount)) if raw_discount else Fraction(0)
        exact_price = (gross - discount) / Fraction(row.quantity)
        assert row.is_violation == (exact_price < Fraction(row.minimum_price))
    assert_frame_equal(rows, original)
    for name in sheets:
        assert_frame_equal(sheets[name], original_sheets[name])
    assert summarize_q1(load_clean_crm()) == q1
    q2_after = reconcile_bank(load_clean_crm(), load_bank())
    assert q2.paid_orders_without_payment == q2_after.paid_orders_without_payment
    assert q2.unidentified_sum_rub == q2_after.unidentified_sum_rub
    assert q2.unidentified_count == q2_after.unidentified_count
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in DATA_DIR.iterdir() if p.is_file()}
