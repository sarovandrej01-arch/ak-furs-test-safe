"""Проверки Q2 на искусственных примерах и исходных локальных файлах."""
from decimal import Decimal
import hashlib

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.bank import (
    extract_order_ids, find_paid_orders_without_payment, is_incoming,
    match_incoming_payments, paid_order_ids, parse_rub_amount, reconcile_bank,
    summarize_unidentified_payments,
)
from src.crm import load_clean_crm, summarize_q1
from src.loaders import DATA_DIR, load_bank


def bank_rows(*rows):
    """Собрать маленькую выписку без зависимости от реального содержимого."""
    return pd.DataFrame([
        {"Тип": kind, "Сумма": amount, "Валюта": "RUR", "Назначение платежа": purpose}
        for kind, amount, purpose in rows
    ], columns=["Тип", "Сумма", "Валюта", "Назначение платежа"])


@pytest.mark.parametrize("purpose,expected", [
    ("Оплата заказа A-00075", (75,)),
    ("Оплата по заказу А-75", (75,)),
    (" a-75 доплата", (75,)),
    ("A75 доплата", (75,)),
    ("заказ а \u00a0- 00075", (75,)),
    ("Предоплата заказ 75, тел. +7 999 000-75-00", (75,)),
    ("зак. 00075", (75,)),
    ("ЗАКАЗ № 75", (75,)),
    ("Оплата по договору", ()),
    ("сумма 75,00 от 01.02.2025, счёт №75, тел. +7 999 000-75-00", ()),
    ("Счёт № 00075 от 16.02.2026", ()),
    ("по счету №75 от 13.12.2025", ()),
    ("Счёт № A75", ()),
    ("договор A75", ()),
    ("дата 2025-01-01, счёт 40702810000000000075", ()),
    ("XA75 A75tail A--75 B-A75 A75.00 A75/12", ()),
    ("заказ 01.02.2025", ()),
    ("заказ 75,00 рублей", ()),
    ("A75; А-00076 и a77", (75, 76, 77)),
    ("заказы №75, 76 и №77", (75, 76, 77)),
    ("A75, 76 и 77", (75, 76, 77)),
    ("A75, сумма 76 руб.", (75,)),
    ("заказ 75, сумма 76 руб., счёт №77", (75,)),
    ("заказ 75; A76; A75", (75, 76)),
    ("заказ A0 и A75", (75,)),
    ("", ()),
])
def test_extract_order_ids(purpose, expected):
    assert extract_order_ids(purpose) == expected


@pytest.mark.parametrize("value,expected", [
    ("Поступление", True), (" поступление ", True),
    ("Списание", False), ("СПИСАНИЕ", False),
])
def test_incoming_type(value, expected):
    assert is_incoming(value) is expected


def test_unknown_type_fails():
    with pytest.raises(ValueError, match="Неизвестный тип"):
        is_incoming("Неизвестно")


@pytest.mark.parametrize("value,expected", [
    ("1 234,56", "1234.56"), ("1\u00a0234,56", "1234.56"),
    ("1\u202f234,56", "1234.56"), ("0,01", "0.01"),
    ("100", "100.00"), ("-100,01", "-100.01"),
    ("+12.34", "12.34"), (Decimal("0.10"), "0.10"),
])
def test_money(value, expected):
    assert parse_rub_amount(value) == Decimal(expected)


@pytest.mark.parametrize("value", [0.1, "NaN", "", "12 34,56", "1,234.56", "0.001", Decimal("NaN"), Decimal("0.001")])
def test_invalid_money(value):
    with pytest.raises(ValueError):
        parse_rub_amount(value)


def test_matching_repeats_outgoing_multiple_unknown_and_exact_sum():
    crm = pd.DataFrame({"order_id": [75, 75, 76, 77, 78],
                        "status": ["Оплачен", "Возврат", "Оплачен", "Оплачен", "В работе"]})
    bank = bank_rows(
        ("Поступление", "0,01", "A75"),
        ("Поступление", "0,02", "А-00075"),
        ("Списание", "-100,00", "заказ A77"),
        ("Поступление", "0,10", "Перевод"),
        ("Поступление", "0,20", "A9999"),
        ("Поступление", "0,30", "заказы 76, 78 и 9999"),
        ("Поступление", "0,40", "Счёт №77 от 01.01.2025"),
    )
    original_crm, original_bank = crm.copy(deep=True), bank.copy(deep=True)
    result = reconcile_bank(crm, bank)
    assert result.operations_count == 7
    assert result.incoming_count == 6
    assert result.identified_count == 3
    assert result.unidentified_count == 3
    assert result.unidentified_sum_rub == Decimal("0.70")
    assert result.paid_orders_without_payment == (77,)
    assert result.payments.iloc[-2]["matched_order_ids"] == (76, 78)
    assert result.payments.iloc[-2]["candidate_order_ids"] == (76, 78, 9999)
    assert result.payments["bank_row"].tolist() == [1, 2, 4, 5, 6, 7]
    assert_frame_equal(crm, original_crm)
    assert_frame_equal(bank, original_bank)
    again = reconcile_bank(crm, bank)
    assert_frame_equal(result.payments, again.payments)
    assert result.paid_orders_without_payment == again.paid_orders_without_payment
    assert result.unidentified_sum_rub == again.unidentified_sum_rub


def test_mixed_status_any_paid_line_and_sorted_unique_result():
    crm = pd.DataFrame({"order_id": [76, 75, 75, 74, 74],
                        "status": ["paid", "Оплачен", "В работе", "Возврат", "Отменён"]})
    payments = match_incoming_payments(bank_rows(), crm["order_id"])
    assert paid_order_ids(crm) == {75, 76}
    assert find_paid_orders_without_payment(crm, payments) == [75, 76]
    assert summarize_unidentified_payments(payments) == (0, Decimal("0.00"))


def test_outgoing_positive_amount_does_not_count_as_incoming():
    matched = match_incoming_payments(bank_rows(("Списание", "100,00", "A75")), [75])
    assert matched.empty


@pytest.mark.parametrize("column,value", [("Сумма", "-1,00"), ("Валюта", "USD"), ("Тип", "неизвестно")])
def test_inconsistent_incoming_rejected(column, value):
    bank = bank_rows(("Поступление", "1,00", "A75"))
    bank.loc[0, column] = value
    with pytest.raises(ValueError):
        match_incoming_payments(bank, [75])


def test_no_existing_orders_and_empty_input():
    result = match_incoming_payments(bank_rows(("Поступление", "0,01", "A75")), [])
    assert summarize_unidentified_payments(result) == (1, Decimal("0.01"))
    empty_crm = pd.DataFrame(columns=["order_id", "status"])
    empty = reconcile_bank(empty_crm, bank_rows())
    assert empty.incoming_count == empty.unidentified_count == empty.identified_count == 0
    assert empty.unidentified_sum_rub == Decimal("0.00")


def test_missing_bank_column():
    with pytest.raises(ValueError, match="Отсутствуют поля"):
        match_incoming_payments(pd.DataFrame(), [75])


def test_actual_sources_accounting_repeatability_q1_and_integrity():
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DATA_DIR.iterdir() if p.is_file()}
    crm, bank = load_clean_crm(), load_bank()
    crm_before, bank_before = crm.copy(deep=True), bank.copy(deep=True)
    q1_before = summarize_q1(crm)
    result = reconcile_bank(crm, bank)
    repeat = reconcile_bank(load_clean_crm(), load_bank())
    assert_frame_equal(result.payments, repeat.payments)
    assert result.paid_orders_without_payment == repeat.paid_orders_without_payment
    assert result.unidentified_sum_rub == repeat.unidentified_sum_rub
    assert result.incoming_count == bank["Тип"].map(is_incoming).sum()
    assert result.incoming_count == result.identified_count + result.unidentified_count
    identified = result.payments["matched_order_ids"].map(bool)
    total = sum(result.payments["amount_rub"], Decimal("0.00"))
    assert total == sum(result.payments.loc[identified, "amount_rub"], Decimal("0.00")) + result.unidentified_sum_rub
    existing = set(crm["order_id"])
    assert all(set(ids) <= existing for ids in result.payments["matched_order_ids"])
    linked = {order for ids in result.payments["matched_order_ids"] for order in ids}
    assert set(result.paid_orders_without_payment) == paid_order_ids(crm) - linked
    assert list(result.paid_orders_without_payment) == sorted(set(result.paid_orders_without_payment))
    assert set(result.payments["Тип"]) == {"Поступление"}
    assert all(isinstance(amount, Decimal) for amount in result.payments["amount_rub"])
    assert_frame_equal(crm, crm_before)
    assert_frame_equal(bank, bank_before)
    assert summarize_q1(load_clean_crm()) == q1_before
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in DATA_DIR.iterdir() if p.is_file()}
