"""Сверка поступлений с очищенной CRM, без изменения исходных таблиц."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable

import pandas as pd

from .normalizers import normalize_order_id, normalize_status


NOTES = (
    "Поступления определяются по полю «Тип», а не по знаку суммы; неизвестный тип вызывает ошибку.",
    "Номер после A/А либо после «заказ», «заказы», «заказа», «заказу», «зак.» — ссылка на заказ.",
    "Голый номер счёта, дата, телефон или сумма не считаются ссылкой на заказ.",
    "Номера счетов без явной ссылки на заказ не приравниваются к номерам заказов даже при совпадении цифр.",
    "Несколько явных ссылок связываются со всеми существующими заказами; поступление учитывается один раз.",
    "Заказ считается оплаченным, если хотя бы одна его очищенная строка имеет статус «Оплачен».",
    "Сумма и дата поступления не ограничивают подтверждение связи; возвраты и списания её не отменяют.",
    "Повторные поступления сохраняются как отдельные операции, номера заказов в Q2a не повторяются.",
)


def is_incoming(operation_type: str) -> bool:
    """Определить направление по фактическому полю «Тип» банковской выгрузки."""
    key = str(operation_type).strip().casefold()
    if key == "поступление":
        return True
    if key == "списание":
        return False
    raise ValueError(f"Неизвестный тип банковской операции: {operation_type!r}")


def parse_rub_amount(value: str | Decimal) -> Decimal:
    """Разобрать рубли без float: пробелы тысяч, запятая или точка копеек.

    Принимаются только целые рубли или ровно две цифры копеек; валюта
    проверяется отдельно. Группировка тысяч должна быть корректной.
    """
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.quantize(Decimal("0.01")):
            raise ValueError(f"Некорректная сумма в рублях: {value!r}")
        return value.quantize(Decimal("0.01"))
    if not isinstance(value, str):
        raise ValueError("Сумма должна быть строкой или Decimal, без float")
    text = value.strip().replace("\u00a0", " ").replace("\u202f", " ")
    if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?: [0-9]{3})+)(?:[.,][0-9]{2})?", text):
        raise ValueError(f"Некорректная сумма в рублях: {value!r}")
    try:
        return Decimal(text.replace(" ", "").replace(",", ".")).quantize(Decimal("0.01"))
    except InvalidOperation as exc:
        raise ValueError(f"Некорректная сумма в рублях: {value!r}") from exc


# Границы запрещают частичное совпадение внутри ID, даты, суммы или счёта.
_END = r"(?![\w]|[.,:/-][0-9])"
_ID = r"(?:[aа]\s*-?\s*)?[0-9]+"
_PREFIXED = re.compile(r"(?<![\w-])[aа]\s*-?\s*[0-9]+" + _END, re.IGNORECASE)
_ORDER = re.compile(r"\b(?:заказ(?:а|у|ы|ов)?|зак\.)\s*(?:№|N|#)?\s*", re.IGNORECASE)
_FIRST = re.compile(_ID + _END, re.IGNORECASE)
_NEXT = re.compile(r"\s*(?:,\s+|;\s*|\bи\s+|\bили\s+|&\s*)(?:№\s*)?(" + _ID + r")" + _END, re.IGNORECASE)
_OTHER_LABEL = re.compile(r"(?:сч[её]т(?:а|у)?|договор(?:а|у)?|тел(?:ефон)?\.?)\s*(?:№|N|#)?\s*$", re.IGNORECASE)


def extract_order_ids(purpose: str) -> tuple[int, ...]:
    """Извлечь только явные ссылки, не проверяя их наличие в CRM.

    Голые числа допустимы после маркера заказа и в непосредственном списке
    после него. Номер счёта без маркера заказа не является ссылкой.
    """
    if not isinstance(purpose, str):
        raise ValueError("Назначение платежа должно быть строкой")
    candidates = []

    def collect(match):
        candidates.append(match[0])
        end = match.end()
        while continuation := _NEXT.match(purpose, end):
            candidates.append(continuation[1])
            end = continuation.end()

    for match in _PREFIXED.finditer(purpose):
        if not _OTHER_LABEL.search(purpose[:match.start()]):
            collect(match)
    for marker in _ORDER.finditer(purpose):
        match = _FIRST.match(purpose, marker.end())
        if match is None:
            continue
        collect(match)
    result = set()
    for candidate in candidates:
        try:
            result.add(normalize_order_id(candidate))
        except ValueError:
            # Ноль не является номером заказа, но не препятствует другим ссылкам.
            continue
    return tuple(sorted(result))


def match_incoming_payments(bank: pd.DataFrame, order_ids: Iterable[int]) -> pd.DataFrame:
    """Вернуть поступления с точной суммой и всеми явными/опознанными ссылками.

    Исходные столбцы сохранены. bank_row — позиция записи с 1 без заголовков;
    candidate_order_ids и matched_order_ids — отсортированные кортежи.
    """
    required = {"Тип", "Сумма", "Валюта", "Назначение платежа"}
    if missing := required - set(bank.columns):
        raise ValueError(f"Отсутствуют поля банка: {sorted(missing)}")
    existing = {normalize_order_id(order) for order in order_ids}
    incoming = bank["Тип"].map(is_incoming)
    result = bank.loc[incoming].copy()
    result["bank_row"] = [i for i, keep in enumerate(incoming, 1) if keep]
    amounts, candidates, matched = [], [], []
    for _, row in result.iterrows():
        if str(row["Валюта"]).strip().casefold() not in {"rub", "rur", "руб", "руб.", "₽"}:
            raise ValueError(f"Ожидались рубли в записи банка {row['bank_row']}")
        amount = parse_rub_amount(row["Сумма"])
        if amount < 0:
            raise ValueError(f"Отрицательная сумма поступления в записи банка {row['bank_row']}")
        ids = extract_order_ids(row["Назначение платежа"])
        amounts.append(amount)
        candidates.append(ids)
        matched.append(tuple(order for order in ids if order in existing))
    result["amount_rub"] = pd.Series(amounts, index=result.index, dtype=object)
    result["candidate_order_ids"] = pd.Series(candidates, index=result.index, dtype=object)
    result["matched_order_ids"] = pd.Series(matched, index=result.index, dtype=object)
    return result.reset_index(drop=True)


def paid_order_ids(crm: pd.DataFrame) -> set[int]:
    """Достаточно одной оплаченной строки; другие строки не снимают её статус."""
    if not {"order_id", "status"} <= set(crm.columns):
        raise ValueError("Требуется очищенная CRM с order_id и status")
    return {normalize_order_id(order) for order, status in zip(crm["order_id"], crm["status"])
            if normalize_status(status) == "Оплачен"}


def find_paid_orders_without_payment(crm: pd.DataFrame, payments: pd.DataFrame) -> list[int]:
    """Найти оплаченные заказы без единого связанного поступления."""
    linked = {order for ids in payments["matched_order_ids"] for order in ids}
    return sorted(paid_order_ids(crm) - linked)


def summarize_unidentified_payments(payments: pd.DataFrame) -> tuple[int, Decimal]:
    """Посчитать неопознанные операции, сохранив копейки через Decimal."""
    amounts = [amount for ids, amount in zip(payments["matched_order_ids"], payments["amount_rub"])
               if not ids]
    return len(amounts), sum(amounts, Decimal("0.00"))


@dataclass(frozen=True)
class Reconciliation:
    """Результаты Q2 и таблица поступлений для проверки связей."""
    payments: pd.DataFrame
    paid_orders_without_payment: tuple[int, ...]
    unidentified_count: int
    unidentified_sum_rub: Decimal
    operations_count: int

    @property
    def incoming_count(self) -> int:
        return len(self.payments)

    @property
    def identified_count(self) -> int:
        return self.incoming_count - self.unidentified_count


def reconcile_bank(crm: pd.DataFrame, bank: pd.DataFrame) -> Reconciliation:
    """Выполнить Q2 по явно переданным таблицам; файлы не читаются."""
    payments = match_incoming_payments(bank, crm["order_id"])
    count, amount = summarize_unidentified_payments(payments)
    return Reconciliation(payments, tuple(find_paid_orders_without_payment(crm, payments)),
                          count, amount, len(bank))


if __name__ == "__main__":
    import json
    from .crm import load_clean_crm
    from .loaders import load_bank

    report = reconcile_bank(load_clean_crm(), load_bank())
    # Точную сумму выводим JSON-строкой, чтобы не переводить Decimal во float.
    print(json.dumps({
        "q2a_paid_orders_without_payment": report.paid_orders_without_payment,
        "q2b_unidentified_payments_count": report.unidentified_count,
        "q2b_unidentified_payments_sum_rub": str(report.unidentified_sum_rub),
        "operations_count": report.operations_count,
        "incoming_count": report.incoming_count,
        "identified_count": report.identified_count,
        "notes": NOTES,
    }, ensure_ascii=False, indent=2))
