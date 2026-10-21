"""Проверка исторического минимума цены; расчёты Q3 без изменения источников."""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from numbers import Integral
import re

import pandas as pd

from .normalizers import normalize_datetime, normalize_status


NOTES = (
    "Старые и новые артикулы объединяются до конечного кода независимо от даты переименования.",
    "Минимум выбирается на дату создания заказа по включительному интервалу с открытым концом при пустой дате.",
    "Пересекающиеся периоды, циклы переименований и неоднозначные псевдонимы вызывают ошибку справочника.",
    "Фиксированная скидка относится ко всей строке; процентная — к цене, умноженной на количество.",
    "До сравнения цена не округляется до копеек; проверяется также эквивалентное неравенство сумм строки.",
    "Только отменённые строки исключаются; возвраты и строки в работе участвуют.",
    "Неизвестные артикулы, отсутствующий прайс и некорректные значения выводятся в диагностике, а не считаются отсутствием нарушения.",
    "Неизвестный менеджер остаётся отдельной явно помеченной группой; даты трудоустройства не исключают строки.",
)


def parse_decimal(value) -> Decimal:
    """Прочитать число, не выполняя денежных операций через float.

    Поддержаны пробелы тысяч, десятичные запятая/точка, смешанная запись
    17,000.00 / 17.000,00, суффиксы «млн» и «тыс». Одиночный разделитель
    с тремя цифрами справа неоднозначен и не принимается.
    """
    if isinstance(value, bool):
        raise ValueError("Логическое значение не является числом")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Требуется конечное число")
        return value
    if isinstance(value, Integral):
        return Decimal(int(value))
    if not isinstance(value, str):
        raise ValueError("Требуется строка, целое число или Decimal, без float")
    text = value.strip().casefold().replace("\u00a0", " ").replace("\u202f", " ")
    multiplier = Decimal(1)
    suffix = re.search(r"\s*(млн|тыс)\.?$", text)
    if suffix:
        multiplier = Decimal(1000000 if suffix[1] == "млн" else 1000)
        text = text[:suffix.start()].strip()
    # При двух знаках последний из них — десятичный; другой группирует тысячи.
    if "," in text and "." in text:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        group_sep = "." if decimal_sep == "," else ","
        pattern = r"[+-]?[0-9]{1,3}(?:" + re.escape(group_sep) + r"[0-9]{3})+" + re.escape(decimal_sep) + r"[0-9]{1,2}"
        if not re.fullmatch(pattern, text):
            raise ValueError(f"Некорректное число: {value!r}")
        text = text.replace(group_sep, "").replace(decimal_sep, ".")
    else:
        if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?: [0-9]{3})+)(?:[.,][0-9]{1,2})?", text):
            raise ValueError(f"Некорректное или неоднозначное число: {value!r}")
        text = text.replace(" ", "").replace(",", ".")
    with localcontext() as context:
        context.prec = 50
        return Decimal(text) * multiplier


def normalize_sku(value: str) -> str:
    """Убрать внешние пробелы и регистр, не склеивая разные артикулы."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Пустой или некорректный артикул")
    return value.strip().upper()


def build_sku_map(catalog: pd.DataFrame, renames: pd.DataFrame) -> dict[str, str]:
    """Разрешить все цепочки; проверить циклы, ветвления и конечные коды."""
    catalog_codes = [normalize_sku(value) for value in catalog["Артикул"]]
    if len(catalog_codes) != len(set(catalog_codes)):
        raise ValueError("Неоднозначные артикулы каталога")
    edges = {}
    for _, row in renames.iterrows():
        old, new = normalize_sku(row["Старый артикул"]), normalize_sku(row["Новый артикул"])
        normalize_datetime(row["Дата перехода"])
        if old in edges and edges[old] != new:
            raise ValueError(f"Неоднозначное переименование: {old}")
        edges[old] = new
    result = {}
    for code in set(catalog_codes) | set(edges) | set(edges.values()):
        trail, current = set(), code
        while current in edges:
            if current in trail:
                raise ValueError(f"Цикл переименований: {code}")
            trail.add(current)
            current = edges[current]
        if current not in catalog_codes:
            raise ValueError(f"Конечный артикул отсутствует в каталоге: {current}")
        result[code] = current
    return result


@dataclass(frozen=True)
class PricePeriod:
    """Обе границы включительны, None означает отсутствие верхней границы."""
    start: date
    end: date | None
    minimum: Decimal


def build_price_history(history: pd.DataFrame, sku_map: dict[str, str]) -> dict[str, tuple[PricePeriod, ...]]:
    """Подготовить интервалы, не допуская неоднозначных пересечений."""
    result = {}
    for _, row in history.iterrows():
        code = normalize_sku(row["Артикул"])
        if code not in sku_map:
            raise ValueError(f"Артикул прайса отсутствует в каталоге: {code}")
        if pd.isna(row["Действует с"]) or row["Действует с"] == "":
            raise ValueError(f"Отсутствует начало периода прайса: {code}")
        start = normalize_datetime(row["Действует с"]).date()
        raw_end = row["Действует по"]
        end = None if pd.isna(raw_end) or raw_end == "" else normalize_datetime(raw_end).date()
        minimum = parse_decimal(row["Минимально допустимая цена, ₽"])
        if (end is not None and end < start) or minimum < 0:
            raise ValueError(f"Некорректный период/минимум прайса: {code}")
        result.setdefault(sku_map[code], []).append(PricePeriod(start, end, minimum))
    for code, periods in result.items():
        periods.sort(key=lambda p: p.start)
        for previous, current in zip(periods, periods[1:]):
            if previous.end is None or current.start <= previous.end:
                raise ValueError(f"Пересекающиеся периоды прайса: {code}")
        result[code] = tuple(periods)
    return result


def historical_price(code: str, created_at, history: dict[str, tuple[PricePeriod, ...]]) -> PricePeriod | None:
    """Последняя применимая запись именно на день создания, с проверкой конца."""
    if pd.isna(created_at):
        raise ValueError("Отсутствует дата создания заказа")
    day = normalize_datetime(created_at).date()
    for period in reversed(history.get(code, ())):
        if period.start <= day:
            return period if period.end is None or day <= period.end else None
    return None


def calculate_discount(unit_price: Decimal, quantity: Decimal, discount) -> Decimal:
    """Вернуть скидку на всю строку, проверив её диапазон."""
    if unit_price < 0 or quantity <= 0:
        raise ValueError("Цена должна быть неотрицательной, количество — положительным")
    with localcontext() as context:
        context.prec = 50
        gross = unit_price * quantity
        if discount is None or (isinstance(discount, str) and not discount.strip()):
            return Decimal(0)
        if isinstance(discount, str) and discount.strip().endswith("%"):
            percent = parse_decimal(discount.strip()[:-1])
            if not Decimal(0) <= percent <= Decimal(100):
                raise ValueError("Процент скидки вне диапазона 0–100")
            amount = gross * percent / Decimal(100)
        else:
            amount = parse_decimal(discount)
        if not Decimal(0) <= amount <= gross:
            raise ValueError("Скидка отрицательна или превышает сумму строки")
        return amount


def effective_unit_price(unit_price: Decimal, quantity: Decimal, discount) -> Decimal:
    """Цена после скидки без округления до копеек."""
    amount = calculate_discount(unit_price, quantity, discount)
    with localcontext() as context:
        context.prec = 50
        return (unit_price * quantity - amount) / quantity


def _manager_key(value: str) -> str:
    return " ".join(value.split()).casefold().replace("ё", "е")


def build_manager_map(managers: pd.DataFrame) -> dict[str, str]:
    """Сопоставить канонические имена и перечисленные варианты без угадывания."""
    result = {}
    for _, row in managers.iterrows():
        canonical = str(row["Каноническое имя"]).strip()
        if not canonical or pd.isna(row["Каноническое имя"]):
            raise ValueError("Пустое каноническое имя менеджера")
        variants = row["Варианты написания в CRM"]
        aliases = [] if pd.isna(variants) else str(variants).split("|")
        for alias in [canonical, *aliases]:
            key = _manager_key(alias)
            if not key:
                continue
            if key in result and result[key] != canonical:
                raise ValueError(f"Неоднозначный вариант имени менеджера: {alias}")
            result[key] = canonical
    return result


def normalize_manager(value, manager_map: dict[str, str]) -> str | None:
    """Вернуть None для неизвестного имени, чтобы вызвать явную диагностику."""
    return manager_map.get(_manager_key(value)) if isinstance(value, str) else None


def group_violations_by_manager(checked: pd.DataFrame, manager_names=()) -> dict[str, int]:
    """Суммировать только нарушающие строки, включая явную неизвестную группу."""
    counts = checked.loc[checked["is_violation"], "canonical_manager"].value_counts()
    return {name: int(counts.get(name, 0)) for name in sorted(set(counts.index) | set(manager_names))}


@dataclass(frozen=True)
class PricingReport:
    """Проверенные строки и диагностические случаи доступны локально."""
    checked: pd.DataFrame
    diagnostics: pd.DataFrame
    cancelled_count: int
    eligible_count: int
    manager_names: tuple[str, ...]

    @property
    def violations(self) -> pd.DataFrame:
        return self.checked.loc[self.checked["is_violation"]].copy()

    @property
    def missing_price_count(self) -> int:
        return int(self.diagnostics["issue"].isin(["unknown_sku", "missing_price"]).sum())

    def answers(self) -> dict:
        return {"q3_price_violation_lines_total": len(self.violations),
                "q3_price_violation_lines_by_manager": group_violations_by_manager(self.checked, self.manager_names)}


def find_price_violations(crm: pd.DataFrame, sheets: dict[str, pd.DataFrame]) -> PricingReport:
    """Проверить очищенные строки; проблемы данных вернуть явно, без догадок."""
    if crm.duplicated(["order_id", "line_id"]).any():
        raise ValueError("Ожидалась очищенная CRM без повторяющихся ключей")
    sku_map = build_sku_map(sheets["Каталог"], sheets["Переименования"])
    history = build_price_history(sheets["Прайс_история"], sku_map)
    managers = build_manager_map(sheets["Менеджеры"])
    cancelled = crm["status"].map(normalize_status).eq("Отменён")
    checked, diagnostics = [], []
    for _, row in crm.loc[~cancelled].iterrows():
        base = {"order_id": row["order_id"], "line_id": row["line_id"]}
        manager = normalize_manager(row["Менеджер"], managers)
        if manager is None:
            diagnostics.append({**base, "issue": "unknown_manager", "detail": str(row["Менеджер"])})
            manager = f"Неизвестный менеджер: {row['Менеджер']}"
        try:
            source_sku = normalize_sku(row["Артикул"])
            if source_sku not in sku_map:
                diagnostics.append({**base, "issue": "unknown_sku", "detail": source_sku})
                continue
            code = sku_map[source_sku]
            period = historical_price(code, row["created_at"], history)
            if period is None:
                diagnostics.append({**base, "issue": "missing_price", "detail": code})
                continue
            price, quantity = parse_decimal(row["Цена за ед."]), parse_decimal(row["Кол-во"])
            amount = calculate_discount(price, quantity, row["Скидка"])
            effective = effective_unit_price(price, quantity, row["Скидка"])
            with localcontext() as context:
                context.prec = 50
                net = price * quantity - amount
                violation = net < period.minimum * quantity
            checked.append({**row.to_dict(), "canonical_sku": code, "canonical_manager": manager,
                            "unit_price": price, "quantity": quantity, "line_discount": amount,
                            "net_line_total": net, "effective_unit_price": effective,
                            "minimum_price": period.minimum, "price_from": period.start,
                            "price_to": period.end, "is_violation": violation})
        except (ValueError, TypeError) as exc:
            diagnostics.append({**base, "issue": "invalid_value", "detail": str(exc)})
    extra = ["canonical_sku", "canonical_manager", "unit_price", "quantity", "line_discount",
             "net_line_total", "effective_unit_price", "minimum_price", "price_from", "price_to", "is_violation"]
    frame = pd.DataFrame(checked, columns=list(crm.columns) + extra)
    frame["is_violation"] = frame["is_violation"].astype(bool)
    return PricingReport(frame, pd.DataFrame(diagnostics, columns=["order_id", "line_id", "issue", "detail"]),
                         int(cancelled.sum()), int((~cancelled).sum()), tuple(sorted(set(managers.values()))))


if __name__ == "__main__":
    import json
    from .crm import load_clean_crm
    from .loaders import load_catalog

    report = find_price_violations(load_clean_crm(), load_catalog())
    print(json.dumps({**report.answers(), "checked_count": len(report.checked),
                      "cancelled_count": report.cancelled_count,
                      "missing_price_count": report.missing_price_count,
                      "diagnostics": report.diagnostics.to_dict("records"),
                      "notes": NOTES}, ensure_ascii=False, indent=2))
