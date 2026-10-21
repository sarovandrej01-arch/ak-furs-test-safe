"""Единая команда расчёта Q1–Q4: python solve.py data/ > answers.json."""
from __future__ import annotations

import argparse
from decimal import Decimal, DecimalException
import json
from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile

from openpyxl.utils.exceptions import InvalidFileException

from src.bank import reconcile_bank
from src.chats import analyze_chats, violation_ids
from src.crm import clean_crm, summarize_q1
from src.loaders import load_answers_template, load_bank, load_catalog, load_chats, load_crm
from src.pricing import find_price_violations


FILES = (
    "01_crm_orders_export.csv", "02_bank_statement_rub.csv",
    "03_catalog_prices_reference.xlsx", "04_client_chats.zip", "answers_template.json",
)
FIELDS = {
    "q1_unique_orders": int,
    "q1_unique_order_lines": int,
    "q2a_paid_orders_without_payment": list,
    "q2b_unidentified_payments_count": int,
    "q2b_unidentified_payments_sum_rub": Decimal,
    "q3_price_violation_lines_total": int,
    "q3_price_violation_lines_by_manager": dict,
    "q4_chats_meeting_before_price": list,
    "run_command": str,
    "notes": str,
}
NOTES = (
    "Q1: при равной дате выгрузки сохраняется последняя запись исходного CSV; "
    "даты со слешами — ДД/ММ/ГГГГ, Excel serial — система 1900. "
    "Q2: заказ оплачен при хотя бы одной оплаченной строке; поступление определяется по полю «Тип». "
    "Номер счёта без явной ссылки на заказ не приравнивается к номеру заказа. "
    "Из нескольких явных ссылок учитываются все существующие заказы; операция считается один раз. "
    "Для связи достаточно одного поступления независимо от суммы и даты; списания её не отменяют. "
    "Q3: старые артикулы связываются с конечным кодом независимо от даты переименования; "
    "минимум выбирается на дату создания заказа по включительным границам периода, пустой конец открыт. "
    "Фиксированная скидка относится ко всей строке, процентная — к цене × количеству. "
    "Только отменённые строки исключаются; возвраты и строки в работе учитываются. "
    "Расчёт через Decimal, без округления до копеек перед сравнением; равенство минимуму не нарушение. "
    "Менеджеры сопоставляются по справочнику, даты трудоустройства не фильтруют продажи. "
    "Q4: учитываются первые события менеджера в исходном порядке; цены клиента и условия оплаты "
    "не считаются ценой изделия, описание шоурума не считается приглашением. "
    "Принимаются конкретные первоначальные и ориентировочные цены, в том числе прописью. "
    "Приглашение без цены — нарушение; оба события в одной реплике не задают строгого предшествования "
    "и помечаются для проверки. Составные правила работают без внешнего AI API и могут требовать "
    "расширения для новых формулировок."
)


def validate_answers(answers: dict, template: dict) -> None:
    """Проверить ключи и реальные типы ответа, отличая подсказки от схемы."""
    if set(template) != set(FIELDS) or set(answers) != set(FIELDS):
        raise ValueError("Ключи ответа или шаблона не соответствуют схеме Q1–Q4")
    for key, expected in FIELDS.items():
        # Ноль суммы в шаблоне — образец JSON number; текст в списке заказов — подсказка.
        template_type = (int, float) if expected is Decimal else (expected,)
        if type(template[key]) not in template_type:
            raise ValueError(f"Некорректный тип поля шаблона: {key}")
        value = answers[key]
        if type(value) is not expected:
            raise ValueError(f"Некорректный тип ответа: {key}, ожидался {expected.__name__}")
        if expected is int and value < 0:
            raise ValueError(f"Отрицательный счётчик: {key}")
    money = answers["q2b_unidentified_payments_sum_rub"]
    if not money.is_finite() or money < 0 or money != money.quantize(Decimal("0.01")):
        raise ValueError("Сумма должна быть конечной, неотрицательной и точной до копеек")
    orders = answers["q2a_paid_orders_without_payment"]
    if any(type(order) is not int or order <= 0 for order in orders):
        raise ValueError("Номера заказов должны быть положительными целыми числами")
    chats = answers["q4_chats_meeting_before_price"]
    if any(type(chat) is not str or not chat for chat in chats):
        raise ValueError("ID чатов должны быть непустыми строками")
    if orders != sorted(set(orders)) or chats != sorted(set(chats)):
        raise ValueError("Массивы ответа должны быть отсортированы и не содержать дублей")
    groups = answers["q3_price_violation_lines_by_manager"]
    if any(type(name) is not str or not name or type(count) is not int or count < 0
           for name, count in groups.items()):
        raise ValueError("Некорректная разбивка нарушений по менеджерам")
    if sum(groups.values()) != answers["q3_price_violation_lines_total"]:
        raise ValueError("Сумма нарушений по менеджерам не совпадает с итогом")
    if not answers["run_command"] or not answers["notes"]:
        raise ValueError("Команда запуска и notes должны быть заполнены")


def dumps_exact(value, level: int = 0) -> str:
    """Сериализовать JSON с числовым Decimal напрямую, без float и подстановок.

    Ключи и строки экранирует стандартный json. Поддержаны только типы
    итоговой схемы; неподдерживаемый тип вызывает ошибку до записи stdout.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Неконечный Decimal нельзя записать в JSON")
        return format(value, "f")
    if type(value) in (str, int, bool) or value is None:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if type(value) in (list, dict):
        if not value:
            return "[]" if isinstance(value, list) else "{}"
        indent, closing = "  " * (level + 1), "  " * level
        if isinstance(value, list):
            items = [dumps_exact(item, level + 1) for item in value]
            left, right = "[", "]"
        else:
            if any(type(key) is not str for key in value):
                raise TypeError("Ключи JSON должны быть строками")
            items = [json.dumps(key, ensure_ascii=False) + ": " + dumps_exact(item, level + 1)
                     for key, item in value.items()]
            left, right = "{", "}"
        return left + "\n" + ",\n".join(indent + item for item in items) + "\n" + closing + right
    raise TypeError(f"Неподдерживаемый тип JSON: {type(value).__name__}")


def solve(data_dir: str | Path) -> tuple[dict, tuple[str, ...]]:
    """Рассчитать ответы по произвольной папке; вернуть отдельные предупреждения."""
    folder = Path(data_dir)
    if not folder.is_dir():
        raise ValueError(f"Папка исходных данных не существует или не является каталогом: {folder}")
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise ValueError("Не найдены обязательные файлы: " + ", ".join(missing))
    template = load_answers_template(folder / "answers_template.json")
    raw_crm = load_crm(folder / "01_crm_orders_export.csv")
    bank = load_bank(folder / "02_bank_statement_rub.csv")
    sheets = load_catalog(folder / "03_catalog_prices_reference.xlsx")
    chats = load_chats(folder / "04_client_chats.zip")
    crm = clean_crm(raw_crm)
    q1 = summarize_q1(crm)
    q2 = reconcile_bank(crm, bank)
    q3 = find_price_violations(crm, sheets)
    if not q3.diagnostics.empty:
        kinds = ", ".join(f"{key}: {count}" for key, count in q3.diagnostics.issue.value_counts().items())
        raise ValueError(f"Q3 требует проверки данных ({kinds}); полный ответ не сформирован")
    if len(q3.checked) + q3.cancelled_count != len(crm):
        raise ValueError("Q3: не все строки учтены при проверке")
    if q2.incoming_count != q2.identified_count + q2.unidentified_count:
        raise ValueError("Q2: не сходится число поступлений")
    q4 = analyze_chats(chats)
    warnings = ()
    if flagged := sum(bool(result.review_flags) for result in q4):
        warnings = (f"Q4: чатов с неоднозначными формулировками: {flagged}; подробности: python -m src.chats --input <путь к ZIP>",)
    display_path = folder.as_posix()
    if not display_path.endswith("/"):
        display_path += "/"
    command = subprocess.list2cmdline(["python", "solve.py", display_path]) + " > answers.json"
    answers = {
        **q1,
        "q2a_paid_orders_without_payment": sorted(q2.paid_orders_without_payment),
        "q2b_unidentified_payments_count": q2.unidentified_count,
        "q2b_unidentified_payments_sum_rub": q2.unidentified_sum_rub,
        **q3.answers(),
        "q4_chats_meeting_before_price": violation_ids(q4),
        "run_command": command,
        "notes": NOTES,
    }
    validate_answers(answers, template)
    return answers, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Расчёт ответов AK Furs Q1–Q4 из сырых файлов")
    parser.add_argument("data_dir", help="Папка с пятью исходными файлами")
    args = parser.parse_args(argv)
    try:
        answers, warnings = solve(args.data_dir)
        payload = dumps_exact(answers) + "\n"
    except (OSError, ValueError, KeyError, TypeError, DecimalException, BadZipFile, InvalidFileException) as exc:
        print(f"Ошибка расчёта: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"Предупреждение: {warning}", file=sys.stderr)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    # UTF-8 без BOM нужен и при перенаправлении стандартного потока в Windows.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
