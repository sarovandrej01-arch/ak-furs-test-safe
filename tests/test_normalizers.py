from datetime import datetime

import pytest

from src.normalizers import (
    normalize_datetime, normalize_line_id, normalize_order_id, normalize_status,
)


@pytest.mark.parametrize("value", ["A-00075", "А-75", "a-75", "A75", " а - 00075 ", "\tA\u00a0- 75 ", "75", 75])
def test_order_id(value):
    assert normalize_order_id(value) == 75
    assert isinstance(normalize_order_id(value), int)


@pytest.mark.parametrize("value", ["B75", "order 75", "A75tail", "A-7 5", "A--75", "-75", "A-0", "", None, True, "75.0", "A７５"])
def test_invalid_order_id(value):
    with pytest.raises(ValueError):
        normalize_order_id(value)


@pytest.mark.parametrize("value", ["1", "01", " 1 ", 1])
def test_line_id(value):
    assert normalize_line_id(value) == 1


@pytest.mark.parametrize("value", ["1.0", "0", "-1", "строка 1", None])
def test_invalid_line_id(value):
    with pytest.raises(ValueError):
        normalize_line_id(value)


@pytest.mark.parametrize("value,expected", [
    ("2025-04-05", datetime(2025, 4, 5)),
    ("05.04.2025", datetime(2025, 4, 5)),
    ("05/04/2025", datetime(2025, 4, 5)),
    ("5 апр 2025", datetime(2025, 4, 5)),
    ("23 СЕН 2026", datetime(2026, 9, 23)),
    ("2026-10-01 09:05", datetime(2026, 10, 1, 9, 5)),
    ("2026-10-01T09:05:30", datetime(2026, 10, 1, 9, 5, 30)),
    ("01.10.2026 09:05", datetime(2026, 10, 1, 9, 5)),
    ("45658", datetime(2025, 1, 1)),
    (45658.5, datetime(2025, 1, 1, 12)),
    ("45658,5", datetime(2025, 1, 1, 12)),
    (datetime(2025, 1, 1), datetime(2025, 1, 1)),
])
def test_dates(value, expected):
    assert normalize_datetime(value) == expected


@pytest.mark.parametrize("value", ["", None, "not a date", "31/02/2025", "2025-13-01", "0", "-1", "999999999", True])
def test_invalid_dates(value):
    with pytest.raises(ValueError):
        normalize_datetime(value)


@pytest.mark.parametrize("value,expected", [
    ("Оплачен ", "Оплачен"), (" PAID ", "Оплачен"), ("оплачено", "Оплачен"),
    ("отменен", "Отменён"), ("ОТМЕНЁН", "Отменён"), ("CANCELLED", "Отменён"),
    ("REFUND", "Возврат"), ("возврат", "Возврат"),
    ("NEW", "В работе"), ("Ждём оплату", "В работе"),
    ("В работе", "В работе"), ("unknown", "В работе"), (None, "В работе"),
])
def test_status(value, expected):
    assert normalize_status(value) == expected
