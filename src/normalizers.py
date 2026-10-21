"""Explicit, locale-independent normalization shared by analysis modules."""
from datetime import date, datetime
import math
from numbers import Real
import re

from openpyxl.utils.datetime import from_excel


def normalize_order_id(value: str | int) -> int:
    """Accept an optional Latin/Cyrillic A prefix and a positive integer only."""
    text = str(value).strip()
    match = re.fullmatch(r"(?:[aа]\s*-?\s*)?([0-9]+)", text, re.IGNORECASE)
    if match is None or int(match[1]) <= 0:
        raise ValueError(f"Invalid order identifier: {value!r}")
    return int(match[1])


def normalize_line_id(value: str | int) -> int:
    """Line IDs are positive decimal integers; leading zeros are insignificant."""
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text) or int(text) <= 0:
        raise ValueError(f"Invalid line identifier: {value!r}")
    return int(text)


_MONTHS = dict(zip(
    ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"),
    range(1, 13),
))


def normalize_datetime(value: str | Real | date) -> datetime:
    """Parse ISO, day-first dates, Russian abbreviated months or Excel 1900 dates.

    Missing/invalid dates and timezone-aware datetimes fail explicitly. Slash
    dates are always DD/MM/YYYY. Excel fractions preserve time of day.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise ValueError("Timezone-aware dates are not supported")
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip().casefold()
    if not isinstance(value, bool) and re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", text):
        serial = float(text.replace(",", "."))
        if not math.isfinite(serial) or serial < 1:
            raise ValueError(f"Invalid Excel date: {value!r}")
        try:
            result = from_excel(serial)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"Invalid Excel date: {value!r}") from exc
        if isinstance(result, datetime):
            return result
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        for suffix in ("", " %H:%M", " %H:%M:%S", "T%H:%M", "T%H:%M:%S"):
            try:
                return datetime.strptime(text, (fmt + suffix).replace("T", "t"))
            except ValueError:
                pass
    match = re.fullmatch(r"([0-9]{1,2})\s+([а-я]+)\s+([0-9]{4})", text)
    if match and match[2] in _MONTHS:
        return datetime(int(match[3]), _MONTHS[match[2]], int(match[1]))
    raise ValueError(f"Invalid date: {value!r}")


def normalize_status(value: object) -> str:
    """Map known spellings to four categories; unknown values are in progress."""
    key = str(value).strip().casefold().replace("ё", "е")
    if key in {"оплачен", "оплачено", "paid"}:
        return "Оплачен"
    if key in {"отменен", "отменено", "cancelled", "canceled"}:
        return "Отменён"
    if key in {"возврат", "refund", "refunded"}:
        return "Возврат"
    return "В работе"
