"""CRM cleaning for Q1; raw source columns remain available to Q2/Q3."""
from pathlib import Path

import pandas as pd

from .loaders import DATA_DIR, load_crm
from .normalizers import (
    normalize_datetime, normalize_line_id, normalize_order_id, normalize_status,
)

NOTES = (
    "При равных датах выгрузки сохраняется последняя запись в порядке исходного CSV.",
    "Даты со слешами трактуются как ДД/ММ/ГГГГ; Excel serial — система 1900.",
    "Номера строк — положительные целые; ведущие нули незначимы.",
    "Числовой номер заказа без префикса A также допустим; ноль и некорректные ID запрещены.",
    "Неизвестные статусы относятся к «В работе»; статусы не исключают заказ из Q1.",
    "Неверные ID и даты вызывают ошибку, строки не отбрасываются молча.",
)


def clean_crm(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one latest snapshot per (order_id, line_id), sorted by that key.

    Original columns are unchanged. Added columns: order_id, line_id,
    created_at, exported_at, status, source_row (1-based CSV data record).
    Repeated cleaning and repeated loading produce identical results.
    """
    result = raw.copy(deep=True)
    specs = {
        "order_id": ("Номер заказа", normalize_order_id),
        "line_id": ("№ строки", normalize_line_id),
        "created_at": ("Дата создания", normalize_datetime),
        "exported_at": ("Выгружено", normalize_datetime),
        "status": ("Статус", normalize_status),
    }
    for target, (source, normalizer) in specs.items():
        if source not in result:
            raise ValueError(f"Missing required CRM column: {source}")
        values = []
        for position, value in enumerate(result[source], 1):
            try:
                values.append(normalizer(value))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CRM record {position}, field {source}: {exc}") from exc
        result[target] = pd.Series(values, index=result.index,
                                   dtype="int64" if target in {"order_id", "line_id"} else None)
    if "source_row" not in result:
        result["source_row"] = range(1, len(result) + 1)
    # Stable sorting makes equal export timestamps resolve to the last input row.
    result = result.sort_values("exported_at", kind="stable")
    result = result.drop_duplicates(["order_id", "line_id"], keep="last")
    return result.sort_values(["order_id", "line_id"]).reset_index(drop=True)


def load_clean_crm(path: str | Path = DATA_DIR / "01_crm_orders_export.csv") -> pd.DataFrame:
    return clean_crm(load_crm(path))


def summarize_q1(cleaned: pd.DataFrame) -> dict[str, int]:
    """Count normalized orders and line keys, with no status filtering."""
    return {
        "q1_unique_orders": int(cleaned["order_id"].nunique()),
        "q1_unique_order_lines": len(cleaned[["order_id", "line_id"]].drop_duplicates()),
    }


if __name__ == "__main__":
    import json
    raw = load_crm()
    cleaned = clean_crm(raw)
    print(json.dumps({**summarize_q1(cleaned), "rows_before": len(raw),
                      "rows_after": len(cleaned), "notes": NOTES}, ensure_ascii=False, indent=2))
