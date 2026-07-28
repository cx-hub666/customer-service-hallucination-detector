"""Safe CSV serialization helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

CSV_FIELDS = (
    "id",
    "user_question",
    "is_hallucination",
    "category",
    "severity",
    "confidence",
    "claims",
    "evidence",
    "reason",
    "detection_mode",
)


def safe_csv_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_predictions_csv(handle: TextIO, predictions: Iterable[dict[str, Any]]) -> None:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows({field: safe_csv_value(row.get(field)) for field in CSV_FIELDS} for row in predictions)


def write_predictions_csv_file(path: Path, predictions: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        write_predictions_csv(handle, predictions)
