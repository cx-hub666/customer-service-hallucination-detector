"""JSON I/O with an explicit boundary between detection and evaluation data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import InputValidationError, REQUIRED_INPUT_FIELDS

# These names are evaluation-only. Their presence anywhere in detector input is
# rejected so an upstream join cannot silently leak labels into predictions.
FORBIDDEN_DETECTION_KEYS = {
    "ground_truth",
    "is_hallucination",
    "hallucination_type",
    "label",
    "expected",
    "gold",
    "target",
}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(target)


def _find_forbidden_keys(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_DETECTION_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def validate_detection_input(data: Any) -> list[dict[str, str]]:
    if not isinstance(data, list):
        raise InputValidationError("Detection input must be a JSON array")

    leaked = _find_forbidden_keys(data)
    if leaked:
        locations = ", ".join(leaked[:5])
        raise InputValidationError(f"Evaluation label leakage detected at: {locations}")

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputValidationError(f"Item {index} must be an object")
        missing = [field for field in REQUIRED_INPUT_FIELDS if field not in item]
        if missing:
            raise InputValidationError(f"Item {index} missing fields: {', '.join(missing)}")
        normalized_item = {field: str(item[field]) for field in REQUIRED_INPUT_FIELDS}
        if not normalized_item["id"]:
            raise InputValidationError(f"Item {index} has an empty id")
        if normalized_item["id"] in seen_ids:
            raise InputValidationError(f"Duplicate id: {normalized_item['id']}")
        seen_ids.add(normalized_item["id"])
        normalized.append(normalized_item)
    return normalized


def load_detection_input(path: str | Path) -> list[dict[str, str]]:
    """The only loader used by detection; it intentionally has no label argument."""
    return validate_detection_input(read_json(path))


def ensure_unique_ids(items: Iterable[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            raise InputValidationError(f"Every {name} item must contain id")
        item_id = str(item["id"])
        if item_id in indexed:
            raise InputValidationError(f"Duplicate {name} id: {item_id}")
        indexed[item_id] = item
    return indexed
