"""Shared schema constants and validation helpers."""

from __future__ import annotations

from typing import Any

CATEGORIES = (
    "政策与优惠",
    "产品事实与参数",
    "能力越界",
    "无依据事实",
    "安全误导",
    "关键信息遗漏",
)
SEVERITIES = ("无", "低", "中", "高", "严重")
REQUIRED_INPUT_FIELDS = ("id", "user_question", "system_reply", "knowledge_base")
REQUIRED_PREDICTION_FIELDS = (
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


class DetectorError(Exception):
    """Base error raised by the detector."""


class InputValidationError(DetectorError):
    """Input data is missing fields or contains evaluation labels."""


class LLMError(DetectorError):
    """The configured LLM endpoint could not produce a valid result."""


def validate_result(
    result: dict[str, Any],
    expected_id: str | None = None,
    expected_question: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one detector result without trusting model output."""
    if not isinstance(result, dict):
        raise LLMError("LLM response must be a JSON object")

    is_hallucination = result.get("is_hallucination")
    if not isinstance(is_hallucination, bool):
        raise LLMError("is_hallucination must be boolean")

    category = result.get("category")
    if is_hallucination and category not in CATEGORIES:
        raise LLMError("category is not one of the six supported categories")
    if not is_hallucination:
        category = None

    severity = result.get("severity", "中" if is_hallucination else "无")
    if severity not in SEVERITIES:
        raise LLMError("severity is invalid")
    if not is_hallucination:
        severity = "无"

    confidence = result.get("confidence", 0.5)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise LLMError("confidence must be numeric")
    confidence = round(max(0.0, min(1.0, float(confidence))), 4)

    def as_text(value: Any) -> str:
        if isinstance(value, list):
            return "；".join(str(part) for part in value if part is not None)
        return str(value or "")

    normalized = {
        "id": str(expected_id if expected_id is not None else result.get("id", "")),
        "user_question": str(expected_question if expected_question is not None else result.get("user_question", "")),
        "is_hallucination": is_hallucination,
        "category": category,
        "severity": severity,
        "confidence": confidence,
        "claims": as_text(result.get("claims")),
        "evidence": as_text(result.get("evidence")),
        "reason": as_text(result.get("reason")),
        "detection_mode": str(result.get("detection_mode") or "llm"),
    }
    if not normalized["reason"]:
        raise LLMError("reason must not be empty")
    return normalized


def validate_prediction(result: Any) -> dict[str, Any]:
    """Strictly validate a stored prediction before evaluation or export."""
    if not isinstance(result, dict):
        raise InputValidationError("Every prediction must be a JSON object")
    missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in result]
    if missing:
        raise InputValidationError(f"Prediction missing fields: {', '.join(missing)}")
    if not isinstance(result["id"], str) or not result["id"]:
        raise InputValidationError("Prediction id must be a non-empty string")
    if not isinstance(result["user_question"], str):
        raise InputValidationError("Prediction user_question must be a string")
    decision = result["is_hallucination"]
    if not isinstance(decision, bool):
        raise InputValidationError("Prediction is_hallucination must be boolean")
    category = result["category"]
    if decision and category not in CATEGORIES:
        raise InputValidationError("Hallucination prediction category is invalid")
    if not decision and category is not None:
        raise InputValidationError("Normal prediction category must be null")
    if result["severity"] not in SEVERITIES:
        raise InputValidationError("Prediction severity is invalid")
    if not decision and result["severity"] != "无":
        raise InputValidationError("Normal prediction severity must be 无")
    if decision and result["severity"] == "无":
        raise InputValidationError("Hallucination prediction severity must not be 无")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise InputValidationError("Prediction confidence must be a number between 0 and 1")
    for field in ("claims", "evidence", "reason", "detection_mode"):
        if not isinstance(result[field], str):
            raise InputValidationError(f"Prediction {field} must be a string")
    if not result["reason"]:
        raise InputValidationError("Prediction reason must not be empty")
    return result
