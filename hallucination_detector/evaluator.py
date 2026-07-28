"""Evaluation-only code; this module is the ground-truth access boundary."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .data_io import ensure_unique_ids, read_json
from .models import InputValidationError, validate_prediction


def _validate_truth(item: Any) -> None:
    if not isinstance(item, dict):
        raise InputValidationError("Every ground-truth item must be a JSON object")
    if "is_hallucination" not in item:
        raise InputValidationError("Ground-truth item missing is_hallucination")
    if not isinstance(item["is_hallucination"], bool):
        raise InputValidationError("Ground-truth is_hallucination must be boolean")


def evaluate_predictions(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(predictions, list) or not isinstance(ground_truth, list):
        raise InputValidationError("Predictions and ground truth must both be JSON arrays")
    for item in predictions:
        validate_prediction(item)
    for item in ground_truth:
        _validate_truth(item)
    pred_by_id = ensure_unique_ids(predictions, "prediction")
    truth_by_id = ensure_unique_ids(ground_truth, "ground-truth")
    if set(pred_by_id) != set(truth_by_id):
        missing = sorted(set(truth_by_id) - set(pred_by_id))
        extra = sorted(set(pred_by_id) - set(truth_by_id))
        raise InputValidationError(f"Prediction/ground-truth ids differ; missing={missing}, extra={extra}")

    tp = tn = fp = fn = 0
    false_positives: list[str] = []
    false_negatives: list[str] = []
    disagreements: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    for item in predictions:
        item_id = str(item["id"])
        predicted = item["is_hallucination"]
        expected = truth_by_id[item_id]["is_hallucination"]
        if predicted:
            category_counts[str(item.get("category") or "未分类")] += 1
        if predicted and expected:
            tp += 1
        elif not predicted and not expected:
            tn += 1
        elif predicted:
            fp += 1
            false_positives.append(item_id)
        else:
            fn += 1
            false_negatives.append(item_id)
        if predicted != expected:
            disagreements.append(
                {
                    "id": item_id,
                    "predicted": predicted,
                    "expected": expected,
                    "reason": item.get("reason", ""),
                    "ground_truth_detail": truth_by_id[item_id].get("detail", ""),
                }
            )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0
    return {
        "total": len(predictions),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "category_distribution": dict(category_counts),
        "disagreements": disagreements,
    }


def evaluate_files(predictions_path: str | Path, ground_truth_path: str | Path) -> dict[str, Any]:
    predictions = read_json(predictions_path)
    ground_truth = read_json(ground_truth_path)
    if not isinstance(predictions, list) or not isinstance(ground_truth, list):
        raise InputValidationError("Predictions and ground truth must both be JSON arrays")
    return evaluate_predictions(predictions, ground_truth)
