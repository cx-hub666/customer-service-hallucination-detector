"""Flask dashboard for running and inspecting hallucination detection."""

from __future__ import annotations

import io
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from hallucination_detector.csv_utils import write_predictions_csv
from hallucination_detector.data_io import load_detection_input, read_json, write_json
from hallucination_detector.detector import HallucinationDetector
from hallucination_detector.evaluator import evaluate_predictions
from hallucination_detector.models import DetectorError, InputValidationError, LLMError

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "replies.json"
GROUND_TRUTH_PATH = BASE_DIR / "data" / "ground_truth.json"
OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
RUNS_DIR = OUTPUT_DIR / "runs"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
RUN_LOCK = threading.RLock()

app = Flask(__name__)


def _mode() -> str:
    if request.mimetype != "application/json":
        raise InputValidationError("Content-Type must be application/json")
    try:
        payload = request.get_json(silent=False)
    except BadRequest as exc:
        raise InputValidationError("Request body must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise InputValidationError("Request JSON must be an object")
    mode = payload.get("mode")
    if mode not in {"offline", "llm", "hybrid"}:
        raise InputValidationError("mode must be offline, llm or hybrid")
    return mode


def _predict(mode: str) -> list[dict[str, Any]]:
    # Detection has no reference to GROUND_TRUTH_PATH by design.
    replies = load_detection_input(DATA_PATH)
    return HallucinationDetector(mode).detect(replies)


def _evaluate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ground_truth = read_json(GROUND_TRUTH_PATH)
    return evaluate_predictions(predictions, ground_truth)


def _store_snapshot(predictions: list[dict[str, Any]], metrics: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    write_json(run_dir / "predictions.json", predictions)
    if metrics is not None:
        write_json(run_dir / "metrics.json", metrics)

    # Canonical artifacts remain convenient for CLI users; the manifest is
    # committed last and is the only source used by results/export endpoints.
    write_json(PREDICTIONS_PATH, predictions)
    if metrics is None:
        METRICS_PATH.unlink(missing_ok=True)
    else:
        write_json(METRICS_PATH, metrics)
    manifest = {"run_id": run_id, "mode": mode, "metrics_available": metrics is not None}
    write_json(MANIFEST_PATH, manifest)
    return {"run_id": run_id, "mode": mode, "predictions": predictions, "metrics": metrics}


def _read_snapshot() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict) or not re.fullmatch(r"[0-9a-f]{32}", str(manifest.get("run_id", ""))):
        raise InputValidationError("Stored run manifest is invalid")
    run_id = manifest["run_id"]
    run_dir = RUNS_DIR / run_id
    predictions = read_json(run_dir / "predictions.json")
    metrics = read_json(run_dir / "metrics.json") if manifest.get("metrics_available") is True else None
    return {"run_id": run_id, "mode": manifest.get("mode"), "predictions": predictions, "metrics": metrics}


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/detect")
def detect_api() -> Response:
    mode = _mode()
    with RUN_LOCK:
        snapshot = _store_snapshot(_predict(mode), None, mode)
    return jsonify(snapshot)


@app.post("/api/evaluate")
def evaluate_api() -> Response:
    with RUN_LOCK:
        current = _read_snapshot()
        if current is None:
            return jsonify({"error": "请先运行检测"}), 404
        metrics = _evaluate(current["predictions"])
        snapshot = _store_snapshot(current["predictions"], metrics, str(current.get("mode") or "offline"))
    return jsonify(snapshot)


@app.post("/api/run")
def run_api() -> Response:
    mode = _mode()
    with RUN_LOCK:
        predictions = _predict(mode)
        snapshot = _store_snapshot(predictions, _evaluate(predictions), mode)
    return jsonify(snapshot)


@app.get("/api/results")
def results_api() -> Response:
    with RUN_LOCK:
        snapshot = _read_snapshot()
    return jsonify(snapshot or {"run_id": None, "mode": None, "predictions": [], "metrics": None})


@app.get("/api/export/<file_type>")
def export_api(file_type: str) -> Response:
    with RUN_LOCK:
        snapshot = _read_snapshot()
    if snapshot is None:
        return jsonify({"error": "暂无可导出的检测结果"}), 404
    predictions = snapshot["predictions"]
    if file_type == "json":
        content = json.dumps(predictions, ensure_ascii=False, indent=2) + "\n"
        return Response(
            content,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=predictions.json"},
        )
    if file_type == "csv":
        stream = io.StringIO()
        write_predictions_csv(stream, predictions)
        return Response(
            "\ufeff" + stream.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    return jsonify({"error": "file_type must be json or csv"}), 400


@app.errorhandler(DetectorError)
@app.errorhandler(ValueError)
@app.errorhandler(json.JSONDecodeError)
def handle_known_error(error: Exception) -> tuple[Response, int]:
    message = "LLM 服务暂不可用或返回了无效响应" if isinstance(error, LLMError) else str(error)
    return jsonify({"error": message}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
