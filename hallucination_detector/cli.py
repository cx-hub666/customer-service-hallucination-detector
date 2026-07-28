"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data_io import load_detection_input, read_json, write_json
from .csv_utils import write_predictions_csv_file
from .detector import HallucinationDetector
from .evaluator import evaluate_files, evaluate_predictions
from .models import DetectorError


def _add_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("offline", "llm", "hybrid"), default="offline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hallucination_detector", description="客服回复幻觉检测工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="检测回复，不读取人工标注")
    detect.add_argument("--input", required=True)
    detect.add_argument("--output", required=True)
    _add_mode(detect)

    evaluate = subparsers.add_parser("evaluate", help="评估已有预测")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--ground-truth", required=True)

    run = subparsers.add_parser("run", help="检测并在独立阶段评估")
    run.add_argument("--input", required=True)
    run.add_argument("--ground-truth", required=True)
    run.add_argument("--output-dir", required=True)
    _add_mode(run)
    return parser


def detect_file(input_path: str, output_path: str, mode: str) -> list[dict[str, Any]]:
    # Labels are never accepted by this data-loading path.
    items = load_detection_input(input_path)
    predictions = HallucinationDetector(mode).detect(items)
    write_json(output_path, predictions)
    return predictions


def _write_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    write_predictions_csv_file(path, predictions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "detect":
            predictions = detect_file(args.input, args.output, args.mode)
            print(json.dumps({"detected": len(predictions), "output": args.output, "mode": args.mode}, ensure_ascii=False))
            return 0

        if args.command == "evaluate":
            metrics = evaluate_files(args.predictions, args.ground_truth)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0

        output_dir = Path(args.output_dir)
        predictions_path = output_dir / "predictions.json"
        predictions = detect_file(args.input, str(predictions_path), args.mode)
        # Ground truth is first loaded here, after detection is complete.
        ground_truth = read_json(args.ground_truth)
        metrics = evaluate_predictions(predictions, ground_truth)
        write_json(output_dir / "metrics.json", metrics)
        _write_csv(output_dir / "predictions.csv", predictions)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except (DetectorError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
