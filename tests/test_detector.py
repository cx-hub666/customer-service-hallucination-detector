from __future__ import annotations

import json
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from hallucination_detector.cli import detect_file, main
from hallucination_detector.data_io import read_json, validate_detection_input
from hallucination_detector.detector import HallucinationDetector
from hallucination_detector.evaluator import evaluate_predictions
from hallucination_detector.models import CATEGORIES, InputValidationError


ROOT = Path(__file__).resolve().parents[1]


class OfflineDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replies = read_json(ROOT / "data" / "replies.json")
        cls.truth = read_json(ROOT / "data" / "ground_truth.json")
        cls.predictions = HallucinationDetector("offline").detect(cls.replies)

    def test_full_dataset_metrics_and_normal_cases(self) -> None:
        metrics = evaluate_predictions(self.predictions, self.truth)
        self.assertEqual(metrics["confusion_matrix"], {"tp": 18, "tn": 2, "fp": 0, "fn": 0})
        by_id = {row["id"]: row for row in self.predictions}
        self.assertFalse(by_id["h12"]["is_hallucination"])
        self.assertFalse(by_id["h16"]["is_hallucination"])

    def test_all_six_categories_are_covered(self) -> None:
        found = {row["category"] for row in self.predictions if row["is_hallucination"]}
        self.assertEqual(found, set(CATEGORIES))

    def test_input_order_is_preserved(self) -> None:
        inputs = list(reversed(self.replies))
        outputs = HallucinationDetector("offline").detect(inputs)
        self.assertEqual([row["id"] for row in outputs], [row["id"] for row in inputs])

    def test_rules_are_not_tied_to_sample_ids(self) -> None:
        item = dict(self.replies[0], id="custom-policy-case")
        result = HallucinationDetector("offline").detect([item])[0]
        self.assertEqual(result["id"], "custom-policy-case")
        self.assertTrue(result["is_hallucination"])
        self.assertEqual(result["category"], "政策与优惠")

    def test_detection_rejects_evaluation_labels_at_any_depth(self) -> None:
        leaked = [dict(self.replies[0], metadata={"ground_truth": True})]
        with self.assertRaisesRegex(InputValidationError, "label leakage"):
            validate_detection_input(leaked)

    def test_detect_cli_stage_reads_no_ground_truth(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.json"
            predictions = detect_file(str(ROOT / "data" / "replies.json"), str(output), "offline")
            self.assertEqual(len(predictions), 20)
            self.assertTrue(output.exists())
            self.assertNotIn("ground_truth_detail", json.dumps(predictions, ensure_ascii=False))

    def test_blind_fact_slot_fixture_and_paraphrases(self) -> None:
        cases = read_json(ROOT / "tests" / "fixtures" / "blind_cases.json")
        inputs = [{key: value for key, value in case.items() if key != "expected"} for case in cases]
        outputs = HallucinationDetector("offline").detect(inputs)
        self.assertEqual(
            [row["is_hallucination"] for row in outputs],
            [case["expected"] for case in cases],
        )
        self.assertEqual(outputs[0]["category"], "产品事实与参数")
        self.assertEqual(outputs[1]["category"], "产品事实与参数")

    def test_predictions_include_original_question(self) -> None:
        self.assertEqual(self.predictions[0]["user_question"], self.replies[0]["user_question"])


class EvaluationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        replies = read_json(ROOT / "data" / "replies.json")
        self.prediction = HallucinationDetector("offline").detect([replies[0]])[0]
        self.truth = [{"id": self.prediction["id"], "is_hallucination": True}]

    def test_rejects_string_and_missing_ground_truth_labels(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "must be boolean"):
            evaluate_predictions([self.prediction], [{"id": self.prediction["id"], "is_hallucination": "false"}])
        with self.assertRaisesRegex(InputValidationError, "missing is_hallucination"):
            evaluate_predictions([self.prediction], [{"id": self.prediction["id"]}])

    def test_rejects_string_and_missing_prediction_labels(self) -> None:
        invalid = dict(self.prediction, is_hallucination="false")
        with self.assertRaisesRegex(InputValidationError, "must be boolean"):
            evaluate_predictions([invalid], self.truth)
        invalid = dict(self.prediction)
        del invalid["is_hallucination"]
        with self.assertRaisesRegex(InputValidationError, "missing fields"):
            evaluate_predictions([invalid], self.truth)

    def test_rejects_invalid_prediction_category(self) -> None:
        invalid = dict(self.prediction, category="其他")
        with self.assertRaisesRegex(InputValidationError, "category is invalid"):
            evaluate_predictions([invalid], self.truth)

    def test_cli_reports_invalid_label_type(self) -> None:
        with TemporaryDirectory() as directory:
            predictions_path = Path(directory) / "predictions.json"
            truth_path = Path(directory) / "truth.json"
            predictions_path.write_text(json.dumps([self.prediction], ensure_ascii=False), encoding="utf-8")
            truth_path.write_text(
                json.dumps([{"id": self.prediction["id"], "is_hallucination": "false"}]),
                encoding="utf-8",
            )
            errors = io.StringIO()
            with redirect_stderr(errors):
                status = main(["evaluate", "--predictions", str(predictions_path), "--ground-truth", str(truth_path)])
        self.assertEqual(status, 2)
        self.assertIn("must be boolean", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
