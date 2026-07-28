from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hallucination_detector.csv_utils import write_predictions_csv_file


class CsvSafetyTests(unittest.TestCase):
    def test_file_export_has_bom_and_escapes_formula_fields(self) -> None:
        row = {
            "id": "=1+1",
            "user_question": "+question",
            "is_hallucination": True,
            "category": "无依据事实",
            "severity": "中",
            "confidence": 0.8,
            "claims": "-claim",
            "evidence": "@evidence",
            "reason": "normal",
            "detection_mode": "offline",
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_predictions_csv_file(path, [row])
            raw = path.read_bytes()
            content = raw.decode("utf-8-sig")
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        for value in ("'=1+1", "'+question", "'-claim", "'@evidence"):
            self.assertIn(value, content)


if __name__ == "__main__":
    unittest.main()
