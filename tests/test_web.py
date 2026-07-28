from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app as webapp
from hallucination_detector.models import LLMError

ROOT = Path(__file__).resolve().parents[1]


def prediction(marker: str = "safe") -> dict[str, object]:
    return {
        "id": marker,
        "user_question": f"问题-{marker}",
        "is_hallucination": False,
        "category": None,
        "severity": "无",
        "confidence": 0.9,
        "claims": marker,
        "evidence": "证据",
        "reason": "一致",
        "detection_mode": "offline",
    }


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        output = Path(self.temporary.name)
        replacements = {
            "OUTPUT_DIR": output,
            "PREDICTIONS_PATH": output / "predictions.json",
            "METRICS_PATH": output / "metrics.json",
            "RUNS_DIR": output / "runs",
            "MANIFEST_PATH": output / "manifest.json",
        }
        self.patchers = [patch.object(webapp, name, value) for name, value in replacements.items()]
        for patcher in self.patchers:
            patcher.start()
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def test_homepage(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("客服回复幻觉检测".encode(), response.data)

    def test_mobile_table_overflow_is_scoped_to_table_container(self) -> None:
        css = (ROOT / "static" / "app.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"html,\s*body\s*\{[^}]*max-width:\s*100%[^}]*overflow-x:\s*hidden")
        self.assertRegex(
            css,
            r"@supports\s*\(overflow-x:\s*clip\)\s*\{\s*html,\s*body\s*\{[^}]*overflow-x:\s*clip",
        )
        self.assertRegex(css, r"\.workspace\s*\{[^}]*min-width:\s*0")
        self.assertRegex(css, r"\.workspace\s*>\s*section\s*\{[^}]*min-width:\s*0[^}]*max-width:\s*100%")
        self.assertRegex(
            css,
            r"\.table-wrap\s*\{[^}]*width:\s*100%[^}]*min-width:\s*0[^}]*max-width:\s*100%[^}]*overflow-x:\s*auto[^}]*contain:\s*inline-size\s+layout\s+paint",
        )
        self.assertRegex(css, r"table\s*\{[^}]*min-width:\s*900px")

    def test_run_results_and_exports(self) -> None:
        run = self.client.post("/api/run", json={"mode": "offline"})
        self.assertEqual(run.status_code, 200)
        payload = run.get_json()
        self.assertRegex(payload["run_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(len(payload["predictions"]), 20)
        self.assertIn("user_question", payload["predictions"][0])
        self.assertEqual(payload["metrics"]["confusion_matrix"], {"tp": 18, "tn": 2, "fp": 0, "fn": 0})

        results = self.client.get("/api/results").get_json()
        self.assertEqual(results["run_id"], payload["run_id"])
        self.assertEqual(results["metrics"]["accuracy"], 1.0)

        exported_json = self.client.get("/api/export/json")
        self.assertEqual(exported_json.status_code, 200)
        self.assertEqual(len(exported_json.get_json()), 20)

        exported_csv = self.client.get("/api/export/csv")
        self.assertEqual(exported_csv.status_code, 200)
        self.assertTrue(exported_csv.data.startswith(b"\xef\xbb\xbf"))
        self.assertIn("user_question", exported_csv.data.decode("utf-8-sig").splitlines()[0])

    def test_detect_publishes_snapshot_with_invalidated_metrics(self) -> None:
        first = self.client.post("/api/run", json={"mode": "offline"}).get_json()
        detected = self.client.post("/api/detect", json={"mode": "offline"}).get_json()
        self.assertNotEqual(first["run_id"], detected["run_id"])
        self.assertIsNone(detected["metrics"])
        results = self.client.get("/api/results").get_json()
        self.assertEqual(results["run_id"], detected["run_id"])
        self.assertIsNone(results["metrics"])
        self.assertFalse(webapp.METRICS_PATH.exists())

    def test_concurrent_runs_never_mix_snapshot_components(self) -> None:
        def fake_predict(mode):
            time.sleep(0.02)
            return [prediction(mode)]

        def fake_evaluate(rows):
            return {"marker": rows[0]["id"]}

        def call(mode):
            with webapp.app.test_client() as client:
                return client.post("/api/run", json={"mode": mode}).get_json()

        with patch.object(webapp, "_predict", side_effect=fake_predict), patch.object(webapp, "_evaluate", side_effect=fake_evaluate):
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(call, ("offline", "hybrid")))
            current = self.client.get("/api/results").get_json()

        self.assertIn(current["run_id"], {item["run_id"] for item in responses})
        self.assertEqual(current["mode"], current["predictions"][0]["id"])
        self.assertEqual(current["metrics"]["marker"], current["predictions"][0]["id"])

    def test_json_contract_rejects_bad_requests(self) -> None:
        cases = (
            self.client.post("/api/run", data="{}"),
            self.client.post("/api/run", data="{", content_type="application/json"),
            self.client.post("/api/run", json=[]),
            self.client.post("/api/run", json={}),
        )
        for response in cases:
            with self.subTest(body=response.data):
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.content_type, "application/json")
                self.assertIsInstance(response.get_json().get("error"), str)

    def test_upstream_secret_is_redacted_from_api_error(self) -> None:
        secret = "reflected-test-bearer-secret"
        with patch.object(webapp, "_predict", side_effect=LLMError(f"Bearer {secret}")):
            response = self.client.post("/api/run", json={"mode": "llm"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(secret, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["error"], "LLM 服务暂不可用或返回了无效响应")

    def test_evaluate_api_reports_invalid_prediction_schema(self) -> None:
        invalid = prediction("invalid-label")
        invalid["is_hallucination"] = "false"
        with webapp.RUN_LOCK:
            webapp._store_snapshot([invalid], None, "offline")
        response = self.client.post("/api/evaluate")
        self.assertEqual(response.status_code, 400)
        self.assertIn("must be boolean", response.get_json()["error"])

    def test_web_csv_export_blocks_formula_injection(self) -> None:
        malicious = prediction("=2+2")
        malicious["user_question"] = "+cmd"
        malicious["claims"] = "@SUM(A1:A2)"
        with patch.object(webapp, "_predict", return_value=[malicious]):
            self.client.post("/api/detect", json={"mode": "offline"})
        content = self.client.get("/api/export/csv").data.decode("utf-8-sig")
        self.assertIn("'=2+2", content)
        self.assertIn("'+cmd", content)
        self.assertIn("'@SUM(A1:A2)", content)

    def test_invalid_mode(self) -> None:
        response = self.client.post("/api/run", json={"mode": "unknown"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("mode must be", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
