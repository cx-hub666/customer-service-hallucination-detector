from __future__ import annotations

import unittest

from hallucination_detector.data_io import read_json
from hallucination_detector.detector import HallucinationDetector


class FakeLLM:
    def __init__(self, decision: bool, confidence: float) -> None:
        self.decision = decision
        self.confidence = confidence

    def detect_one(self, item):
        return {
            "id": item["id"],
            "user_question": item["user_question"],
            "is_hallucination": self.decision,
            "category": "无依据事实" if self.decision else None,
            "severity": "中" if self.decision else "无",
            "confidence": self.confidence,
            "claims": item["system_reply"],
            "evidence": item["knowledge_base"],
            "reason": "模拟 LLM 结论",
            "detection_mode": "llm",
        }


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        items = read_json("data/replies.json")
        cls.positive = items[0]
        cls.normal = items[11]

    def test_agreement_merges_confidence(self) -> None:
        result = HallucinationDetector("hybrid", llm_client=FakeLLM(True, 0.999)).detect([self.positive])[0]
        self.assertEqual(result["confidence"], 0.999)
        self.assertFalse(result["disagreement"])

    def test_conflict_preserves_adopted_branch_confidence(self) -> None:
        result = HallucinationDetector("hybrid", llm_client=FakeLLM(False, 0.999)).detect([self.positive])[0]
        self.assertTrue(result["is_hallucination"])
        self.assertEqual(result["confidence"], 0.98)
        self.assertTrue(result["disagreement"])
        self.assertIn("不一致", result["disagreement_reason"])

        result = HallucinationDetector("hybrid", llm_client=FakeLLM(True, 0.61)).detect([self.normal])[0]
        self.assertTrue(result["is_hallucination"])
        self.assertEqual(result["confidence"], 0.61)
        self.assertTrue(result["disagreement"])
