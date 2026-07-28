from __future__ import annotations

import os
import socket
import unittest
from unittest.mock import patch

from hallucination_detector.llm import LLMClient
from hallucination_detector.models import LLMError


ITEM = {"id": "case-1", "user_question": "支持吗？", "system_reply": "支持。", "knowledge_base": "不支持。"}
VALID = {
    "is_hallucination": True,
    "category": "无依据事实",
    "severity": "中",
    "confidence": 0.8,
    "claims": "支持",
    "evidence": "不支持",
    "reason": "与知识库矛盾",
}


def client(transport, retries: int = 0) -> LLMClient:
    return LLMClient("https://example.test/v1", "test-key", "test-model", max_retries=retries, transport=transport)


class LLMClientTests(unittest.TestCase):
    def test_missing_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(LLMError, "required"):
                LLMClient()

    def test_timeout_is_normalized(self) -> None:
        for exception in (TimeoutError(), socket.timeout()):
            with self.subTest(exception=type(exception).__name__):
                with self.assertRaisesRegex(LLMError, "timeout"):
                    client(lambda *_: (_ for _ in ()).throw(exception)).detect_one(ITEM)

    def test_invalid_json(self) -> None:
        response = {"choices": [{"message": {"content": "not-json"}}]}
        with self.assertRaisesRegex(LLMError, "valid JSON"):
            client(lambda *_: response).detect_one(ITEM)

    def test_incompatible_response(self) -> None:
        with self.assertRaisesRegex(LLMError, "incompatible"):
            client(lambda *_: {"output": []}).detect_one(ITEM)

    def test_response_format_fallback(self) -> None:
        payloads = []

        def transport(_url, _headers, payload, _timeout):
            payloads.append(payload)
            if len(payloads) == 1:
                raise LLMError("response_format json_schema unsupported")
            return {"choices": [{"message": {"content": __import__("json").dumps(VALID, ensure_ascii=False)}}]}

        result = client(transport, retries=0).detect_one(ITEM)
        self.assertIn("response_format", payloads[0])
        self.assertNotIn("response_format", payloads[1])
        self.assertTrue(result["is_hallucination"])

    def test_reflected_bearer_key_is_never_exposed(self) -> None:
        secret = "test-secret-reflected-by-upstream"

        def transport(*_args):
            raise LLMError(f"upstream echoed Authorization: Bearer {secret}")

        configured = LLMClient("https://example.test/v1", secret, "test-model", max_retries=0, transport=transport)
        with self.assertRaises(LLMError) as raised:
            configured.detect_one(ITEM)
        self.assertEqual(str(raised.exception), "LLM upstream request failed")
        self.assertNotIn(secret, str(raised.exception))

    def test_result_contains_caller_question_not_model_value(self) -> None:
        response_value = dict(VALID, user_question="伪造问题")
        response = {"choices": [{"message": {"content": __import__("json").dumps(response_value, ensure_ascii=False)}}]}
        result = client(lambda *_: response).detect_one(ITEM)
        self.assertEqual(result["user_question"], ITEM["user_question"])


if __name__ == "__main__":
    unittest.main()
