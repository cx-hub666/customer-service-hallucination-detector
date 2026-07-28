"""Detection mode orchestration."""

from __future__ import annotations

from typing import Any

from .data_io import validate_detection_input
from .llm import LLMClient
from .models import LLMError
from .offline import OfflineDetector


class HallucinationDetector:
    def __init__(self, mode: str = "offline", *, llm_client: LLMClient | None = None) -> None:
        if mode not in {"offline", "llm", "hybrid"}:
            raise ValueError("mode must be offline, llm or hybrid")
        self.mode = mode
        self.offline = OfflineDetector()
        self._llm_client = llm_client

    def _llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def detect(self, data: Any) -> list[dict[str, object]]:
        items = validate_detection_input(data)
        return [self.detect_one(item) for item in items]

    def detect_one(self, item: dict[str, str]) -> dict[str, object]:
        if self.mode == "offline":
            return self.offline.detect_one(item, "offline")
        if self.mode == "llm":
            return self._llm().detect_one(item)

        offline_result = self.offline.detect_one(item, "hybrid")
        try:
            llm_result = self._llm().detect_one(item)
        except LLMError:
            offline_result["reason"] = f"{offline_result['reason']}（LLM 不可用，混合模式采用离线结果。）"
            return offline_result

        agreed = offline_result["is_hallucination"] == llm_result["is_hallucination"]
        if agreed:
            adopted = offline_result if offline_result["is_hallucination"] else llm_result
            adopted["confidence"] = round(max(float(offline_result["confidence"]), float(llm_result["confidence"])), 4)
            adopted["detection_mode"] = "hybrid"
            adopted["disagreement"] = False
            return adopted

        adopted = offline_result if offline_result["is_hallucination"] else llm_result
        adopted["detection_mode"] = "hybrid"
        adopted["disagreement"] = True
        adopted["disagreement_reason"] = (
            "离线规则与 LLM 结论不一致；采用检测到幻觉的分支，且保留该分支原始置信度。"
        )
        return adopted
