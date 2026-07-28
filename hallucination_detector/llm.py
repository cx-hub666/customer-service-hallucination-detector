"""OpenAI-compatible chat completions client with schema fallback."""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .models import CATEGORIES, LLMError, validate_result

Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def _endpoint(base: str) -> str:
    cleaned = base.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _default_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace").lower()
        if any(token in message for token in ("response_format", "json_schema", "unsupported", "not support")):
            raise LLMError("LLM response_format is unsupported") from exc
        raise LLMError(f"LLM upstream request failed (HTTP {exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise LLMError(f"LLM request failed: {type(reason).__name__}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("LLM endpoint returned non-JSON HTTP content") from exc


class LLMClient:
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 20.0,
        max_retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        self.api_base = api_base or os.getenv("LLM_API_BASE", "")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout = timeout
        self.max_retries = max(0, min(max_retries, 4))
        self.transport = transport or _default_transport
        if not self.api_base or not self.api_key or not self.model:
            raise LLMError("LLM_API_BASE, LLM_API_KEY and LLM_MODEL are required for LLM mode")
        if not self.api_base.startswith(("http://", "https://")):
            raise LLMError("LLM_API_BASE must be an HTTP(S) URL")

    @property
    def url(self) -> str:
        return _endpoint(self.api_base)

    def _messages(self, item: dict[str, str], strict_json: bool) -> list[dict[str, str]]:
        format_note = "只返回一个 JSON 对象，不要使用 Markdown。" if strict_json else ""
        return [
            {
                "role": "system",
                "content": (
                    "你是客服回复事实核查器。只使用提供的知识库，判断回复是否存在幻觉。"
                    f"分类只能是：{'、'.join(CATEGORIES)}。{format_note}"
                    "字段必须为 is_hallucination(boolean), category(string|null), severity(无/低/中/高/严重), "
                    "confidence(0到1), claims(string), evidence(string), reason(string)。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_question": item["user_question"],
                        "system_reply": item["system_reply"],
                        "knowledge_base": item["knowledge_base"],
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "hallucination_detection",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["is_hallucination", "category", "severity", "confidence", "claims", "evidence", "reason"],
                    "properties": {
                        "is_hallucination": {"type": "boolean"},
                        "category": {"type": ["string", "null"], "enum": [*CATEGORIES, None]},
                        "severity": {"type": "string", "enum": ["无", "低", "中", "高", "严重"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "claims": {"type": "string"},
                        "evidence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        }

    @staticmethod
    def _content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response is incompatible with chat/completions") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM response content is empty or incompatible")
        return content.strip()

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise LLMError("LLM response did not contain valid JSON")
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMError("LLM response did not contain valid JSON") from exc
        if not isinstance(value, dict):
            raise LLMError("LLM response JSON must be an object")
        return value

    @staticmethod
    def _format_unsupported(error: LLMError) -> bool:
        message = str(error).lower()
        return any(token in message for token in ("response_format", "json_schema", "unsupported", "not support"))

    def detect_one(self, item: dict[str, str]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        use_schema = True
        compatibility_fallback_used = False
        last_error: LLMError | None = None
        failures = 0
        while failures <= self.max_retries:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self._messages(item, strict_json=not use_schema),
                "temperature": 0,
            }
            if use_schema:
                payload["response_format"] = self._schema()
            try:
                try:
                    response = self.transport(self.url, headers, payload, self.timeout)
                except LLMError as exc:
                    if use_schema and not compatibility_fallback_used and self._format_unsupported(exc):
                        use_schema = False
                        compatibility_fallback_used = True
                        continue
                    raise LLMError("LLM upstream request failed") from exc
                except (TimeoutError, socket.timeout):
                    raise
                except Exception as exc:
                    raise LLMError("LLM upstream request failed") from exc
                parsed = self._parse_json(self._content(response))
                parsed["detection_mode"] = "llm"
                return validate_result(parsed, expected_id=item["id"], expected_question=item["user_question"])
            except (TimeoutError, socket.timeout):
                last_error = LLMError("LLM request failed: timeout")
            except LLMError as exc:
                last_error = exc
            failures += 1
            if failures <= self.max_retries:
                time.sleep(min(0.15 * (2 ** (failures - 1)), 0.5))
        raise last_error or LLMError("LLM detection failed")
