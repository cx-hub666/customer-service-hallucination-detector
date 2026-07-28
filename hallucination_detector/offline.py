"""Deterministic knowledge-grounded detector for common service reply risks."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    confidence: float
    reason: str


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|ms|天|小时|个月|年|折|元|码)?", text, re.IGNORECASE))


def _affirmative(text: str) -> bool:
    return _has(text, r"(?:^|[，。；])(?:是的|有的|可以|支持)", r"放心使用", r"已经?", r"已帮")


def _contradicts_absence(reply: str, kb: str) -> bool:
    kb_denies = _has(kb, r"不支持", r"无(?:线下|学生|任何|相关)", r"未(?:标注|提及)", r"不可", r"不具备")
    return kb_denies and _affirmative(reply) and not _has(reply, r"不支持", r"没有", r"无法")


_COUNTRIES = ("中国", "法国", "德国", "意大利", "日本", "韩国", "美国", "英国", "瑞士", "西班牙")
_COLORS = ("红色", "蓝色", "绿色", "黑色", "白色", "黄色", "紫色", "粉色", "灰色", "棕色")
_CHINESE_NUMBERS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12, "二十四": 24, "三十": 30}


def _duration_months(text: str) -> set[str]:
    values: set[str] = set()
    for raw, unit in re.findall(r"(\d+|[一二两三四五六七八九十]{1,3})(个?月|年)", text):
        number = int(raw) if raw.isdigit() else _CHINESE_NUMBERS.get(raw)
        if number is not None:
            values.add(f"{number * 12 if unit == '年' else number}个月")
    return values


def _slot_values(text: str, context: str) -> dict[str, set[str]]:
    slots: dict[str, set[str]] = {}
    if _has(context, r"产地|原产|生产地|哪里.*(?:产|制造)"):
        slots["产地"] = {value for value in _COUNTRIES if value in text}
    if _has(context, r"颜色|配色|色款|什么色"):
        slots["颜色"] = {value for value in _COLORS if value in text}
    if _has(context, r"保修|质保|有效期|期限|多久"):
        slots["期限"] = _duration_months(text)
    return slots


def _slot_conflict(question: str, reply: str, kb: str) -> str | None:
    context = question + kb
    reply_slots = _slot_values(reply, context)
    kb_slots = _slot_values(kb, context)
    for name in reply_slots.keys() & kb_slots.keys():
        claimed = reply_slots[name]
        supported = kb_slots[name]
        if claimed and supported and claimed.isdisjoint(supported):
            return name
    return None


class OfflineDetector:
    """Rule ensemble based only on question, reply and supplied knowledge."""

    def detect_one(self, item: dict[str, str], mode: str = "offline") -> dict[str, object]:
        question = item["user_question"]
        reply = item["system_reply"]
        kb = item["knowledge_base"]

        finding = (
            self._safety(reply, kb)
            or self._omission(question, reply, kb)
            or self._promotion_or_policy(question, reply, kb)
            or self._capability(reply, kb)
            or self._product_fact(question, reply, kb)
            or self._generic_fact(question, reply, kb)
            or self._unsupported_fact(reply, kb)
        )

        if finding is None:
            return {
                "id": item["id"],
                "user_question": question,
                "is_hallucination": False,
                "category": None,
                "severity": "无",
                "confidence": 0.93,
                "claims": reply,
                "evidence": kb,
                "reason": "客服回复与给定知识库在用户所问事项上未发现实质冲突。",
                "detection_mode": mode,
            }

        return {
            "id": item["id"],
            "user_question": question,
            "is_hallucination": True,
            "category": finding.category,
            "severity": finding.severity,
            "confidence": finding.confidence,
            "claims": reply,
            "evidence": kb,
            "reason": finding.reason,
            "detection_mode": mode,
        }

    @staticmethod
    def _safety(reply: str, kb: str) -> Finding | None:
        caution = _has(kb, r"咨询医生", r"孕妇.*建议", r"注意事项", r"禁用", r"风险")
        assurance = _has(reply, r"放心使用", r"绝对安全", r"孕妇可以", r"没有任何风险")
        if caution and assurance:
            return Finding("安全误导", "严重", 0.99, "回复给出确定的安全保证，但知识库要求风险提示或专业咨询。")
        return None

    @staticmethod
    def _omission(question: str, reply: str, kb: str) -> Finding | None:
        size_context = _has(question + kb, r"尺码", r"偏大", r"偏小")
        evidence_has_feedback = _has(kb, r"\d+%.*(?:反馈|偏大|偏小)", r"建议.*(?:大|小)半码")
        reply_overgeneralizes = _has(reply, r"尺码标准", r"不偏大.*不偏小", r"按.*平时.*尺码")
        if size_context and evidence_has_feedback and reply_overgeneralizes:
            return Finding("关键信息遗漏", "中", 0.97, "回复作出笼统尺码判断，遗漏了知识库中的用户反馈和选码建议。")
        return None

    @staticmethod
    def _promotion_or_policy(question: str, reply: str, kb: str) -> Finding | None:
        context = question + kb
        is_policy = _has(context, r"政策", r"优惠", r"活动", r"退货", r"发票", r"发货", r"快递", r"运费", r"学生")
        if not is_policy:
            return None

        no_reason_reply = re.search(r"(\d+)天无理由", reply)
        no_reason_kb = re.search(r"(\d+)天无理由", kb)
        term_conflict = bool(no_reason_reply and no_reason_kb and no_reason_reply.group(1) != no_reason_kb.group(1))
        is_address_request = _has(question + reply, r"地址", r"寄到", r"邮编")
        unsupported_number = bool(_numbers(reply) - _numbers(kb)) and not is_address_request
        absence_conflict = _contradicts_absence(reply, kb)
        invoice_conflict = _has(reply, r"支持.*纸质发票") and _has(kb, r"不支持.*纸质发票")
        process_conflict = _has(reply, r"备注.*发票") and _has(kb, r"订单详情页.*申请")
        promotion_denial = _has(kb, r"无.*(?:优惠|活动)", r"没有.*(?:优惠|活动)") and _has(reply, r"有的", r"现在有", r"可以.*优惠")
        shipping_conflict = _has(context, r"发货|快递") and (
            unsupported_number
            or (_has(reply, r"顺丰") and _has(kb, r"中通|韵达|圆通"))
        )
        freight_conflict = _has(reply, r"运费.*(?:我们|商家).*承担") and _has(kb, r"运费.*买家承担")

        if term_conflict or absence_conflict or invoice_conflict or process_conflict or promotion_denial or shipping_conflict or freight_conflict or unsupported_number:
            severity = "高" if _has(context, r"退货|优惠|学生|发票") else "中"
            return Finding("政策与优惠", severity, 0.98, "回复中的政策、优惠、时效或办理方式与知识库的明确规定不一致。")
        return None

    @staticmethod
    def _capability(reply: str, kb: str) -> Finding | None:
        unavailable = _has(kb, r"未接入.*(?:接口|查询)", r"不具备.*功能", r"需人工.*操作", r"需转人工")
        claimed_action = _has(
            reply,
            r"(?:帮您|我).*查(?:了|到)",
            r"已经?帮您",
            r"已经.*(?:修改|升级|处理)",
            r"预计.*(?:送达|到账)",
            r"(?:专属客服|人工).*(?:小时|分钟)内联系",
        )
        if unavailable and claimed_action:
            return Finding("能力越界", "高", 0.99, "知识库说明系统不具备该查询或操作能力，回复却声称已经查询或执行成功。")
        return None

    @staticmethod
    def _product_fact(question: str, reply: str, kb: str) -> Finding | None:
        context = question + kb
        is_product = _has(context, r"产品参数", r"材质", r"保修", r"接口", r"蓝牙", r"NFC", r"成分", r"型号")
        if not is_product:
            return None
        absence_conflict = _has(kb, r"未标注.*NFC") and _has(reply, r"支持.*NFC")
        numeric_conflict = bool(_numbers(reply) - _numbers(kb))
        known_pairs = (
            (r"PU合成革", r"(?:头层|真)牛皮"),
            (r"USB-A", r"(?:是|为).*Type-C接口"),
            (r"单设备", r"多设备"),
        )
        semantic_conflict = any(_has(kb, left) and _has(reply, right) for left, right in known_pairs)
        if absence_conflict or numeric_conflict or semantic_conflict:
            return Finding("产品事实与参数", "高", 0.98, "回复给出的产品参数、材质、功能或保修信息与知识库不一致或无依据。")
        return None

    @staticmethod
    def _unsupported_fact(reply: str, kb: str) -> Finding | None:
        address_claim = _has(reply, r"(?:省|市|区).*(?:路|号).*(?:收|邮编)") and _has(kb, r"地址.*(?:自动匹配|不可口头告知)")
        store_claim = _has(reply, r"线下.*(?:门店|体验店)", r"门店查询") and _has(kb, r"无线下门店|纯线上")
        relation_claim = _has(reply, r"(?:旗下|子品牌|同一家公司|共享.*供应链)") and _has(kb, r"未提及.*关联|无.*关联")
        if address_claim or store_claim or relation_claim or _contradicts_absence(reply, kb):
            severity = "高" if address_claim else "中"
            return Finding("无依据事实", severity, 0.97, "回复陈述了知识库明确否定或未提供依据的地址、门店、品牌关系等事实。")
        return None

    @staticmethod
    def _generic_fact(question: str, reply: str, kb: str) -> Finding | None:
        slot = _slot_conflict(question, reply, kb)
        if slot:
            return Finding("产品事实与参数", "高", 0.96, f"回复中的{slot}属性值与知识库提供的值不一致。")
        return None
