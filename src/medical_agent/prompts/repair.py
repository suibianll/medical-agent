"""Small code-owned repair instructions passed to task prompts."""

from __future__ import annotations

from typing import Any


REPAIR_GUIDANCE = {
    "NO_REF": "删除无依据结论，或只使用本轮已登记证据补充引用。",
    "BAD_REF": "只使用输入中存在的 P# 或 K#，不得改写证据编号。",
    "MISSING_PATIENT_REF": "补充与结论直接相关的患者病历事实 P#。",
    "MISSING_KB_REF": "补充与结论直接相关的医学知识证据 K#。",
    "NOT_SUPPORTED": "缩小结论范围，使每个表述都能被引用事实直接支持。",
}


def build_repair_context(codes: list[str]) -> dict[str, Any]:
    normalized = list(dict.fromkeys(codes))[:8]
    return {
        "codes": normalized,
        "guidance": [REPAIR_GUIDANCE[code] for code in normalized if code in REPAIR_GUIDANCE],
    }
