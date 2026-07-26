"""Claim-to-evidence semantic judgement prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt
from ..utils.text import compact_text


def _evidence_view(evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "id": str(item.get("id", "")),
            "text": compact_text(item.get("text"), 900),
        }
        for item in evidence[:6]
        if isinstance(item, dict)
    ]


def build_claim_batch_judge_prompt(items: list[dict[str, Any]]) -> JsonPrompt:
    """Build one bounded semantic-evaluation batch for several claims."""

    payload_items = []
    for item in items:
        claim = item.get("claim") if isinstance(item, dict) else {}
        payload_items.append(
            {
                "id": str(item.get("id", "")),
                "claim": compact_text(
                    claim.get("text", "") if isinstance(claim, dict) else "", 800
                ),
                "evidence": _evidence_view(item.get("evidence", [])),
            }
        )
    return JsonPrompt(
        task=(
            "逐条判断证据是否直接支持结论，不使用外部知识。输出结构："
            '{"verdicts":[{"id":"C1","verdict":"SUPPORTED"}]}。'
            "必须为每个输入 id 返回一次。全部关键表述均被直接支持选 SUPPORTED；"
            "证据矛盾或不支持选 NOT_SUPPORTED；间接、含糊或不足以判断选 UNCERTAIN。"
        ),
        payload={"items": payload_items},
        max_tokens=min(900, 120 + 70 * len(payload_items)),
    )
