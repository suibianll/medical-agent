"""Claim-to-evidence semantic judgement prompt."""

from __future__ import annotations

import json
from typing import Any

from .types import ChatPrompt
from ..utils.text import compact_text


def build_claim_judge_prompt(
    *, claim: dict[str, Any], evidence: list[dict[str, Any]]
) -> ChatPrompt:
    return ChatPrompt(
        system=(
            "你是引用核验器。不要解释，也不要输出思维过程。"
            "仅输出一个大写词：SUPPORTED、NOT_SUPPORTED 或 UNCERTAIN。"
        ),
        user=(
            "判断给定证据是否直接支持结论。\n"
            f"结论：{compact_text(claim.get('text'), 1200)}\n"
            f"证据：{json.dumps(evidence, ensure_ascii=False)}"
        ),
        max_tokens=20,
    )
