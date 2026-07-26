"""Small code-owned repair instructions passed to task prompts."""

from __future__ import annotations

from typing import Any


REPAIR_INSTRUCTION = (
    "扩大检索表达并修正引用；缺少 P# 时引用患者事实，"
    "缺少 K# 时引用知识库证据，不得编造证据编号。"
)


def build_repair_context(codes: list[str]) -> dict[str, Any]:
    return {"codes": list(dict.fromkeys(codes))[:8], "instruction": REPAIR_INSTRUCTION}
