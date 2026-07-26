"""Evidence fact-extraction prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt


def build_fact_extraction_prompt(
    *, task: dict[str, Any], evidence: list[dict[str, str]]
) -> JsonPrompt:
    return JsonPrompt(
        task=(
            "仅从输入证据中提取最多 6 条关键事实。返回模板："
            '{"facts":[{"text":"原子事实","ref":"输入中的单个证据ID"}]}。'
            "不得推断、合并多个来源或编造引用。"
        ),
        payload={"task": task, "evidence": evidence[:12]},
        max_tokens=900,
    )
