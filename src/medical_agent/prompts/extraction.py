"""Evidence fact-extraction prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt
from .task_context import task_prompt_view


def build_fact_extraction_prompt(
    *, task: dict[str, Any], evidence: list[dict[str, str]]
) -> JsonPrompt:
    return JsonPrompt(
        task=(
            "仅从证据原文提取与任务直接相关的最多 6 条原子事实。输出结构："
            '{"facts":[{"text":"原子事实","ref":"输入中的单个证据ID"}]}。'
            "每条事实只对应一个 ref，忠实保留原意；不得推断、合并来源、重复或加入建议。"
            "没有相关事实时返回空 facts。"
        ),
        payload={"task": task_prompt_view(task), "evidence": evidence[:12]},
        max_tokens=700,
    )
