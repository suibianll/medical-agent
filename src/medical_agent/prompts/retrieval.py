"""Retrieval-query generation prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt
from .task_context import task_prompt_view
from ..utils.text import compact_text


def build_query_prompt(
    *,
    task: dict[str, Any],
    request: str,
    patient_record: str,
    upstream: dict[int, Any],
) -> JsonPrompt:
    upstream_summaries = {
        str(task_id): [
            compact_text(claim.get("text"), 220)
            for claim in (result or {}).get("claims", [])[:2]
        ]
        for task_id, result in upstream.items()
    }
    return JsonPrompt(
        task=(
            '为当前任务生成 1 到 3 条可直接用于检索的短查询。输出结构：{"queries":["查询"]}。'
            "查询应覆盖任务中的核心医学概念和必要患者特征；不要回答问题，不要写布尔运算符，"
            "不要包含姓名、证件号等身份信息。去除同义重复；修复提示存在时优先补足缺失证据类型。"
        ),
        payload={
            "task": task_prompt_view(task),
            "request": compact_text(request, 1200),
            "patient_record": compact_text(patient_record, 2500),
            "upstream_summaries": upstream_summaries,
        },
        max_tokens=350,
    )
