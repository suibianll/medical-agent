"""Retrieval-query generation prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt
from ..utils.text import compact_text


def build_query_prompt(
    *,
    task: dict[str, Any],
    request: str,
    patient_record: str,
    upstream: dict[int, Any],
) -> JsonPrompt:
    upstream_claims = {
        str(task_id): [
            {
                "text": compact_text(claim.get("text"), 300),
                "refs": claim.get("refs", []),
            }
            for claim in (result or {}).get("claims", [])[:3]
        ]
        for task_id, result in upstream.items()
    }
    return JsonPrompt(
        task='为当前子任务生成 1 到 3 条简短检索查询。返回模板：{"queries":["查询"]}。',
        payload={
            "task": task,
            "request": compact_text(request, 2000),
            "patient_record": compact_text(patient_record, 4000),
            "upstream_claims": upstream_claims,
        },
        max_tokens=500,
    )
