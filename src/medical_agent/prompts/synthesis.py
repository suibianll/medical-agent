"""Evidence-grounded conclusion synthesis prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt
from ..utils.text import compact_text


def build_synthesis_prompt(
    *,
    task: dict[str, Any],
    request: str,
    facts: list[dict[str, str]],
    evidence: list[dict[str, str]],
) -> JsonPrompt:
    citation_policy = (
        "涉及个体患者分析、风险或建议时，同时引用一个 P# 患者事实和一个 K# 知识库证据。"
        if bool(task.get("patient_grounding_required", True))
        else "当前没有患者病历；每条结论至少引用一个 K# 知识库证据，且不得声称适用于某个具体患者。"
    )
    return JsonPrompt(
        task=(
            "根据事实回答当前子任务。返回模板："
            '{"claims":[{"text":"一条原子、审慎的结论","refs":["P1","K1"]}],'
            '"unknowns":["缺失信息"]}。'
            f"每条可验证结论必须给出至少一个真实 refs；{citation_policy}"
            "证据不足时写入 unknowns。"
        ),
        payload={
            "task": task,
            "request": compact_text(request, 2000),
            "facts": facts[:8],
            "evidence": evidence[:12],
        },
        max_tokens=1000,
    )
