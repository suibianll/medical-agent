"""Evidence-grounded conclusion synthesis prompt."""

from __future__ import annotations

from typing import Any

from .types import JsonPrompt
from .task_context import task_prompt_view
from ..utils.text import compact_text


def build_synthesis_prompt(
    *,
    task: dict[str, Any],
    request: str,
    facts: list[dict[str, str]],
) -> JsonPrompt:
    citation_policy = (
        "涉及个体患者分析、风险或建议时，同时引用一个 P# 患者事实和一个 K# 知识库证据。"
        if bool(task.get("patient_grounding_required", True))
        else "当前没有患者病历；每条结论至少引用一个 K# 知识库证据，且不得声称适用于某个具体患者。"
    )
    return JsonPrompt(
        task=(
            "仅根据 facts 完成当前任务，生成最多 4 条原子结论。输出结构："
            '{"claims":[{"text":"一条原子、审慎的结论","refs":["P1","K1"]}],'
            '"unknowns":["缺失信息"]}。'
            "每条 claim 只表达一个可核验意思，refs 只能取自支持该结论的 facts.ref。"
            f"{citation_policy}不得补充诊断、处方、剂量或 facts 中没有的信息。"
            "无法支持的内容不要生成 claim，只写入 unknowns。"
        ),
        payload={
            "task": task_prompt_view(task),
            "request": compact_text(request, 1200),
            "facts": facts[:8],
        },
        max_tokens=800,
    )
