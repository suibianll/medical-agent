"""Task-planning prompt."""

from __future__ import annotations

from .types import JsonPrompt
from ..utils.text import compact_text


def build_plan_prompt(request: str, patient_record: str) -> JsonPrompt:
    patient_instruction = (
        "已提供患者病历：可以设置患者事实提取任务。"
        if patient_record.strip()
        else "未提供患者病历：只规划一般医学知识检索与回答任务，不要虚构患者事实。"
    )
    return JsonPrompt(
        task=(
            "用尽可能少的任务规划请求，通常 2 到 4 个，最多 5 个。输出结构："
            '{"tasks":[{"id":1,"goal":"一句话任务目标","deps":[]}]}。'
            "id 从 1 连续递增；goal 必须单一、可执行；deps 只填写真正需要其输出的较小 id，"
            "可并行任务使用空 deps。不要规划报告排版、引用编号或评估任务。"
            f"{patient_instruction}"
        ),
        payload={
            "request": compact_text(request, 1500),
            "patient_record": compact_text(patient_record, 3500),
        },
        max_tokens=700,
    )
