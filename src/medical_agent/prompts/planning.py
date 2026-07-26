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
            "将请求拆为 2 到 6 个可执行任务。返回模板："
            '{"tasks":[{"id":1,"goal":"一句话任务目标","deps":[]}]}。'
            "id 必须从 1 连续递增；deps 只能包含小于当前 id 的整数；"
            f"保留必要依赖，不要生成报告任务。{patient_instruction}"
        ),
        payload={
            "request": compact_text(request),
            "patient_record": compact_text(patient_record),
        },
        max_tokens=900,
    )
