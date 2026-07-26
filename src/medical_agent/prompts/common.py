"""Shared wrapper for model calls that must return compact JSON."""

from __future__ import annotations

import json

from .types import ChatPrompt, JsonPrompt


JSON_SYSTEM_PROMPT = (
    "你是医疗信息系统中的受限组件。只处理给定数据，不输出诊断、处方、剂量或完整思维链。"
    "必须严格按用户给定的 JSON 模板返回一个 JSON 对象，禁止 Markdown、解释文字和额外字段。"
    "引用只能使用输入中已有的证据 ID。"
)


def render_json_prompt(prompt: JsonPrompt) -> ChatPrompt:
    """Render one task and its bounded payload into provider messages."""

    return ChatPrompt(
        system=JSON_SYSTEM_PROMPT,
        user=(
            f"任务：{prompt.task}\n\n"
            f"输入：\n{json.dumps(prompt.payload, ensure_ascii=False)}"
        ),
        max_tokens=prompt.max_tokens,
    )
