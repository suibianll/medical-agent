"""Conversation-context prompt preparation."""

from __future__ import annotations

from typing import Any


def normalize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in history[-4:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content[:600]})
    return normalized


def build_contextual_request(request: str, history: Any) -> str:
    context = normalize_history(history)
    if not context:
        return request
    turns = "\n".join(
        f"{'用户' if item['role'] == 'user' else '系统先前回答'}：{item['content']}"
        for item in context
    )
    return (
        f"当前问题：{request}\n"
        "对话上下文（仅用于消解指代，不是患者事实或医学证据）：\n"
        f"{turns}"
    )
