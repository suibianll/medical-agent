"""Conversation-context prompt preparation."""

from __future__ import annotations

from typing import Any


def normalize_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", item.get("text", ""))).strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content[:1200]})
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
        f"当前问题：{request}\n\n"
        "以下对话仅用于理解指代与上下文，不能作为患者事实或外部医学证据：\n"
        f"{turns}"
    )
