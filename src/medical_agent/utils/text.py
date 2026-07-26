"""Bounded text transformations used before model calls and audit output."""

from __future__ import annotations

from typing import Any


def compact_text(value: Any, limit: int = 10_000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}\n[已截断]"
