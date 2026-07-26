"""Small immutable prompt protocols shared by prompt modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatPrompt:
    system: str
    user: str
    max_tokens: int = 1200


@dataclass(frozen=True, slots=True)
class JsonPrompt:
    task: str
    payload: dict[str, Any]
    max_tokens: int = 1200
