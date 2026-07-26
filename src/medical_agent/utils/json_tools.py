"""Tolerant parsing for weak-model JSON responses."""

from __future__ import annotations

import json
import re
from typing import Any, Type


def extract_json_object(
    content: str, *, error_type: Type[Exception] = ValueError
) -> dict[str, Any]:
    source = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", source, flags=re.DOTALL | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(source)
    object_start = source.find("{")
    if object_start >= 0:
        candidates.append(source[object_start:])

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise error_type("模型未返回有效的 JSON 对象。")
