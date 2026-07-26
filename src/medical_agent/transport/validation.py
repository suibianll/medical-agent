"""Validate and bound untrusted HTTP payloads before application dispatch."""

from __future__ import annotations

from typing import Any


MAX_REQUEST_CHARS = 20_000
MAX_PATIENT_RECORD_CHARS = 200_000
MAX_HISTORY_ITEMS = 20


def text_field(
    payload: dict[str, Any],
    key: str,
    *,
    maximum: int,
    default: str = "",
) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} 必须是字符串")
    if len(value) > maximum:
        raise ValueError(f"{key} 不能超过 {maximum:,} 个字符")
    return value


def history_field(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("history", [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("history 必须是数组")
    if len(value) > MAX_HISTORY_ITEMS:
        raise ValueError(f"history 不能超过 {MAX_HISTORY_ITEMS} 项")
    return value


def validate_run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "request": text_field(payload, "request", maximum=MAX_REQUEST_CHARS),
        "patient_record": text_field(
            payload,
            "patientRecord",
            maximum=MAX_PATIENT_RECORD_CHARS,
        ),
        "plan": payload.get("plan"),
        "report_template": payload.get("reportTemplate"),
        "model_profile": text_field(payload, "modelProfile", maximum=64, default="") or None,
    }


def validate_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": text_field(payload, "message", maximum=MAX_REQUEST_CHARS),
        "patient_record": text_field(
            payload,
            "patientRecord",
            maximum=MAX_PATIENT_RECORD_CHARS,
        ),
        "history": history_field(payload),
        "report_template": payload.get("reportTemplate"),
        "model_profile": text_field(payload, "modelProfile", maximum=64, default="") or None,
    }

