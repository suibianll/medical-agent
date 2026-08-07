"""Safe model-call telemetry helpers.

The workflow must account for latency and token usage without capturing
prompts, patient records, evidence text, or provider responses.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


SAFE_METRIC_KEYS = {
    "stage",
    "provider",
    "model",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "success",
    "retry_count",
    "attempts",
    "error_type",
}


def drain_model_metrics(model: Any) -> list[dict[str, Any]]:
    """Drain metrics from adapters that support the optional telemetry port."""

    drain = getattr(model, "drain_call_metrics", None)
    if not callable(drain):
        return []
    try:
        values = drain()
    except Exception:  # noqa: BLE001 - telemetry must never fail the workflow
        return []
    if not isinstance(values, list):
        return []
    return [
        {
            key: value
            for key, value in item.items()
            if key in SAFE_METRIC_KEYS
            and (
                isinstance(value, (str, bool))
                or (isinstance(value, int) and value >= 0)
            )
        }
        for item in values
        if isinstance(item, dict)
    ]


def summarize_model_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate safe per-call metrics for a run result and audit view."""

    stages = Counter(
        str(item.get("stage", "unknown"))
        for item in metrics
        if item.get("stage")
    )
    summary: dict[str, Any] = {
        "calls": len(metrics),
        "stages": dict(stages),
        "successes": sum(1 for item in metrics if item.get("success", True)),
        "failures": sum(1 for item in metrics if item.get("success") is False),
    }
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "latency_ms",
    ):
        values = [item.get(key) for item in metrics if isinstance(item.get(key), int)]
        if values:
            summary[key] = sum(values)
    retry_values = [
        item.get("retry_count")
        for item in metrics
        if isinstance(item.get("retry_count"), int)
    ]
    attempt_values = [
        item.get("attempts")
        for item in metrics
        if isinstance(item.get("attempts"), int)
    ]
    if retry_values:
        summary["retries"] = sum(retry_values)
    if attempt_values:
        summary["provider_attempts"] = sum(attempt_values)
    return summary
