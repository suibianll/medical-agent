"""Minimal safe runtime logger for medical-agent execution milestones.

Only pre-redacted process metadata reaches the logger.  In particular, this
module never receives or writes patient records, user requests, conversation
history, evidence/claim text, provider responses, API credentials, or hidden
reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .ports import AuditEventSink
from .run_archive import redact_audit_event

__all__ = ["AuditEventSink", "SafeAuditLogger"]


class SafeAuditLogger:
    """Emit compact JSON lines through the application's standard logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        # The server's normal logging configuration supplies the handler.  We
        # intentionally do not create a file handler here: archive/result data
        # must remain process-memory-only unless an application explicitly
        # configures its own safe logging policy.
        self._logger = logger or logging.getLogger("medical_agent.audit")

    @staticmethod
    def _summary(event: dict[str, Any]) -> dict[str, Any] | None:
        safe_event = redact_audit_event(event)
        if safe_event is None:
            return None
        reference_count = len(safe_event.get("evidence_ids", []))
        reference_count += len(safe_event.get("fact_refs", []))
        reference_count += sum(
            len(refs) for refs in safe_event.get("claim_refs", []) if isinstance(refs, list)
        )
        payload: dict[str, Any] = {
            "run_id": safe_event["run_id"],
            "stage": safe_event["stage"],
            "status": safe_event.get("status", ""),
            "task_id": safe_event.get("task_id"),
            "reference_count": reference_count,
        }
        if "round" in safe_event:
            payload["round"] = safe_event["round"]
        if "counts" in safe_event:
            payload["counts"] = safe_event["counts"]
        if "evaluation" in safe_event:
            evaluation = safe_event["evaluation"]
            payload["evaluation"] = {
                "pass": evaluation.get("pass", False),
                "issue_count": evaluation.get("issue_count", 0),
            }
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != ""
        }

    def record(self, event: dict[str, Any]) -> None:
        payload = self._summary(event)
        if payload is None:
            return
        # One JSON object per log line is easy to ship/parse while keeping the
        # data contract demonstrably free of clinical text and model output.
        self._logger.info(
            "medical_agent_audit %s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
