"""Short-lived in-memory run results plus a redacted audit timeline.

This module deliberately has no filesystem, logging, or model-provider
integration.  A full result is kept only so a local user can reopen the
evidence view during the configured process lifetime.  The separately exposed
audit timeline is aggressively redacted and never contains patient-record
text, prompt/history text, raw model responses, or hidden reasoning.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
from threading import RLock
from time import monotonic
from typing import Any


DEFAULT_MAX_RUNS = 24
DEFAULT_TTL_SECONDS = 30 * 60
DEFAULT_MAX_EVENTS_PER_RUN = 240
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_SAFE_STAGES = {
    "planning",
    "model_call",
    "task_started",
    "task_completed",
    "query",
    "retrieve",
    "extracting",
    "extract",
    "synthesizing",
    "synthesize",
    "evaluate",
    "repair",
    "completed",
    "error",
}
_STAGE_MESSAGES = {
    "planning": "任务规划状态已更新",
    "model_call": "模型调用成本状态已更新",
    "task_started": "子任务已开始",
    "task_completed": "子任务已完成",
    "query": "正在生成检索查询",
    "retrieve": "检索阶段已更新",
    "extracting": "正在提取关键信息",
    "extract": "关键信息提取已完成",
    "synthesizing": "正在生成带引用结论",
    "synthesize": "带引用结论已生成",
    "evaluate": "证据评估状态已更新",
    "repair": "证据修正状态已更新",
    "completed": "运行已完成",
    "error": "运行出现可审计错误",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_code(value: Any, *, fallback: str = "unknown") -> str:
    candidate = str(value or "")
    return candidate if _SAFE_CODE.fullmatch(candidate) else fallback


def _safe_evidence_ids(values: Any, limit: int = 24) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values[:limit]:
        if isinstance(value, str) and re.fullmatch(r"[PK]\d{1,6}", value):
            if value not in result:
                result.append(value)
    return result


def _safe_task(task: Any) -> dict[str, Any] | None:
    if not isinstance(task, dict) or not isinstance(task.get("id"), int):
        return None
    return {
        "id": task["id"],
        "deps": [dep for dep in task.get("deps", []) if isinstance(dep, int)][:12],
    }


def redact_audit_event(event: Any) -> dict[str, Any] | None:
    """Keep process facts and identifiers, never patient/model prose."""

    if not isinstance(event, dict):
        return None
    run_id = _safe_code(event.get("run_id"), fallback="")
    if not run_id:
        return None
    stage = str(event.get("stage", "error"))
    if stage not in _SAFE_STAGES:
        stage = "error"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        # Use only server-authored labels; do not copy an event message.
        "message": _STAGE_MESSAGES[stage],
    }
    if isinstance(event.get("sequence"), int):
        payload["sequence"] = event["sequence"]
    if isinstance(event.get("timestamp"), str):
        payload["timestamp"] = event["timestamp"][:40]
    status = _safe_code(event.get("status"), fallback="")
    if status:
        payload["status"] = status
    if isinstance(event.get("task_id"), int):
        payload["task_id"] = event["task_id"]
    task = _safe_task(event.get("task"))
    if task is not None:
        payload["task"] = task
    if isinstance(event.get("tasks"), list):
        payload["tasks"] = [
            safe_task
            for item in event["tasks"][:12]
            if (safe_task := _safe_task(item)) is not None
        ]

    evidence_ids = _safe_evidence_ids(event.get("evidence_ids"))
    if evidence_ids:
        payload["evidence_ids"] = evidence_ids

    raw_facts = event.get("facts")
    if isinstance(raw_facts, list):
        payload["fact_count"] = len(raw_facts[:8])
        payload["fact_refs"] = _safe_evidence_ids(
            [item.get("ref") for item in raw_facts if isinstance(item, dict)], 8
        )

    raw_claims = event.get("claims")
    if isinstance(raw_claims, list):
        payload["claim_count"] = len(raw_claims[:5])
        payload["claim_refs"] = [
            _safe_evidence_ids(item.get("refs"), 12)
            for item in raw_claims[:5]
            if isinstance(item, dict)
        ]

    raw_evaluation = event.get("evaluation")
    if isinstance(raw_evaluation, dict):
        issue_codes = [
            _safe_code(code, fallback="")
            for code in raw_evaluation.get("issue_codes", [])[:12]
            if isinstance(code, str)
        ]
        payload["evaluation"] = {
            "pass": bool(raw_evaluation.get("pass", False)),
            "issue_count": max(0, int(raw_evaluation.get("issue_count", 0))),
            "judgement_count": max(0, int(raw_evaluation.get("judgement_count", 0))),
            "issue_codes": [code for code in issue_codes if code],
        }
    if isinstance(event.get("round"), int):
        payload["round"] = event["round"]
    raw_metrics = event.get("metrics")
    if isinstance(raw_metrics, dict):
        metric: dict[str, Any] = {}
        for key in (
            "stage",
            "provider",
            "model",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "success",
        ):
            value = raw_metrics.get(key)
            if isinstance(value, (str, bool)) or (
                isinstance(value, int) and value >= 0
            ):
                metric[key] = value
        if metric:
            payload["metrics"] = metric
    if isinstance(event.get("counts"), dict):
        payload["counts"] = {
            key: value
            for key, value in event["counts"].items()
            if key in {"tasks", "claims", "evidence"}
            and isinstance(value, int)
            and value >= 0
        }
    return payload


class InMemoryRunArchive:
    """Bounded, TTL-limited process-memory archive for local run navigation."""

    def __init__(
        self,
        *,
        max_runs: int = DEFAULT_MAX_RUNS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_events_per_run: int = DEFAULT_MAX_EVENTS_PER_RUN,
    ) -> None:
        if max_runs < 1:
            raise ValueError("max_runs must be positive")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        if max_events_per_run < 1:
            raise ValueError("max_events_per_run must be positive")
        self.max_runs = max_runs
        self.ttl_seconds = ttl_seconds
        self.max_events_per_run = max_events_per_run
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = RLock()

    def _purge_expired_locked(self) -> None:
        now = monotonic()
        expired = [
            run_id
            for run_id, entry in self._entries.items()
            if entry["expires_monotonic"] <= now
        ]
        for run_id in expired:
            self._entries.pop(run_id, None)

    def _trim_locked(self) -> None:
        while len(self._entries) > self.max_runs:
            self._entries.popitem(last=False)

    def _metadata_locked(self, run_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": run_id,
            "state": entry["state"],
            "result_status": entry.get("result_status"),
            "created_at": entry["created_at"],
            "stored_at": entry["stored_at"],
            "expires_at": entry["expires_at"],
            "event_count": len(entry["events"]),
            "full_result_available": entry["result"] is not None,
            # Explicit privacy/lifetime boundary for callers reopening a run.
            "scope": "process_memory_only",
            "persistence": "none",
            "audit_events": "redacted",
        }

    def start(self, run_id: str, created_at: str | None = None) -> dict[str, Any]:
        """Open an archive slot before a run starts emitting audit events."""

        safe_run_id = _safe_code(run_id, fallback="")
        if not safe_run_id:
            raise ValueError("invalid run id")
        now_wall = _utc_now()
        stored_at = now_wall.isoformat()
        expires_at = (now_wall + timedelta(seconds=self.ttl_seconds)).isoformat()
        with self._lock:
            self._purge_expired_locked()
            self._entries.pop(safe_run_id, None)
            self._entries[safe_run_id] = {
                "created_at": created_at or stored_at,
                "stored_at": stored_at,
                "expires_at": expires_at,
                "expires_monotonic": monotonic() + self.ttl_seconds,
                "state": "running",
                "result_status": None,
                "result": None,
                "events": [],
            }
            self._trim_locked()
            return self._metadata_locked(safe_run_id, self._entries[safe_run_id])

    def append_event(self, event: dict[str, Any]) -> None:
        """Append a redacted audit event; unknown/expired runs are ignored."""

        safe_event = redact_audit_event(event)
        if safe_event is None:
            return
        run_id = safe_event["run_id"]
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(run_id)
            if entry is None:
                return
            events = entry["events"]
            events.append(safe_event)
            if len(events) > self.max_events_per_run:
                del events[: len(events) - self.max_events_per_run]

    def finalize(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Keep the complete, validated response only for this process/TTL."""

        run = result.get("run") if isinstance(result, dict) else None
        run_id = _safe_code((run or {}).get("id"), fallback="")
        if not run_id:
            return None
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(run_id)
            if entry is None:
                self.start(run_id, (run or {}).get("created_at"))
                entry = self._entries.get(run_id)
            if entry is None:  # defensive; start above validates and inserts
                return None
            entry["result"] = deepcopy(result)
            entry["state"] = "completed"
            entry["result_status"] = _safe_code(result.get("status"), fallback="unknown")
            self._entries.move_to_end(run_id)
            return self._metadata_locked(run_id, entry)

    def mark_failed(self, run_id: str) -> None:
        """Retain only safe run state/events if a provider call raises."""

        safe_run_id = _safe_code(run_id, fallback="")
        if not safe_run_id:
            return
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(safe_run_id)
            if entry is not None:
                entry["state"] = "failed"
                entry["result_status"] = "failed"

    def get_result(self, run_id: str) -> dict[str, Any] | None:
        """Return a deepcopy of the short-lived full result plus boundary metadata."""

        safe_run_id = _safe_code(run_id, fallback="")
        if not safe_run_id:
            return None
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(safe_run_id)
            if entry is None:
                return None
            metadata = self._metadata_locked(safe_run_id, entry)
            if entry["result"] is None:
                return {
                    "status": entry.get("result_status") or entry["state"],
                    "run": {"id": safe_run_id, "created_at": entry["created_at"]},
                    "archive": metadata,
                }
            result = deepcopy(entry["result"])
            result["archive"] = metadata
            return result

    def get_events(self, run_id: str) -> dict[str, Any] | None:
        """Return only the redacted audit event timeline for one archived run."""

        safe_run_id = _safe_code(run_id, fallback="")
        if not safe_run_id:
            return None
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(safe_run_id)
            if entry is None:
                return None
            return {
                "run_id": safe_run_id,
                "events": deepcopy(entry["events"]),
                "archive": self._metadata_locked(safe_run_id, entry),
            }
