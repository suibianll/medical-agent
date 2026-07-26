"""Build redacted, sequenced progress events for SSE and audit sinks."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from threading import Lock
from typing import Any, Callable


ProgressCallback = Callable[[dict[str, Any]], None]


def audit_text(value: Any, limit: int = 240) -> str:
    compact = " ".join(str(value or "").split())
    return compact if len(compact) <= limit else f"{compact[:limit]}…"


def _safe_task_event(task: Any) -> dict[str, Any] | None:
    if not isinstance(task, dict) or not isinstance(task.get("id"), int):
        return None
    return {
        "id": task["id"],
        "goal": audit_text(task.get("goal", ""), 220),
        "deps": [dep for dep in task.get("deps", []) if isinstance(dep, int)][:12],
    }


def safe_progress_event(
    event: dict[str, Any], *, run_id: str, sequence: int
) -> dict[str, Any]:
    """Whitelist public event fields and exclude raw model/private input text."""

    allowed_stages = {
        "planning",
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
    stage = str(event.get("stage", "error"))
    if stage not in allowed_stages:
        stage = "error"
    payload: dict[str, Any] = {
        "stage": stage,
        "message": audit_text(event.get("message", "执行状态已更新"), 180),
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if isinstance(event.get("task_id"), int):
        payload["task_id"] = event["task_id"]
    task = _safe_task_event(event.get("task"))
    if task is not None:
        payload["task"] = task
    raw_tasks = event.get("tasks")
    if isinstance(raw_tasks, list):
        payload["tasks"] = [
            safe_task
            for item in raw_tasks[:12]
            if (safe_task := _safe_task_event(item)) is not None
        ]
    if isinstance(event.get("status"), str):
        payload["status"] = audit_text(event["status"], 60)
    raw_queries = event.get("queries")
    if isinstance(raw_queries, list):
        payload["queries"] = [
            audit_text(query, 180)
            for query in raw_queries[:3]
            if isinstance(query, str) and query.strip()
        ]
    raw_evidence_ids = event.get("evidence_ids")
    if isinstance(raw_evidence_ids, list):
        payload["evidence_ids"] = [
            evidence_id
            for evidence_id in raw_evidence_ids[:24]
            if isinstance(evidence_id, str) and evidence_id[:1] in {"P", "K"}
        ]
    raw_facts = event.get("facts")
    if isinstance(raw_facts, list):
        payload["facts"] = [
            {"ref": fact["ref"], "summary": audit_text(fact.get("summary", ""), 220)}
            for fact in raw_facts[:8]
            if isinstance(fact, dict)
            and isinstance(fact.get("ref"), str)
            and fact["ref"][:1] in {"P", "K"}
        ]
    raw_claims = event.get("claims")
    if isinstance(raw_claims, list):
        claims: list[dict[str, Any]] = []
        for claim in raw_claims[:5]:
            if not isinstance(claim, dict):
                continue
            refs = [
                ref
                for ref in claim.get("refs", [])
                if isinstance(ref, str) and ref[:1] in {"P", "K"}
            ][:12]
            claims.append(
                {"refs": refs, "summary": audit_text(claim.get("summary", ""), 240)}
            )
        payload["claims"] = claims
    raw_evaluation = event.get("evaluation")
    if isinstance(raw_evaluation, dict):
        issue_codes = [
            audit_text(code, 80)
            for code in raw_evaluation.get("issue_codes", [])[:12]
            if isinstance(code, str)
        ]
        payload["evaluation"] = {
            "pass": bool(raw_evaluation.get("pass", False)),
            "issue_count": int(raw_evaluation.get("issue_count", 0)),
            "judgement_count": int(raw_evaluation.get("judgement_count", 0)),
            "issue_codes": list(dict.fromkeys(issue_codes)),
        }
    if isinstance(event.get("round"), int):
        payload["round"] = event["round"]
    if isinstance(event.get("counts"), dict):
        payload["counts"] = {
            key: int(value)
            for key, value in event["counts"].items()
            if key in {"tasks", "claims", "evidence"} and isinstance(value, int)
        }
    return payload


def make_progress_emitter(
    callback: ProgressCallback | None,
    run_id: str,
    audit_callback: ProgressCallback | None = None,
) -> ProgressCallback:
    """Serialize concurrent events and isolate best-effort observers."""

    if not callable(callback) and not callable(audit_callback):
        return lambda _event: None
    callback_lock = Lock()
    sequence = count(1)

    def emit(event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        with callback_lock:
            payload = safe_progress_event(
                event, run_id=run_id, sequence=next(sequence)
            )
            if audit_callback is not None:
                try:
                    audit_callback(payload)
                except Exception:  # noqa: BLE001 - audit is best effort
                    pass
            if callable(callback):
                try:
                    callback(payload)
                except Exception:  # noqa: BLE001 - monitoring is best effort
                    return

    return emit
