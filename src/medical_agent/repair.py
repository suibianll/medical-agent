"""Fixed, code-owned mapping from evaluation failures to targeted reruns."""

from __future__ import annotations

from typing import Any

from .dag_scheduler import descendants


REPAIR_ACTIONS = {
    "NO_REF": "重新检索并补充引用，或删除无依据结论",
    "BAD_REF": "重新选择已注册的证据编号",
    "MISSING_PATIENT_REF": "重新抽取患者病历事实",
    "MISSING_KB_REF": "重新检索医学知识库",
    "NOT_SUPPORTED": "缩小结论范围或检索更直接的证据",
    "TASK_EXECUTION_FAILED": "重试失败任务并重新执行其下游任务",
    "TASK_BLOCKED": "先修复被阻塞任务的上游失败原因",
}


def build_repair_plan(
    *,
    issues: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_to_task = {claim.get("id"): claim.get("task_id") for claim in claims}
    failed_roots = {
        issue.get("task_id", claim_to_task.get(issue.get("claim"))) for issue in issues
    }
    failed_roots.discard(None)
    affected = descendants(tasks, set(failed_roots)) if failed_roots else set()
    actions = [
        {
            "claim": issue.get("claim"),
            "code": issue.get("code"),
            "action": REPAIR_ACTIONS.get(issue.get("code"), "转人工审核"),
        }
        for issue in issues
    ]
    return {
        "root_tasks": sorted(failed_roots),
        "rerun_tasks": sorted(affected),
        "actions": actions,
        "repairable": bool(affected)
        and all(issue.get("code") in REPAIR_ACTIONS for issue in issues),
    }
