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
    "PARTIAL_SUPPORT": "拆分结论并删除未被证据覆盖的表述",
    "CONTRADICTED": "优先处理冲突证据，禁止继续输出相反结论",
    "INSUFFICIENT_EVIDENCE": "补充直接证据或转人工审核，不要用常识填空",
    "TASK_EXECUTION_FAILED": "重试失败任务并重新执行其下游任务",
    "TASK_BLOCKED": "先修复被阻塞任务的上游失败原因",
    "TASK_OUTPUT_MISSING": "重新执行终端任务；若仍无结论则转人工审核",
}


def _source_task_ids(tasks: list[dict[str, Any]], evidence_kind: str) -> set[int]:
    """Find retrieval roots from the plan's structured evidence scope."""

    expected_scope = "patient" if evidence_kind == "patient" else "knowledge"
    result: set[int] = set()
    for task in tasks:
        if task.get("evidence_scope") in {expected_scope, "both"}:
            result.add(task["id"])
    return result


def build_repair_plan(
    *,
    issues: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_to_task = {claim.get("id"): claim.get("task_id") for claim in claims}
    failed_roots: set[int | None] = set()
    patient_source_tasks = _source_task_ids(tasks, "patient")
    knowledge_source_tasks = _source_task_ids(tasks, "knowledge")
    for issue in issues:
        task_id = issue.get("task_id", claim_to_task.get(issue.get("claim")))
        if isinstance(task_id, int):
            failed_roots.add(task_id)
        # Missing evidence types cannot reliably be fixed by rerunning only
        # the downstream synthesis task.  Reopen the matching retrieval roots
        # and then rerun their descendants with an explicit repair directive.
        if issue.get("code") == "MISSING_PATIENT_REF" and patient_source_tasks:
            failed_roots.update(patient_source_tasks)
        if issue.get("code") == "MISSING_KB_REF" and knowledge_source_tasks:
            failed_roots.update(knowledge_source_tasks)
    failed_roots.discard(None)
    affected = descendants(tasks, set(failed_roots)) if failed_roots else set()
    issue_codes = sorted(
        {
            str(issue.get("code"))
            for issue in issues
            if issue.get("code") in REPAIR_ACTIONS
        }
    )
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
        "task_directives": [
            {"task_id": task_id, "codes": issue_codes}
            for task_id in sorted(affected)
        ],
        "actions": actions,
        "repairable": bool(affected)
        and all(issue.get("code") in REPAIR_ACTIONS for issue in issues),
    }
