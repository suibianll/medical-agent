"""Validation for the intentionally small model-facing task-plan schema."""

from __future__ import annotations

from typing import Any

MAX_TASKS = 6
MAX_DEPS_PER_TASK = 3
MAX_GOAL_LENGTH = 180
EVIDENCE_SCOPES = {"patient", "knowledge", "both", "none"}
ANALYSIS_MODES = {"retrieval", "risk_review", "analysis", "synthesis"}


def build_fallback_plan(*, has_patient_record: bool) -> dict[str, Any]:
    """Build a conservative, deterministic plan when model planning fails.

    The fallback intentionally has no user-controlled text in task goals.  It
    keeps the workflow useful for evidence retrieval and synthesis while
    avoiding speculative clinical actions or model-authored graph structure.
    """

    if not has_patient_record:
        return {
            "tasks": [
                {
                    "id": 1,
                    "goal": "检索与用户问题相关的医学知识依据",
                    "deps": [],
                    "evidence_scope": "knowledge",
                    "analysis_mode": "retrieval",
                },
                {
                    "id": 2,
                    "goal": "基于已检索证据形成审慎的可追溯回答",
                    "deps": [1],
                    "evidence_scope": "knowledge",
                    "analysis_mode": "synthesis",
                },
            ]
        }
    return {
        "tasks": [
            {
                "id": 1,
                "goal": "提取与用户请求相关的患者事实",
                "deps": [],
                "evidence_scope": "patient",
                "analysis_mode": "retrieval",
            },
            {
                "id": 2,
                "goal": "检索与患者情况相关的医学知识依据",
                "deps": [1],
                "evidence_scope": "knowledge",
                "analysis_mode": "retrieval",
            },
            {
                "id": 3,
                "goal": "基于患者事实和医学依据形成需复核的回答",
                "deps": [1, 2],
                "evidence_scope": "both",
                "analysis_mode": "synthesis",
            },
        ]
    }


def _issue(code: str, message: str, task_id: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if task_id is not None:
        item["task_id"] = task_id
    return item


def validate_plan(plan: Any) -> dict[str, Any]:
    """Validate and normalize a DAG plan.

    The only model-authored fields are ``id``, ``goal`` and ``deps``.  IDs must
    be consecutive and dependencies may point only to earlier IDs.  The latter
    constraint makes cycles impossible without asking a weaker model to reason
    about graph theory.
    """

    if not isinstance(plan, dict):
        return {
            "valid": False,
            "errors": [_issue("PLAN_NOT_OBJECT", "计划必须包含 tasks 数组。")],
            "tasks": [],
        }

    raw_tasks = plan.get("tasks")
    if not isinstance(raw_tasks, list):
        return {
            "valid": False,
            "errors": [_issue("TASKS_NOT_LIST", "tasks 必须是数组。")],
            "tasks": [],
        }

    if not 1 <= len(raw_tasks) <= MAX_TASKS:
        return {
            "valid": False,
            "errors": [
                _issue(
                    "TASK_COUNT_OUT_OF_RANGE",
                    f"任务数量必须在 1 到 {MAX_TASKS} 之间。",
                )
            ],
            "tasks": [],
        }

    tasks: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []

    for index, raw_task in enumerate(raw_tasks, start=1):
        error_count_before_task = len(validation_errors)
        if not isinstance(raw_task, dict):
            validation_errors.append(
                _issue("TASK_NOT_OBJECT", "每个任务必须是对象。", index)
            )
            continue

        task_id = raw_task.get("id")
        goal = raw_task.get("goal")
        deps = raw_task.get("deps", [])
        evidence_scope = raw_task.get("evidence_scope")
        analysis_mode = raw_task.get("analysis_mode")

        if not isinstance(task_id, int) or isinstance(task_id, bool):
            validation_errors.append(
                _issue("TASK_ID_NOT_INTEGER", "任务 id 必须是整数。", index)
            )
        elif task_id != index:
            validation_errors.append(
                _issue(
                    "TASK_ID_NOT_CONTINUOUS",
                    "任务 id 必须从 1 开始连续递增。",
                    task_id,
                )
            )

        if not isinstance(goal, str) or not goal.strip():
            validation_errors.append(
                _issue("TASK_GOAL_INVALID", "任务 goal 必须是非空字符串。", index)
            )
        elif len(goal.strip()) > MAX_GOAL_LENGTH:
            validation_errors.append(
                _issue(
                    "TASK_GOAL_TOO_LONG",
                    f"任务 goal 不能超过 {MAX_GOAL_LENGTH} 个字符。",
                    task_id if isinstance(task_id, int) else index,
                )
            )

        if "evidence_scope" in raw_task and evidence_scope not in EVIDENCE_SCOPES:
            validation_errors.append(
                _issue(
                    "EVIDENCE_SCOPE_INVALID",
                    "evidence_scope 必须是 patient、knowledge、both 或 none。",
                    task_id if isinstance(task_id, int) else index,
                )
            )
        if "analysis_mode" in raw_task and analysis_mode not in ANALYSIS_MODES:
            validation_errors.append(
                _issue(
                    "ANALYSIS_MODE_INVALID",
                    "analysis_mode 必须是 retrieval、risk_review、analysis 或 synthesis。",
                    task_id if isinstance(task_id, int) else index,
                )
            )

        if not isinstance(deps, list):
            validation_errors.append(
                _issue("DEPS_NOT_LIST", "deps 必须是数组。", index)
            )
            deps = []
        elif len(deps) > MAX_DEPS_PER_TASK:
            validation_errors.append(
                _issue(
                    "TOO_MANY_DEPENDENCIES",
                    f"每个任务最多依赖 {MAX_DEPS_PER_TASK} 个上游任务。",
                    task_id if isinstance(task_id, int) else index,
                )
            )

        clean_deps: list[int] = []
        if isinstance(deps, list):
            for dep in deps:
                if not isinstance(dep, int) or isinstance(dep, bool):
                    validation_errors.append(
                        _issue("DEPENDENCY_NOT_INTEGER", "deps 中只能包含整数。", index)
                    )
                    continue
                clean_deps.append(dep)

            if len(set(clean_deps)) != len(clean_deps):
                validation_errors.append(
                    _issue("DEPENDENCY_DUPLICATE", "deps 中不能有重复任务。", index)
                )

            for dep in clean_deps:
                if dep < 1 or dep >= index:
                    validation_errors.append(
                        _issue(
                            "INVALID_DEPENDENCY",
                            "依赖只能引用当前任务之前、且真实存在的任务 id。",
                            task_id if isinstance(task_id, int) else index,
                        )
                    )

        # Do not construct a normalized half-task after any field-level error.
        # The final invalid response already hides tasks, but keeping this
        # intermediate list clean prevents future callers from misusing it.
        if (
            len(validation_errors) == error_count_before_task
            and isinstance(task_id, int)
            and isinstance(goal, str)
        ):
            normalized_task: dict[str, Any] = {
                "id": task_id,
                "goal": goal.strip(),
                "deps": clean_deps,
            }
            if "evidence_scope" in raw_task:
                normalized_task["evidence_scope"] = evidence_scope
            if "analysis_mode" in raw_task:
                normalized_task["analysis_mode"] = analysis_mode
            tasks.append(normalized_task)

    if validation_errors:
        return {"valid": False, "errors": validation_errors, "tasks": []}

    return {"valid": True, "errors": [], "tasks": tasks}
