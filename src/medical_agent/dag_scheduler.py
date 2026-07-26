"""Small, dependency-aware task scheduler for synchronous task workers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Callable


def descendants(tasks: list[dict[str, Any]], roots: set[int] | list[int]) -> set[int]:
    """Return all transitive descendants of the given task IDs."""

    affected = set(roots)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            task_id = task["id"]
            if task_id not in affected and any(dep in affected for dep in task["deps"]):
                affected.add(task_id)
                changed = True
    return affected


def execute_dag(
    tasks: list[dict[str, Any]],
    worker: Callable[..., Any],
    *,
    max_workers: int = 3,
    initial_task_states: dict[int, dict[str, Any]] | None = None,
    rerun_task_ids: set[int] | None = None,
    on_status: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Execute all dependency-ready tasks and block descendants after failures.

    ``initial_task_states`` and ``rerun_task_ids`` support targeted repair:
    unaffected completed branches remain locked while a failed task and its
    descendants are scheduled again.
    """

    if max_workers < 1:
        raise ValueError("max_workers 必须大于 0")

    ordered_tasks = sorted(tasks, key=lambda item: item["id"])
    task_by_id = {task["id"]: task for task in ordered_tasks}
    if len(task_by_id) != len(ordered_tasks):
        raise ValueError("任务 id 不能重复")

    states: dict[int, dict[str, Any]] = {}
    for task in ordered_tasks:
        task_id = task["id"]
        prior = (initial_task_states or {}).get(task_id)
        should_rerun = rerun_task_ids is not None and task_id in rerun_task_ids
        if prior and prior.get("status") == "completed" and not should_rerun:
            states[task_id] = deepcopy(prior)
            states[task_id]["task"] = task
        else:
            states[task_id] = {"task": task, "status": "pending"}

    waves: list[list[int]] = []
    while True:
        pending = [state for state in states.values() if state["status"] == "pending"]
        if not pending:
            break

        # A task with a failed prerequisite is never run with partial context.
        blocked_ids: list[int] = []
        for task_id, state in states.items():
            if state["status"] != "pending":
                continue
            failed_dependencies = [
                dep
                for dep in state["task"]["deps"]
                if states[dep]["status"] in {"failed", "blocked"}
            ]
            if failed_dependencies:
                state["status"] = "blocked"
                state["error"] = f"上游任务失败或被阻塞：{failed_dependencies}"
                blocked_ids.append(task_id)
                if on_status:
                    on_status(task_id, "blocked")

        if blocked_ids:
            continue

        ready_ids = [
            task_id
            for task_id, state in states.items()
            if state["status"] == "pending"
            and all(states[dep]["status"] == "completed" for dep in state["task"]["deps"])
        ]
        if not ready_ids:
            raise RuntimeError("DAG 无可执行任务；请先校验任务依赖关系。")

        waves.append(ready_ids[:])
        for task_id in ready_ids:
            states[task_id]["status"] = "running"
            if on_status:
                on_status(task_id, "running")

        def submit_task(task_id: int) -> tuple[int, Any]:
            task = task_by_id[task_id]
            upstream = {
                dep: states[dep].get("result") for dep in task["deps"]
            }
            return task_id, worker(task, upstream)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(ready_ids))) as pool:
            future_map = {pool.submit(submit_task, task_id): task_id for task_id in ready_ids}
            for future in as_completed(future_map):
                task_id = future_map[future]
                try:
                    returned_task_id, result = future.result()
                    states[returned_task_id]["status"] = "completed"
                    states[returned_task_id]["result"] = result
                    if on_status:
                        on_status(returned_task_id, "completed")
                except Exception as exc:  # noqa: BLE001 - scheduler records worker failures
                    states[task_id]["status"] = "failed"
                    states[task_id]["error"] = str(exc)
                    if on_status:
                        on_status(task_id, "failed")

    return {"tasks": states, "waves": waves}
