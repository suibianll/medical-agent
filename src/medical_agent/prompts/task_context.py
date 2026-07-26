"""Minimal task projection shared by model prompts."""

from __future__ import annotations

from typing import Any


def task_prompt_view(task: dict[str, Any]) -> dict[str, Any]:
    """Drop runtime-only task fields before sending a task to a model."""

    view: dict[str, Any] = {
        "id": task.get("id"),
        "goal": " ".join(str(task.get("goal", "")).split())[:300],
        "deps": [dep for dep in task.get("deps", []) if isinstance(dep, int)][:8],
    }
    if "patient_grounding_required" in task:
        view["patient_grounding_required"] = bool(
            task.get("patient_grounding_required")
        )
    repair = task.get("repair")
    if isinstance(repair, dict):
        codes = [
            code
            for code in repair.get("codes", [])
            if isinstance(code, str) and code
        ][:8]
        guidance = [
            item
            for item in repair.get("guidance", [])
            if isinstance(item, str) and item
        ][:4]
        if codes or guidance:
            view["repair"] = {"codes": codes, "guidance": guidance}
    return view
