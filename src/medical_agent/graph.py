"""Build a compact, UI-neutral evidence graph."""

from __future__ import annotations

from typing import Any


def _append_unique(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if not any(existing["id"] == item["id"] for existing in items):
        items.append(item)


def build_evidence_graph(
    *,
    tasks: list[dict[str, Any]],
    task_states: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Create DAG, evidence-support and report-provenance graph edges."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evidence_by_id = {item["id"]: item for item in evidence}

    for task in tasks:
        state = task_states.get(task["id"], {})
        _append_unique(
            nodes,
            {
                "id": f"T{task['id']}",
                "type": "task",
                "label": f"T{task['id']}：{task['goal']}",
                "status": state.get("status", "pending"),
            },
        )
        # Preserve the validated execution DAG in the same visual payload as
        # the evidence chain.  This lets a reviewer distinguish an upstream
        # task dependency from an evidence-to-claim support relation.
        for dependency_id in task.get("deps", []):
            edges.append(
                {
                    "id": f"T{dependency_id}-T{task['id']}-depends_on",
                    "from": f"T{dependency_id}",
                    "to": f"T{task['id']}",
                    "type": "depends_on",
                }
            )

    for claim in claims:
        claim_id = claim["id"]
        _append_unique(
            nodes,
            {
                "id": claim_id,
                "type": "claim",
                "label": claim["text"],
                "status": claim.get("status", "unknown"),
            },
        )
        edges.append(
            {
                "id": f"T{claim['task_id']}-{claim_id}",
                "from": f"T{claim['task_id']}",
                "to": claim_id,
                "type": "produces",
            }
        )
        for evidence_id in claim.get("refs", []):
            evidence_item = evidence_by_id.get(evidence_id)
            if not evidence_item:
                continue
            _append_unique(
                nodes,
                {
                    "id": evidence_id,
                    "type": "patient_evidence"
                    if evidence_item["kind"] == "patient"
                    else "knowledge_evidence",
                    "label": f"{evidence_id}：{evidence_item['source']}",
                },
            )
            edges.append(
                {
                    "id": f"{evidence_id}-{claim_id}",
                    "from": evidence_id,
                    "to": claim_id,
                    "type": "supports",
                }
            )

    _append_unique(nodes, {"id": "REPORT", "type": "report", "label": "最终报告"})
    for claim in claims:
        edges.append(
            {
                "id": f"{claim['id']}-REPORT",
                "from": claim["id"],
                "to": "REPORT",
                "type": "reported_in",
            }
        )

    return {"nodes": nodes, "edges": edges}
