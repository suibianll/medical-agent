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
) -> dict[str, Any]:
    """Create DAG, evidence-support and report-provenance graph edges."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evidence_by_id = {item["id"]: item for item in evidence}
    claim_relations: dict[str, dict[str, list[str]]] = {}
    relation_counts: dict[str, int] = {
        "supports": 0,
        "qualifies": 0,
        "contradicts": 0,
        "uncertain": 0,
    }

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
        claim_node = {
            "id": claim_id,
            "type": "claim",
            "label": claim["text"],
            "status": claim.get("status", "unknown"),
            "support_summary": {
                "supports": 0,
                "qualifies": 0,
                "contradicts": 0,
                "uncertain": 0,
                "has_conflict": False,
            },
        }
        _append_unique(
            nodes,
            claim_node,
        )
        edges.append(
            {
                "id": f"T{claim['task_id']}-{claim_id}",
                "from": f"T{claim['task_id']}",
                "to": claim_id,
                "type": "produces",
            }
        )
        support_edges = claim.get("support_edges", [])
        edge_items = support_edges if isinstance(support_edges, list) and support_edges else [
            {"evidence_id": evidence_id, "relation": "supports"}
            for evidence_id in claim.get("refs", [])
        ]
        for support_edge in edge_items:
            if not isinstance(support_edge, dict):
                continue
            evidence_id = support_edge.get("evidence_id")
            if not isinstance(evidence_id, str):
                continue
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
                    "locator": evidence_item.get("locator", ""),
                    "span_id": (
                        evidence_item.get("span", {}).get("id")
                        if isinstance(evidence_item.get("span"), dict)
                        else None
                    ),
                },
            )
            relation = str(support_edge.get("relation", "supports"))
            edge_type = {
                "supports": "supports",
                "contradicts": "contradicts",
                "qualifies": "qualifies",
                "uncertain": "uncertain",
            }.get(relation, "supports")
            claim_relations.setdefault(claim_id, {}).setdefault(edge_type, []).append(
                evidence_id
            )
            claim_node["support_summary"][edge_type] += 1
            relation_counts[edge_type] += 1
            edges.append(
                {
                    # Keep the stable legacy edge ID; relation is now carried
                    # as a first-class field so old consumers remain valid.
                    "id": f"{evidence_id}-{claim_id}",
                    "from": evidence_id,
                    "to": claim_id,
                    "type": edge_type,
                    "relation": edge_type,
                    "span_id": support_edge.get("span_id"),
                    "verifier": support_edge.get("verifier"),
                    "verifier_score": support_edge.get("verifier_score"),
                    "status": support_edge.get("status"),
                }
            )

        claim_node["support_summary"]["has_conflict"] = bool(
            claim_node["support_summary"]["contradicts"]
            and (
                claim_node["support_summary"]["supports"]
                or claim_node["support_summary"]["qualifies"]
            )
        )

    conflict_claim_ids: list[str] = []
    for claim_id, relations in claim_relations.items():
        positive_ids = list(
            dict.fromkeys(relations.get("supports", []) + relations.get("qualifies", []))
        )
        contradictory_ids = list(dict.fromkeys(relations.get("contradicts", [])))
        if not positive_ids or not contradictory_ids:
            continue
        conflict_claim_ids.append(claim_id)
        for positive_id in positive_ids:
            for contradictory_id in contradictory_ids:
                edges.append(
                    {
                        "id": f"{positive_id}-{contradictory_id}-{claim_id}-conflict",
                        "from": positive_id,
                        "to": contradictory_id,
                        "type": "conflict",
                        "relation": "conflicts",
                        "claim_id": claim_id,
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

    claim_status_counts: dict[str, int] = {}
    claims_without_support: list[str] = []
    contradiction_claim_ids = [
        claim_id
        for claim_id, relations in claim_relations.items()
        if relations.get("contradicts")
    ]
    uncertain_claim_ids = [
        claim_id for claim_id, relations in claim_relations.items() if relations.get("uncertain")
    ]
    for claim in claims:
        status = str(claim.get("status", "unknown"))
        claim_status_counts[status] = claim_status_counts.get(status, 0) + 1
        if not claim_relations.get(claim.get("id", "")):
            claims_without_support.append(str(claim.get("id", "")))
    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "relation_counts": relation_counts,
            "conflict_claims": conflict_claim_ids,
            "conflict_count": len(conflict_claim_ids),
            "contradiction_claims": contradiction_claim_ids,
            "uncertain_claims": uncertain_claim_ids,
            "claim_status_counts": claim_status_counts,
            "claims_without_support": claims_without_support,
        },
    }
