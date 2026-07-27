from __future__ import annotations

import unittest

from medical_agent.graph import build_evidence_graph


class EvidenceGraphTests(unittest.TestCase):
    def test_conflicting_support_edges_are_first_class_graph_relationships(self) -> None:
        graph = build_evidence_graph(
            tasks=[{"id": 1, "goal": "review evidence", "deps": []}],
            task_states={1: {"status": "completed"}},
            claims=[
                {
                    "id": "C1",
                    "task_id": 1,
                    "text": "conflicting conclusion",
                    "status": "needs_repair",
                    "support_edges": [
                        {
                            "evidence_id": "K1",
                            "relation": "supports",
                            "verifier": "semantic_judge",
                            "verifier_score": 1.0,
                        },
                        {
                            "evidence_id": "K2",
                            "relation": "contradicts",
                            "verifier": "semantic_judge",
                            "verifier_score": 0.0,
                        },
                    ],
                    "refs": ["K1", "K2"],
                }
            ],
            evidence=[
                {"id": "K1", "kind": "knowledge", "source": "Guideline A"},
                {"id": "K2", "kind": "knowledge", "source": "Guideline B"},
            ],
        )

        relation_edges = {
            (edge["from"], edge["to"]): edge["type"]
            for edge in graph["edges"]
            if edge.get("from") in {"K1", "K2"}
        }
        self.assertEqual(relation_edges[("K1", "C1")], "supports")
        self.assertEqual(relation_edges[("K2", "C1")], "contradicts")
        conflict_edges = [edge for edge in graph["edges"] if edge["type"] == "conflict"]
        self.assertEqual(len(conflict_edges), 1)
        self.assertEqual(conflict_edges[0]["claim_id"], "C1")
        self.assertEqual(graph["summary"]["conflict_claims"], ["C1"])
        self.assertEqual(graph["summary"]["relation_counts"]["contradicts"], 1)
        claim_node = next(node for node in graph["nodes"] if node["id"] == "C1")
        self.assertTrue(claim_node["support_summary"]["has_conflict"])

    def test_legacy_refs_still_create_support_edges_and_summary(self) -> None:
        graph = build_evidence_graph(
            tasks=[{"id": 1, "goal": "retrieve", "deps": []}],
            task_states={1: {"status": "completed"}},
            claims=[{"id": "C1", "task_id": 1, "text": "supported", "refs": ["K1"]}],
            evidence=[{"id": "K1", "kind": "knowledge", "source": "Guideline"}],
        )

        self.assertIn("K1-C1", {edge["id"] for edge in graph["edges"]})
        self.assertEqual(graph["summary"]["relation_counts"]["supports"], 1)
        self.assertEqual(graph["summary"]["conflict_count"], 0)


if __name__ == "__main__":
    unittest.main()
