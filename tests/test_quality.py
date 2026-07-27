from __future__ import annotations

import unittest

from medical_agent.bootstrap import create_agent
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.quality import (
    evaluate_evidence_chain,
    evaluate_retrieval_cases,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


class QualityMetricTests(unittest.TestCase):
    def test_rank_metrics_are_deterministic_and_bounded(self) -> None:
        ranked = ["K2", "K1", "K3"]
        relevant = ["K1", "K3"]

        self.assertEqual(recall_at_k(ranked, relevant, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(mean_reciprocal_rank(ranked, relevant), 0.5)
        self.assertGreater(ndcg_at_k(ranked, relevant, 3), 0.5)
        self.assertLessEqual(ndcg_at_k(ranked, relevant, 3), 1.0)

    def test_retrieval_benchmark_accepts_document_results_without_echoing_queries(self) -> None:
        calls: list[str] = []

        def search(query: str):
            calls.append(query)
            return [{"id": "K2"}, {"id": "K1"}]

        report = evaluate_retrieval_cases(
            [
                {"id": "case-1", "query": "private query", "relevant_ids": ["K1"]},
                {"query": "invalid", "relevant_ids": []},
            ],
            search,
            ks=(1, 2),
        )

        self.assertEqual(report["evaluated_cases"], 1)
        self.assertEqual(report["invalid_cases"], 1)
        self.assertEqual(report["metrics"]["recall@2"], 1.0)
        self.assertEqual(calls, ["private query"])
        self.assertNotIn("private query", str(report))

    def test_evidence_chain_report_measures_citations_and_dual_support(self) -> None:
        report = evaluate_evidence_chain(
            [
                {"id": "C1", "refs": ["K1"]},
                {"id": "C2", "refs": ["P1"], "requires_dual_support": True},
            ],
            [{"id": "K1"}, {"id": "P1"}],
            {
                "issues": [{"claim": "C2", "code": "MISSING_KB_REF"}],
                "support_edges": [{"claim_id": "C1", "evidence_id": "K1"}],
            },
        )

        self.assertEqual(report["claims"], 2)
        self.assertEqual(report["citation_coverage"], 1.0)
        self.assertEqual(report["citation_precision"], 1.0)
        self.assertEqual(report["support_edge_coverage"], 0.5)
        self.assertEqual(report["dual_support_coverage"], 0.0)
        self.assertEqual(report["unresolved_rate"], 0.5)
        self.assertNotIn("C2", str(report))

    def test_completed_run_exposes_safe_quality_summary(self) -> None:
        knowledge = JsonKnowledgeBase(
            [{"id": "K1", "title": "guide", "text": "quality evidence"}]
        )
        result = create_agent(knowledge_base=knowledge, max_workers=1).chat(
            message="quality evidence"
        )
        quality = result["run"]["quality"]
        self.assertIn("citation_coverage", quality)
        self.assertIn("support_edge_coverage", quality)
        self.assertNotIn("quality evidence", str(quality))


if __name__ == "__main__":
    unittest.main()
