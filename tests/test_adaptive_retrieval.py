from __future__ import annotations

import unittest

from medical_agent.agent_pipeline import ThreeStageTaskAgent
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.evidence import EvidenceRegistry
from medical_agent.retrieval.patient import PatientRecordRetriever


class _AdaptiveModel(DemoModelAdapter):
    def __init__(self, *, evidence_on_first_round: bool) -> None:
        self.query_calls = 0
        self.evidence_on_first_round = evidence_on_first_round

    def make_queries(self, **_kwargs: object) -> dict[str, list[str]]:
        self.query_calls += 1
        if self.query_calls == 1 and not self.evidence_on_first_round:
            return {"queries": ["missing evidence"]}
        return {"queries": ["known evidence"]}


class _RoundKnowledge:
    def __init__(self) -> None:
        self.search_many_calls = 0

    def search_many(self, queries, **_kwargs):
        self.search_many_calls += 1
        if "known evidence" not in queries:
            return []
        return [
            {
                "id": "adaptive-doc",
                "title": "Adaptive guideline",
                "text": "The adaptive retrieval round found this evidence.",
                "locator": "section 1",
                "score": 1.0,
                "version": "2026.1",
                "source_type": "guideline",
            }
        ]

    def search(self, query: str, limit: int = 4):
        return self.search_many([query], limit=limit)

    def import_text(self, *, name: str, content: str):
        return {"name": name, "chunks_added": 0}

    def list_documents(self):
        return []


def _make_agent(model: _AdaptiveModel, knowledge: _RoundKnowledge) -> ThreeStageTaskAgent:
    return ThreeStageTaskAgent(
        model=model,
        patient_retriever=PatientRecordRetriever(""),
        knowledge_base=knowledge,
        registry=EvidenceRegistry(),
        patient_record="",
        request="find evidence",
        patient_grounding_required=False,
        retrieval_max_rounds=2,
        retrieval_refine_on_empty=True,
        retrieval_refine_min_candidates=1,
    )


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_empty_first_round_uses_one_model_driven_refinement_within_budget(self) -> None:
        model = _AdaptiveModel(evidence_on_first_round=False)
        knowledge = _RoundKnowledge()
        result = _make_agent(model, knowledge).run(
            {
                "id": 1,
                "goal": "find evidence",
                "deps": [],
                "evidence_scope": "both",
                "analysis_mode": "synthesis",
            },
            {},
        )

        self.assertEqual(model.query_calls, 2)
        self.assertEqual(knowledge.search_many_calls, 2)
        self.assertEqual(result["retrieval"]["rounds"], 2)
        self.assertEqual(result["retrieval"]["refinement_model_calls"], 1)
        self.assertEqual(result["retrieval"]["stop_reason"], "evidence_found")
        self.assertEqual(result["queries"], ["missing evidence", "known evidence"])
        self.assertTrue(result["evidence_ids"])

    def test_evidence_found_on_first_round_does_not_pay_for_refinement(self) -> None:
        model = _AdaptiveModel(evidence_on_first_round=True)
        knowledge = _RoundKnowledge()
        result = _make_agent(model, knowledge).run(
            {
                "id": 1,
                "goal": "find evidence",
                "deps": [],
                "evidence_scope": "both",
                "analysis_mode": "synthesis",
            },
            {},
        )

        self.assertEqual(model.query_calls, 1)
        self.assertEqual(knowledge.search_many_calls, 1)
        self.assertEqual(result["retrieval"]["rounds"], 1)
        self.assertEqual(result["retrieval"]["refinement_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
