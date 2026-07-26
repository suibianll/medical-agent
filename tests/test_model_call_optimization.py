"""Regression tests that keep external model calls bounded and useful."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_agent.adapters.openai_compatible import OpenAICompatibleModelAdapter
from medical_agent.agent_pipeline import ThreeStageTaskAgent
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.evaluator import evaluate_claims
from medical_agent.evidence import EvidenceRegistry
from medical_agent.retrieval import JsonKnowledgeBase, PatientRecordRetriever


class _CountingModel(DemoModelAdapter):
    def __init__(self, *, return_facts: bool = True) -> None:
        self.return_facts = return_facts
        self.extract_calls = 0
        self.synthesis_calls = 0
        self.last_evidence_ids: list[str] = []

    def extract_facts(self, *, task: dict, evidence: list[dict[str, str]]) -> dict:
        self.extract_calls += 1
        self.last_evidence_ids = [item["id"] for item in evidence]
        if not self.return_facts:
            return {"facts": []}
        return super().extract_facts(task=task, evidence=evidence)

    def synthesize(
        self,
        *,
        task: dict,
        request: str,
        facts: list[dict[str, str]],
    ) -> dict:
        self.synthesis_calls += 1
        return super().synthesize(task=task, request=request, facts=facts)


class _BatchJudge:
    def __init__(self) -> None:
        self.calls = 0
        self.items: list[dict] = []

    def judge_claims(self, items: list[dict]) -> dict[str, str]:
        self.calls += 1
        self.items = items
        return {item["id"]: "SUPPORTED" for item in items}


def _agent(model: DemoModelAdapter, knowledge: JsonKnowledgeBase) -> ThreeStageTaskAgent:
    return ThreeStageTaskAgent(
        model=model,
        patient_retriever=PatientRecordRetriever(""),
        knowledge_base=knowledge,
        registry=EvidenceRegistry(),
        patient_record="",
        request="review evidence",
        patient_grounding_required=False,
    )


class ModelCallOptimizationTests(unittest.TestCase):
    def test_no_evidence_skips_extraction_and_synthesis_calls(self) -> None:
        model = _CountingModel()

        result = _agent(model, JsonKnowledgeBase([])).run(
            {"id": 1, "goal": "find evidence", "deps": []}, {}
        )

        self.assertEqual(model.extract_calls, 0)
        self.assertEqual(model.synthesis_calls, 0)
        self.assertEqual(result["claims"], [])
        self.assertTrue(result["unknowns"])

    def test_no_extracted_facts_skips_synthesis_call(self) -> None:
        model = _CountingModel(return_facts=False)
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide", content="Evidence review is required.")

        result = _agent(model, knowledge).run(
            {"id": 1, "goal": "evidence review", "deps": []}, {}
        )

        self.assertEqual(model.extract_calls, 1)
        self.assertEqual(model.synthesis_calls, 0)
        self.assertTrue(result["unknowns"])

    def test_source_task_projects_cited_facts_without_synthesis_call(self) -> None:
        model = _CountingModel()
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide", content="Evidence review is required.")

        result = _agent(model, knowledge).run(
            {"id": 1, "goal": "检索医学依据", "deps": []}, {}
        )

        self.assertEqual(model.extract_calls, 1)
        self.assertEqual(model.synthesis_calls, 0)
        self.assertTrue(result["claims"])
        self.assertTrue(all(claim["refs"] for claim in result["claims"]))
        self.assertTrue(
            all(claim["refs"][0].startswith("K") for claim in result["claims"])
        )

    def test_patient_source_task_does_not_project_knowledge_facts(self) -> None:
        model = _CountingModel()
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide", content="Clinical evidence is required.")
        agent = ThreeStageTaskAgent(
            model=model,
            patient_retriever=PatientRecordRetriever("Patient has dizziness."),
            knowledge_base=knowledge,
            registry=EvidenceRegistry(),
            patient_record="Patient has dizziness.",
            request="extract patient facts",
            patient_grounding_required=True,
        )

        result = agent.run(
            {"id": 1, "goal": "extract patient record facts", "deps": []}, {}
        )

        self.assertTrue(result["claims"])
        self.assertTrue(
            all(claim["refs"][0].startswith("P") for claim in result["claims"])
        )
        self.assertEqual(model.synthesis_calls, 0)

    def test_extraction_preserves_patient_before_knowledge_retrieval_order(self) -> None:
        model = _CountingModel()
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide", content="Clinical evidence is required.")
        agent = ThreeStageTaskAgent(
            model=model,
            patient_retriever=PatientRecordRetriever("Patient has dizziness."),
            knowledge_base=knowledge,
            registry=EvidenceRegistry(),
            patient_record="Patient has dizziness.",
            request="analyze dizziness",
            patient_grounding_required=True,
        )

        agent.run({"id": 1, "goal": "analyze question", "deps": []}, {})

        self.assertTrue(model.last_evidence_ids)
        self.assertTrue(model.last_evidence_ids[0].startswith("P"))
        self.assertTrue(any(item.startswith("K") for item in model.last_evidence_ids))

    def test_semantic_evaluation_batches_only_deterministically_valid_claims(self) -> None:
        judge = _BatchJudge()
        claims = [
            {"id": "C1", "text": "valid one", "refs": ["K1"], "requires_dual_support": False},
            {"id": "C2", "text": "missing ref", "refs": [], "requires_dual_support": False},
            {"id": "C3", "text": "valid two", "refs": ["K1"], "requires_dual_support": False},
        ]

        result = evaluate_claims(
            claims, {"K1": {"id": "K1", "text": "support"}}, judge
        )

        self.assertEqual(judge.calls, 1)
        self.assertEqual([item["id"] for item in judge.items], ["C1", "C3"])
        self.assertIn({"claim": "C2", "code": "NO_REF"}, result["issues"])

    def test_provider_batches_large_evaluation_instead_of_calling_per_claim(self) -> None:
        adapter = OpenAICompatibleModelAdapter(
            api_key="unit-key",
            base_url="https://example.invalid/v1",
            model="unit-model",
            provider="openai-compatible",
        )
        items = [
            {
                "id": f"C{index}",
                "claim": {"text": f"claim {index}"},
                "evidence": [{"id": "K1", "text": "support"}],
            }
            for index in range(1, 10)
        ]

        def response(prompt: object) -> dict:
            batch = prompt.payload["items"]  # type: ignore[attr-defined]
            return {
                "verdicts": [
                    {"id": item["id"], "verdict": "SUPPORTED"} for item in batch
                ]
            }

        with patch.object(adapter, "_complete_json", side_effect=response) as complete:
            verdicts = adapter.judge_claims(items)

        self.assertEqual(complete.call_count, 2)
        self.assertEqual(set(verdicts), {item["id"] for item in items})

    def test_repair_round_can_reuse_unchanged_semantic_verdicts(self) -> None:
        judge = _BatchJudge()
        cache: dict[tuple[str, tuple[str, ...]], str] = {}
        claims = [
            {
                "id": "C1",
                "text": "unchanged claim",
                "refs": ["K1"],
                "requires_dual_support": False,
            }
        ]
        evidence = {"K1": {"id": "K1", "text": "support"}}

        first = evaluate_claims(claims, evidence, judge, semantic_cache=cache)
        second = evaluate_claims(claims, evidence, judge, semantic_cache=cache)

        self.assertTrue(first["pass"])
        self.assertTrue(second["pass"])
        self.assertEqual(judge.calls, 1)


if __name__ == "__main__":
    unittest.main()
