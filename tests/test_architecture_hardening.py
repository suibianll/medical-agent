"""Regression tests for concurrency, repair routing and model selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import unittest
from unittest.mock import patch

from medical_agent.aliyun_model import AliyunCompatibleModelAdapter, normalize_base_url
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.evidence import EvidenceRegistry
from medical_agent.repair import build_repair_plan
from medical_agent.retrieval import JsonKnowledgeBase
from medical_agent.run_archive import InMemoryRunArchive
from medical_agent.service import MedicalAgentService


class _TaggedDemoModel(DemoModelAdapter):
    def __init__(self, name: str) -> None:
        self.name = name
        self.plan_calls = 0
        self._lock = Lock()

    def runtime_metadata(self) -> dict[str, str]:
        return {"mode": "real", "provider": "unit-provider", "name": self.name}

    def plan(self, request: str, patient_record: str) -> dict:
        with self._lock:
            self.plan_calls += 1
        return super().plan(request, patient_record)


class _CountingArchive(InMemoryRunArchive):
    def __init__(self) -> None:
        super().__init__(max_runs=8, ttl_seconds=60)
        self.finalize_calls = 0

    def finalize(self, result: dict) -> None:
        self.finalize_calls += 1
        super().finalize(result)


class ArchitectureHardeningTests(unittest.TestCase):
    def test_evidence_registry_allocates_unique_ids_under_concurrency(self) -> None:
        registry = EvidenceRegistry()
        workers = 40
        barrier = Barrier(workers)

        def add_unique(index: int) -> str:
            barrier.wait()
            return registry.add_knowledge(
                f"fact-{index}",
                source="parallel-test",
                locator=f"row-{index}",
                document_id="parallel",
            )["id"]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            ids = list(executor.map(add_unique, range(workers)))

        self.assertEqual(len(ids), workers)
        self.assertEqual(len(set(ids)), workers)
        self.assertEqual(
            {int(evidence_id[1:]) for evidence_id in ids}, set(range(1, workers + 1))
        )

    def test_evidence_registry_deduplicates_same_item_under_concurrency(self) -> None:
        registry = EvidenceRegistry()
        workers = 32
        barrier = Barrier(workers)

        def add_same(_index: int) -> str:
            barrier.wait()
            return registry.add_patient("same fact")["id"]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            ids = list(executor.map(add_same, range(workers)))

        self.assertEqual(set(ids), {"P1"})
        self.assertEqual(len(registry.all()), 1)

    def test_chat_archives_only_after_chat_fields_are_complete(self) -> None:
        archive = _CountingArchive()
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide", content="Traceable citations are required.")
        service = MedicalAgentService(
            model=DemoModelAdapter(), knowledge_base=knowledge, run_archive=archive
        )

        result = service.chat(message="What is required?")

        self.assertEqual(archive.finalize_calls, 1)
        archived = service.get_run(result["run"]["id"])
        self.assertEqual(archived["answer"], result["answer"])
        self.assertEqual(archived["mode"], "general")

    def test_missing_evidence_repairs_reopen_source_tasks(self) -> None:
        tasks = [
            {"id": 1, "goal": "提取患者病历事实", "deps": []},
            {"id": 2, "goal": "检索医学知识依据", "deps": []},
            {"id": 3, "goal": "综合分析", "deps": [1, 2]},
        ]
        claims = [{"id": "C1", "task_id": 3}]

        patient_repair = build_repair_plan(
            issues=[{"claim": "C1", "code": "MISSING_PATIENT_REF"}],
            claims=claims,
            tasks=tasks,
        )
        knowledge_repair = build_repair_plan(
            issues=[{"claim": "C1", "code": "MISSING_KB_REF"}],
            claims=claims,
            tasks=tasks,
        )

        self.assertIn(1, patient_repair["root_tasks"])
        self.assertIn(2, knowledge_repair["root_tasks"])
        self.assertTrue(patient_repair["task_directives"])
        self.assertTrue(knowledge_repair["task_directives"])

    def test_each_run_uses_its_requested_model_profile(self) -> None:
        first = _TaggedDemoModel("first-model")
        second = _TaggedDemoModel("second-model")
        service = MedicalAgentService(
            model_profiles={"first": first, "second": second},
            default_model_profile="first",
            knowledge_base=JsonKnowledgeBase([]),
            max_workers=1,
        )

        result = service.run(
            request="Review the evidence.",
            patient_record="Synthetic patient record.",
            model_profile="second",
        )

        self.assertEqual(first.plan_calls, 0)
        self.assertEqual(second.plan_calls, 1)
        self.assertEqual(result["run"]["model"]["profile"], "second")
        self.assertEqual(result["run"]["model"]["name"], "second-model")
        self.assertNotIn("base_url", str(service.model_catalog()))
        self.assertNotIn("api_key", str(service.model_catalog()))

    def test_openrouter_url_and_ambiguous_verdict_are_handled_safely(self) -> None:
        self.assertEqual(
            normalize_base_url("https://openrouter.ai/api/v1", "openrouter"),
            "https://openrouter.ai/api/v1",
        )
        adapter = AliyunCompatibleModelAdapter(
            api_key="unit-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="unit-model",
            provider="openrouter",
        )
        with patch.object(
            adapter, "_chat", return_value="NOT_SUPPORTED or SUPPORTED"
        ):
            verdict = adapter.judge_claim(claim={"text": "x"}, evidence=[])
        self.assertEqual(verdict, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
