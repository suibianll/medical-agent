"""Regression tests for concurrency, repair routing and model selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
import unittest
from unittest.mock import patch

import medical_agent
from medical_agent.adapters.openai_compatible import OpenAICompatibleModelAdapter
from medical_agent.bootstrap import create_agent
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.evidence import EvidenceRegistry
from medical_agent.infrastructure.openai_client import normalize_base_url
from medical_agent.repair import build_repair_plan
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.run_archive import InMemoryRunArchive, redact_audit_event


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
        self.started_ids: list[str] = []

    def start(self, run_id: str, created_at: str | None = None) -> dict:
        self.started_ids.append(run_id)
        return super().start(run_id, created_at)

    def finalize(self, result: dict) -> None:
        self.finalize_calls += 1
        super().finalize(result)


class ArchitectureHardeningTests(unittest.TestCase):
    def test_application_agent_has_no_concrete_runtime_dependencies(self) -> None:
        package_dir = Path(medical_agent.__file__).resolve().parent
        agent_source = (package_dir / "application" / "agent.py").read_text(
            encoding="utf-8"
        )

        for forbidden in (
            ".adapters",
            ".infrastructure",
            ".demo_model",
            "JsonKnowledgeBase",
            "InMemoryRunArchive",
            "load_model_configuration",
        ):
            self.assertNotIn(forbidden, agent_source)

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
        service = create_agent(
            model_profiles={"test": DemoModelAdapter()},
            default_model_profile="test",
            knowledge_base=knowledge,
            run_archive=archive,
        )

        result = service.chat(message="What is required?")

        self.assertEqual(archive.finalize_calls, 1)
        archived = service.get_run(result["run"]["id"])
        self.assertEqual(archived["answer"], result["answer"])
        self.assertEqual(archived["mode"], "general")

    def test_missing_evidence_repairs_reopen_source_tasks(self) -> None:
        tasks = [
            {
                "id": 1,
                "goal": "提取患者病历事实",
                "deps": [],
                "evidence_scope": "patient",
            },
            {
                "id": 2,
                "goal": "检索医学知识依据",
                "deps": [],
                "evidence_scope": "knowledge",
            },
            {"id": 3, "goal": "综合分析", "deps": [1, 2], "evidence_scope": "both"},
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
        service = create_agent(
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
        adapter = OpenAICompatibleModelAdapter(
            api_key="unit-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="unit-model",
            provider="openrouter",
        )
        with patch.object(
            adapter,
            "_complete_json",
            return_value={
                "verdicts": [
                    {"id": "C1", "verdict": "NOT_SUPPORTED or SUPPORTED"}
                ]
            },
        ):
            verdicts = adapter.judge_claims(
                [{"id": "C1", "claim": {"text": "x"}, "evidence": []}]
            )
        self.assertEqual(verdicts["C1"], "UNCERTAIN")

    def test_verifier_accepts_new_claim_level_taxonomy(self) -> None:
        adapter = OpenAICompatibleModelAdapter(
            api_key="unit-test-key",
            base_url="https://example.invalid/v1",
            model="unit-model",
            provider="openai-compatible",
        )
        with patch.object(
            adapter,
            "_complete_json",
            return_value={
                "verdicts": [
                    {"id": "C1", "verdict": "PARTIALLY_SUPPORTED"},
                    {"id": "C2", "verdict": "CONTRADICTED"},
                    {"id": "C3", "verdict": "INSUFFICIENT"},
                ]
            },
        ):
            verdicts = adapter.judge_claims(
                [
                    {"id": "C1", "claim": {"text": "x"}, "evidence": []},
                    {"id": "C2", "claim": {"text": "y"}, "evidence": []},
                    {"id": "C3", "claim": {"text": "z"}, "evidence": []},
                ]
            )
        self.assertEqual(verdicts["C1"], "PARTIALLY_SUPPORTED")
        self.assertEqual(verdicts["C2"], "CONTRADICTED")
        self.assertEqual(verdicts["C3"], "INSUFFICIENT")

    def test_failed_task_does_not_skip_evidence_validation(self) -> None:
        """A failed task must not let unverified claims be marked 'supported'."""

        class _PartialFailureModel(DemoModelAdapter):
            def plan(self, request: str, patient_record: str) -> dict:
                return {
                    "tasks": [
                        {
                            "id": 1,
                            "goal": "提取患者病历关键事实",
                            "deps": [],
                            "evidence_scope": "patient",
                            "analysis_mode": "retrieval",
                        },
                        {
                            "id": 2,
                            "goal": "综合分析临床建议",
                            "deps": [1],
                            "evidence_scope": "both",
                            "analysis_mode": "analysis",
                        },
                        {
                            "id": 3,
                            "goal": "检索医学知识依据",
                            "deps": [],
                            "evidence_scope": "knowledge",
                            "analysis_mode": "retrieval",
                        },
                    ]
                }

            def make_queries(self, *, task, request, patient_record, upstream):
                if task["id"] == 3:
                    raise RuntimeError("知识检索不可用")
                return super().make_queries(
                    task=task,
                    request=request,
                    patient_record=patient_record,
                    upstream=upstream,
                )

            def synthesize(self, *, task, request, facts):
                if task["id"] == 2:
                    patient_ref = next(
                        (f["ref"] for f in facts if f["ref"].startswith("P")), None
                    )
                    if patient_ref:
                        return {
                            "claims": [
                                {
                                    "text": "建议调整当前用药方案。",
                                    "refs": [patient_ref],
                                }
                            ],
                            "unknowns": [],
                        }
                return super().synthesize(task=task, request=request, facts=facts)

        service = create_agent(
            model_profiles={"test": _PartialFailureModel()},
            default_model_profile="test",
            knowledge_base=JsonKnowledgeBase([]),
            max_repair_rounds=0,
        )
        result = service.run(
            request="评估用药风险",
            patient_record="患者58岁，正在服用降压药。",
        )

        self.assertEqual(result["status"], "needs_human_review")
        statuses = {t["id"]: t["status"] for t in result["run"]["tasks"]}
        self.assertEqual(statuses[3], "failed")
        self.assertEqual(statuses[2], "completed")

        issue_codes = [i.get("code") for i in result["run"]["evaluation"]["issues"]]
        self.assertIn("TASK_EXECUTION_FAILED", issue_codes)
        self.assertIn("MISSING_KB_REF", issue_codes)

        clinical_claims = [
            c for c in result["claims"] if "建议" in c.get("text", "")
        ]
        self.assertTrue(clinical_claims, "应收集到含临床建议的结论")
        for claim in clinical_claims:
            self.assertEqual(claim["status"], "needs_repair")
            self.assertIn("MISSING_KB_REF", claim.get("issues", []))

    def test_chat_does_not_publish_claims_rejected_by_verifier(self) -> None:
        class _ContradictingModel(DemoModelAdapter):
            def judge_claims(self, items):
                return {item["id"]: "CONTRADICTED" for item in items}

        service = create_agent(
            model_profiles={"test": _ContradictingModel()},
            default_model_profile="test",
            max_repair_rounds=0,
        )

        result = service.chat(message="What medication risks should be checked?")

        self.assertEqual(result["status"], "needs_human_review")
        self.assertEqual(result["run"]["decision"]["outcome"], "defer")
        self.assertTrue(result["claims"])
        self.assertTrue(
            all(claim.get("status") == "needs_repair" for claim in result["claims"])
        )
        self.assertEqual(
            result["answer"],
            "当前证据链未通过自动核验，建议转人工审核或补充资料。",
        )
        self.assertNotIn(result["claims"][0]["text"], result["answer"])

    def test_missing_terminal_output_cannot_pass_workflow(self) -> None:
        class _NoAnalysisOutputModel(DemoModelAdapter):
            def synthesize(self, *, task, request, facts):
                return {"claims": [], "unknowns": ["analysis omitted"]}

        service = create_agent(
            model_profiles={"test": _NoAnalysisOutputModel()},
            default_model_profile="test",
            max_repair_rounds=0,
        )
        plan = {
            "tasks": [
                {
                    "id": 1,
                    "goal": "patient source",
                    "deps": [],
                    "evidence_scope": "patient",
                    "analysis_mode": "retrieval",
                },
                {
                    "id": 2,
                    "goal": "knowledge source",
                    "deps": [],
                    "evidence_scope": "knowledge",
                    "analysis_mode": "retrieval",
                },
                {
                    "id": 3,
                    "goal": "clinical analysis",
                    "deps": [1, 2],
                    "evidence_scope": "both",
                    "analysis_mode": "analysis",
                },
            ]
        }

        result = service.run(
            request="assess medication safety",
            patient_record="Patient eGFR 42 and allergy history.",
            plan=plan,
        )

        self.assertEqual(result["status"], "needs_human_review")
        self.assertEqual(result["run"]["decision"]["outcome"], "defer")
        self.assertIn(
            "TASK_OUTPUT_MISSING",
            [issue["code"] for issue in result["run"]["evaluation"]["issues"]],
        )

    def test_candidate_budget_limits_registered_response_evidence(self) -> None:
        documents = [
            {
                "id": f"doc-{index}",
                "document_id": f"doc-{index}",
                "title": f"Document {index}",
                "text": f"medication safety evidence item {index}",
                "keywords": ["medication", "safety"],
            }
            for index in range(8)
        ]
        service = create_agent(
            knowledge_base=JsonKnowledgeBase(documents),
            retrieval_candidate_budget=2,
            retrieval_limit=8,
            max_repair_rounds=0,
        )
        plan = {
            "tasks": [
                {
                    "id": 1,
                    "goal": "medication safety",
                    "deps": [],
                    "evidence_scope": "knowledge",
                    "analysis_mode": "retrieval",
                }
            ]
        }

        result = service.run(
            request="medication safety",
            patient_record="",
            plan=plan,
            allow_general=True,
        )

        self.assertEqual(result["run"]["tasks"][0]["retrieval"]["candidate_count"], 2)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["run"]["plan_source"], "provided")

    def test_invalid_provided_plan_is_rejected_without_fallback(self) -> None:
        service = create_agent(max_repair_rounds=0)

        result = service.run(
            request="medication safety",
            patient_record="",
            plan={"tasks": [{"id": 2, "goal": "invalid", "deps": []}]},
            allow_general=True,
        )

        self.assertEqual(result["status"], "plan_rejected")
        self.assertEqual(result["run"]["plan_source"], "provided")
        self.assertNotIn("plan_fallback_reason", result.get("run", {}))

    def test_invalid_model_plan_uses_conservative_fallback(self) -> None:
        class _InvalidPlanModel(DemoModelAdapter):
            def plan(self, request: str, patient_record: str):
                return {"tasks": []}

        service = create_agent(
            model_profiles={"invalid": _InvalidPlanModel()},
            default_model_profile="invalid",
            max_repair_rounds=0,
        )

        result = service.run(
            request="medication safety",
            patient_record="",
            allow_general=True,
        )

        self.assertEqual(result["run"]["plan_source"], "fallback")
        self.assertEqual(
            result["run"]["plan_fallback_reason"]["code"],
            "PLAN_VALIDATION_FAILED",
        )

    def test_post_planning_failure_marks_archive_failed(self) -> None:
        class _FailingVerifier(DemoModelAdapter):
            def judge_claims(self, items):
                raise RuntimeError("verifier unavailable")

        archive = _CountingArchive()
        service = create_agent(
            model_profiles={"test": _FailingVerifier()},
            default_model_profile="test",
            run_archive=archive,
            max_repair_rounds=0,
        )

        with self.assertRaisesRegex(RuntimeError, "verifier unavailable"):
            service.chat(message="general medication safety")

        self.assertEqual(len(archive.started_ids), 1)
        archived = archive.get_result(archive.started_ids[0])
        self.assertEqual(archived["status"], "failed")
        self.assertEqual(archived["archive"]["state"], "failed")

    def test_rerank_event_remains_rerank_after_archive_redaction(self) -> None:
        event = redact_audit_event(
            {
                "run_id": "123e4567-e89b-12d3-a456-426614174000",
                "stage": "rerank",
                "status": "completed",
            }
        )

        self.assertEqual(event["stage"], "rerank")


if __name__ == "__main__":
    unittest.main()
