"""Tests for bounded retrieval fusion and safe model-call telemetry."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from medical_agent.bootstrap import create_agent
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.evaluator import evaluate_claims
from medical_agent.infrastructure.openai_client import OpenAIChatClient
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.retrieval.state import RetrievalState
from medical_agent.risk import route_decision


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.payload


class _TelemetryDemo(DemoModelAdapter):
    def __init__(self) -> None:
        self.metrics: list[dict] = []

    def _mark(self, stage: str) -> None:
        self.metrics.append(
            {
                "stage": stage,
                "provider": "test",
                "model": "demo",
                "latency_ms": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "success": True,
            }
        )

    def drain_call_metrics(self) -> list[dict]:
        values = list(self.metrics)
        self.metrics.clear()
        return values

    def plan(self, request: str, patient_record: str) -> dict:
        self._mark("plan")
        return super().plan(request, patient_record)

    def make_queries(self, **kwargs: object) -> dict:
        self._mark("query")
        return super().make_queries(**kwargs)  # type: ignore[arg-type]

    def extract_facts(self, **kwargs: object) -> dict:
        self._mark("extract")
        return super().extract_facts(**kwargs)  # type: ignore[arg-type]

    def synthesize(self, **kwargs: object) -> dict:
        self._mark("synthesize")
        return super().synthesize(**kwargs)  # type: ignore[arg-type]

    def judge_claims(self, items: list[dict]) -> dict[str, str]:
        self._mark("judge")
        return super().judge_claims(items)


class _NoJudgeGenerator(DemoModelAdapter):
    def judge_claims(self, _items: list[dict]) -> dict[str, str]:
        raise AssertionError("generation model must not be used as verifier")


class _CountingVerifier(DemoModelAdapter):
    def __init__(self) -> None:
        self.judge_calls = 0

    def judge_claims(self, items: list[dict]) -> dict[str, str]:
        self.judge_calls += 1
        return super().judge_claims(items)


class RetrievalAndMetricsTests(unittest.TestCase):
    def test_search_many_fuses_queries_and_limits_same_document(self) -> None:
        knowledge = JsonKnowledgeBase(
            [
                {
                    "id": "doc-a-1",
                    "document_id": "doc-a",
                    "title": "指南 A",
                    "text": "alpha beta evidence",
                    "keywords": ["alpha"],
                    "priority": 5,
                },
                {
                    "id": "doc-a-2",
                    "document_id": "doc-a",
                    "title": "指南 A",
                    "text": "alpha follow-up evidence",
                    "keywords": ["alpha"],
                    "priority": 5,
                },
                {
                    "id": "doc-b-1",
                    "document_id": "doc-b",
                    "title": "指南 B",
                    "text": "beta contraindication evidence",
                    "keywords": ["beta"],
                    "priority": 5,
                },
            ]
        )

        results = knowledge.search_many(["alpha", "beta"], limit=3, max_per_document=1)

        self.assertEqual(len(results), 2)
        self.assertEqual({item["document_id"] for item in results}, {"doc-a", "doc-b"})
        self.assertTrue(all(item["retrieval_method"] == "rrf_lexical" for item in results))
        self.assertTrue(any("beta" in item["retrieval_queries"] for item in results))

    def test_retrieval_state_stops_at_candidate_budget(self) -> None:
        state = RetrievalState(max_rounds=2, max_candidates=2)
        state.begin_round(["one", "two"])
        state.add_evidence(["K1", "K2", "K3"])
        state.finish()

        self.assertEqual(state.rounds, 1)
        self.assertEqual(state.as_dict()["candidate_count"], 2)
        self.assertEqual(state.stop_reason, "candidate_budget_exhausted")

    def test_openai_client_records_usage_without_recording_prompt_text(self) -> None:
        client = OpenAIChatClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="unit-model",
            provider="openai-compatible",
        )
        payload = {
            "choices": [{"message": {"content": "{\"ok\":true}"}}],
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 17,
                "total_tokens": 118,
                "prompt_tokens_details": {"cached_tokens": 9},
            },
        }

        with patch(
            "medical_agent.infrastructure.openai_client.urlopen",
            return_value=_Response(payload),
        ):
            result = client.complete(
                system="private system prompt",
                user="patient record must not be logged",
                stage="synthesize",
            )

        self.assertEqual(result, '{"ok":true}')
        metrics = client.drain_call_metrics()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["stage"], "synthesize")
        self.assertEqual(metrics[0]["total_tokens"], 118)
        self.assertEqual(metrics[0]["cached_tokens"], 9)
        self.assertNotIn("private system prompt", str(metrics))
        self.assertNotIn("patient record", str(metrics))

    def test_run_exposes_safe_model_usage_summary(self) -> None:
        model = _TelemetryDemo()
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide.txt", content="Evidence-backed guidance for ORBIT-42.")
        service = create_agent(
            model_profiles={"test": model},
            default_model_profile="test",
            knowledge_base=knowledge,
            max_workers=1,
        )

        result = service.chat(message="What does ORBIT-42 require?", patient_record="")

        usage = result["run"]["model_usage"]
        self.assertGreater(usage["calls"], 0)
        self.assertIn("query", usage["stages"])
        self.assertIn("total_tokens", usage)
        self.assertTrue(result["run"]["model_calls"])
        self.assertNotIn("ORBIT-42", str(result["run"]["model_usage"]))

    def test_evaluator_emits_first_class_support_edge_with_span(self) -> None:
        result = evaluate_claims(
            [{"id": "C1", "text": "supported", "refs": ["K1"]}],
            {
                "K1": {
                    "id": "K1",
                    "kind": "knowledge",
                    "text": "support",
                    "span": {"id": "K1:span:0"},
                }
            },
        )

        self.assertTrue(result["pass"])
        self.assertEqual(result["support_edges"][0]["span_id"], "K1:span:0")
        self.assertEqual(result["support_edges"][0]["relation"], "supports")

    def test_verifier_model_is_separate_from_generation_model(self) -> None:
        verifier = _CountingVerifier()
        knowledge = JsonKnowledgeBase([])
        knowledge.import_text(name="guide.txt", content="Evidence-backed guidance for separation.")
        service = create_agent(
            model_profiles={"generator": _NoJudgeGenerator()},
            default_model_profile="generator",
            verifier_model=verifier,
            knowledge_base=knowledge,
            max_workers=1,
        )

        result = service.chat(message="What does separation require?", patient_record="")

        self.assertEqual(result["status"], "passed")
        self.assertGreater(verifier.judge_calls, 0)

    def test_default_risk_router_does_not_scan_request_keywords(self) -> None:
        decision = route_decision(
            request="患者胸痛并呼吸困难，应该如何处理？",
            patient_record="",
            claims=[],
            evaluation={"pass": False, "issues": [{"code": "NO_REF"}]},
        )

        self.assertEqual(decision["outcome"], "defer")
        self.assertEqual(decision["risk_level"], "unknown")
        self.assertFalse(decision["emergency_signal"])


if __name__ == "__main__":
    unittest.main()
