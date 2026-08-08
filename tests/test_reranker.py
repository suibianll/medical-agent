from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from medical_agent.infrastructure.model_config import load_model_configuration
from medical_agent.bootstrap import create_agent
from medical_agent.agent_pipeline import ThreeStageTaskAgent
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.evidence import EvidenceRegistry
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.retrieval.patient import PatientRecordRetriever
from medical_agent.retrieval.reranker import ExternalApiReranker, RerankerError, RerankerRun
from medical_agent.observability.progress import safe_progress_event


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ExternalRerankerTests(unittest.TestCase):
    def test_config_reads_external_reranker_secret(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.local.json"
            path.write_text(
                json.dumps(
                    {
                        "reranker": {
                            "enabled": True,
                            "provider": "cohere",
                            "endpoint": "https://rerank.example.invalid",
                            "api_key_env": "RERANK_KEY",
                            "model": "rerank-v3",
                        }
                    }
                ),
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {"MEDICAL_AGENT_CONFIG": str(path), "RERANK_KEY": "secret"}
            )
        self.assertTrue(configuration.reranker.enabled)
        self.assertEqual(configuration.reranker.api_key, "secret")
        self.assertEqual(configuration.reranker.provider, "cohere")

    def test_config_reads_reranker_budget_and_cache_controls(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.local.json"
            path.write_text(
                json.dumps(
                    {
                        "reranker": {
                            "enabled": True,
                            "endpoint": "https://rerank.example.invalid",
                            "max_calls_per_run": 3,
                            "min_candidates": 4,
                            "cache_size": 7,
                            "cache_ttl_seconds": 19,
                        }
                    }
                ),
                encoding="utf-8",
            )
            configuration = load_model_configuration({"MEDICAL_AGENT_CONFIG": str(path)})
        self.assertEqual(configuration.reranker.max_calls_per_run, 3)
        self.assertEqual(configuration.reranker.min_candidates, 4)
        self.assertEqual(configuration.reranker.cache_size, 7)
        self.assertEqual(configuration.reranker.cache_ttl_seconds, 19)

    @patch("medical_agent.retrieval.reranker.urlopen")
    def test_external_results_reorder_candidates_and_preserve_metadata(self, urlopen) -> None:
        urlopen.return_value = _Response(
            {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "score": 0.12}]}
        )
        reranker = ExternalApiReranker(
            endpoint="https://rerank.example.invalid/v1/rerank",
            api_key="test-secret",
            provider="cohere",
            model="rerank-v3",
        )
        documents = [
            {"id": "K1", "title": "first", "text": "first", "score": 0.5},
            {"id": "K2", "title": "second", "text": "second", "score": 0.4},
        ]
        result = reranker.rerank(query="query", documents=documents, limit=2)

        self.assertEqual([item["id"] for item in result], ["K2", "K1"])
        self.assertEqual(result[0]["retrieval_method"], "external_reranker")
        self.assertEqual(result[0]["rerank_provider"], "cohere")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["documents"], ["first\nfirst", "second\nsecond"])
        self.assertEqual(request.headers["X-api-key"], "test-secret")

    @patch("medical_agent.retrieval.reranker.urlopen")
    def test_invalid_external_response_is_rejected(self, urlopen) -> None:
        urlopen.return_value = _Response({"results": [{"index": 7, "score": 1.0}]})
        reranker = ExternalApiReranker(endpoint="https://rerank.example.invalid")
        with self.assertRaises(RerankerError):
            reranker.rerank(
                query="query",
                documents=[{"id": "K1", "title": "first", "text": "first"}],
            )

    def test_task_pipeline_uses_injected_reranker_and_reports_status(self) -> None:
        class _SpyReranker:
            def __init__(self) -> None:
                self.calls = 0

            def rerank(self, *, query, documents, limit=8):
                self.calls += 1
                self.query = query
                return list(reversed(documents))[:limit]

            def runtime_metadata(self):
                return {"provider": "test", "name": "spy"}

        spy = _SpyReranker()
        agent = ThreeStageTaskAgent(
            model=DemoModelAdapter(),
            patient_retriever=PatientRecordRetriever(""),
            knowledge_base=JsonKnowledgeBase(
                [
                    {"id": "K1", "title": "one", "text": "query one"},
                    {"id": "K2", "title": "two", "text": "query two"},
                ]
            ),
            registry=EvidenceRegistry(),
            patient_record="",
            request="query",
            reranker=spy,
        )

        evidence_ids, retrieval = agent._retrieve(
            {"id": 1, "goal": "query", "deps": []}, ["query"]
        )

        self.assertEqual(spy.calls, 1)
        self.assertEqual(spy.query, "query")
        self.assertEqual(retrieval["rerank_status"], "completed")
        self.assertEqual(len(evidence_ids), 2)
        self.assertEqual(agent.registry.get(evidence_ids[0])["source"], "two")

    def test_run_budget_deduplicates_and_limits_external_calls(self) -> None:
        class _SpyReranker:
            def __init__(self) -> None:
                self.calls = 0

            def rerank(self, *, query, documents, limit=8):
                self.calls += 1
                return [
                    {**documents[1], "rerank_score": 0.9},
                    {**documents[0], "rerank_score": 0.2},
                ][:limit]

            def runtime_metadata(self):
                return {"provider": "test", "name": "spy"}

        spy = _SpyReranker()
        runtime = RerankerRun(
            spy,
            max_calls=1,
            min_candidates=2,
            cache_size=4,
            cache_ttl_seconds=60,
        )
        documents = [
            {"id": "K1", "title": "one", "text": "one"},
            {"id": "K2", "title": "two", "text": "two"},
        ]
        first = runtime.rerank(query="same", documents=documents, limit=2)
        second = runtime.rerank(query="same", documents=documents, limit=2)
        blocked = runtime.rerank(query="different", documents=documents, limit=2)

        self.assertEqual(spy.calls, 1)
        self.assertEqual([item["id"] for item in first], ["K2", "K1"])
        self.assertEqual([item["id"] for item in second], ["K2", "K1"])
        self.assertEqual(blocked, [])
        usage = runtime.usage()["reranker"]
        self.assertEqual(usage["external_calls"], 1)
        self.assertEqual(usage["cache_hits"], 1)
        self.assertEqual(usage["skipped_budget"], 1)

    def test_run_normalizes_results_and_preserves_candidate_provenance(self) -> None:
        class _UntrustedReranker:
            def rerank(self, *, query, documents, limit=8):
                return [
                    {
                        **documents[1],
                        "text": "forged text must not replace the candidate",
                        "rerank_score": 0.91,
                        "rerank_provider": "test-provider",
                    },
                    {"id": "K2", "document_id": "D2", "rerank_score": 0.8},
                ][:limit]

        runtime = RerankerRun(
            _UntrustedReranker(),
            max_calls=1,
            min_candidates=2,
            cache_size=4,
            cache_ttl_seconds=60,
        )
        documents = [
            {"id": "K1", "document_id": "D1", "title": "one", "text": "source one"},
            {"id": "K2", "document_id": "D2", "title": "two", "text": "source two"},
        ]

        result = runtime.rerank(query="same", documents=documents, limit=2)
        cached = runtime.rerank(query="same", documents=documents, limit=2)

        self.assertEqual([item["id"] for item in result], ["K2"])
        self.assertEqual(result[0]["text"], "source two")
        self.assertEqual(result[0]["retrieval_method"], "external_reranker")
        self.assertFalse(result[0]["rerank_cached"])
        self.assertTrue(cached[0]["rerank_cached"])
        self.assertEqual(cached[0]["text"], "source two")

    def test_run_budget_skips_single_candidate(self) -> None:
        class _UnexpectedReranker:
            def rerank(self, **_kwargs):
                raise AssertionError("single candidate should not call reranker")

            def runtime_metadata(self):
                return {"provider": "test", "name": "unexpected"}

        runtime = RerankerRun(_UnexpectedReranker(), min_candidates=2)
        self.assertEqual(
            runtime.rerank(
                query="query",
                documents=[{"id": "K1", "title": "one", "text": "one"}],
            ),
            [],
        )
        self.assertEqual(runtime.usage()["reranker"]["skipped_min_candidates"], 1)

    def test_workflow_exposes_retrieval_usage_without_raw_text(self) -> None:
        class _SpyReranker:
            def rerank(self, *, query, documents, limit=8):
                return list(reversed(documents))[:limit]

            def runtime_metadata(self):
                return {"provider": "test", "name": "spy"}

        service = create_agent(
            model_profiles={"demo": DemoModelAdapter()},
            knowledge_base=JsonKnowledgeBase(
                [
                    {"id": "K1", "title": "one", "text": "query one"},
                    {"id": "K2", "title": "two", "text": "query two"},
                ]
            ),
            reranker=_SpyReranker(),
            max_workers=1,
        )
        result = service.chat(message="query")
        usage = result["run"]["retrieval_usage"]["reranker"]
        self.assertGreaterEqual(usage["attempts"], 1)
        self.assertIn("external_calls", usage)
        self.assertNotIn("query", str(usage))
        self.assertNotIn("query one", str(usage))

    def test_rerank_progress_event_is_whitelisted(self) -> None:
        event = safe_progress_event(
            {
                "stage": "rerank",
                "message": "done",
                "status": "completed",
                "candidate_count": 3,
                "raw_documents": "must not leak",
            },
            run_id="run-1",
            sequence=1,
        )
        self.assertEqual(event["stage"], "rerank")
        self.assertEqual(event["candidate_count"], 3)
        self.assertNotIn("raw_documents", event)


if __name__ == "__main__":
    unittest.main()
