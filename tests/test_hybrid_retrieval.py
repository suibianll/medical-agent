from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from medical_agent.infrastructure.model_config import load_model_configuration
from medical_agent.retrieval.hybrid import HybridKnowledgeBase


class _Backend:
    def __init__(self, name: str, results: list[dict]) -> None:
        self.name = name
        self.results = results
        self.documents = list(results)

    def search_many(self, _queries, **_kwargs):
        return [dict(item) for item in self.results]

    def search(self, _query, limit=4):
        return [dict(item) for item in self.results[:limit]]

    def import_text(self, *, name: str, content: str):
        return {"name": name, "content_length": len(content)}

    def list_documents(self):
        return [{"id": item["id"]} for item in self.results]

    def retrieval_metadata(self):
        return {"backend": self.name, "provider": "test"}


class HybridRetrievalTests(unittest.TestCase):
    def test_weighted_fusion_keeps_backend_scores_and_document_cap(self) -> None:
        sparse = _Backend(
            "lexical",
            [
                {
                    "id": "shared",
                    "document_id": "doc-a",
                    "title": "Shared",
                    "text": "sparse",
                    "score": 4,
                    "retrieval_score": 0.02,
                    "retrieval_queries": ["sparse query"],
                },
                {
                    "id": "sparse-only",
                    "document_id": "doc-a",
                    "title": "Sparse only",
                    "text": "sparse-only",
                    "score": 3,
                    "retrieval_score": 0.01,
                },
            ],
        )
        dense = _Backend(
            "faiss",
            [
                {
                    "id": "dense-only",
                    "document_id": "doc-b",
                    "title": "Dense only",
                    "text": "dense-only",
                    "score": 0.9,
                    "retrieval_score": 0.016,
                    "retrieval_queries": ["dense query"],
                },
                {
                    "id": "shared",
                    "document_id": "doc-a",
                    "title": "Shared from dense",
                    "text": "dense",
                    "score": 0.8,
                    "retrieval_score": 0.015,
                },
            ],
        )

        knowledge = HybridKnowledgeBase(
            sparse,
            dense,
            sparse_weight=0.4,
            dense_weight=0.6,
        )
        results = knowledge.search_many(["query"], limit=3, max_per_document=1)

        self.assertEqual({item["id"] for item in results}, {"shared", "dense-only"})
        shared = next(item for item in results if item["id"] == "shared")
        self.assertEqual(shared["retrieval_method"], "hybrid_rrf")
        self.assertEqual(shared["retrieval_sources"], ["sparse", "dense"])
        self.assertIn("sparse_score", shared)
        self.assertIn("dense_score", shared)
        self.assertAlmostEqual(
            shared["retrieval_score"],
            0.4 / 61 + 0.6 / 62,
        )
        self.assertEqual(knowledge.retrieval_metadata()["backend"], "hybrid")

    def test_source_scores_do_not_pollute_weighted_rrf_order(self) -> None:
        sparse = _Backend(
            "lexical",
            [
                {"id": "a", "title": "A", "text": "A", "retrieval_score": 10.0},
                {"id": "b", "title": "B", "text": "B", "retrieval_score": 0.001},
            ],
        )
        dense = _Backend(
            "faiss",
            [
                {"id": "b", "title": "B", "text": "B", "retrieval_score": 0.02},
                {"id": "a", "title": "A", "text": "A", "retrieval_score": 0.01},
            ],
        )

        results = HybridKnowledgeBase(
            sparse,
            dense,
            sparse_weight=0.1,
            dense_weight=0.9,
        ).search_many(["query"], limit=2)

        self.assertEqual([item["id"] for item in results], ["b", "a"])
        self.assertLess(results[0]["retrieval_score"], 0.1)

    def test_backend_type_error_is_not_masked_as_signature_fallback(self) -> None:
        class _BrokenBackend(_Backend):
            def __init__(self):
                super().__init__("broken", [])
                self.calls = 0

            def search_many(self, _queries, **_kwargs):
                self.calls += 1
                raise TypeError("backend implementation failed")

        broken = _BrokenBackend()
        knowledge = HybridKnowledgeBase(broken, _Backend("dense", []))

        with self.assertRaisesRegex(TypeError, "backend implementation failed"):
            knowledge.search_many(["query"])
        self.assertEqual(broken.calls, 1)

    def test_duplicate_backend_ids_do_not_receive_multiple_rrf_votes(self) -> None:
        duplicate = _Backend(
            "lexical",
            [
                {"id": "a", "title": "A", "text": "A", "score": 1.0},
                {"id": "a", "title": "A", "text": "A", "score": 0.9},
                {"id": "b", "title": "B", "text": "B", "score": 0.8},
            ],
        )
        hybrid = HybridKnowledgeBase(
            duplicate,
            _Backend("dense", []),
            sparse_weight=1.0,
            dense_weight=0.0,
        )

        results = hybrid.search_many(["query"], limit=2)

        self.assertEqual([item["id"] for item in results], ["a", "b"])
        self.assertAlmostEqual(results[0]["retrieval_score"], 1 / 61)
        self.assertAlmostEqual(results[1]["retrieval_score"], 1 / 63)

    def test_configuration_parses_hybrid_controls(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "model.local.json"
            config_path.write_text(
                '{"retrieval":{"backend":"hybrid","hybrid_sparse_weight":0.7,'
                '"hybrid_dense_weight":0.3,"hybrid_rrf_k":17,'
                '"relevance_threshold":0.12}}',
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {"MEDICAL_AGENT_CONFIG": str(config_path)}
            )

        self.assertEqual(configuration.retrieval.backend, "hybrid")
        self.assertEqual(configuration.retrieval.hybrid_sparse_weight, 0.7)
        self.assertEqual(configuration.retrieval.hybrid_dense_weight, 0.3)
        self.assertEqual(configuration.retrieval.hybrid_rrf_k, 17)
        self.assertEqual(configuration.retrieval.relevance_threshold, 0.12)


if __name__ == "__main__":
    unittest.main()
