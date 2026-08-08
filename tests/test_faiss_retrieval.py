from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional vector extra
    np = None  # type: ignore[assignment]

from medical_agent.infrastructure.model_config import load_model_configuration
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.retrieval.vector import (
    CachedEmbeddingProvider,
    EmbeddingProviderError,
    FaissKnowledgeBase,
    HashEmbeddingProvider,
    RetrievalBackendUnavailable,
)


class _FakeIndex:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.array(vectors, dtype=np.float32, copy=True)

    def search(self, queries: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.vectors is not None
        scores = queries @ self.vectors.T
        indices = np.argsort(-scores, axis=1)[:, :limit]
        return np.take_along_axis(scores, indices, axis=1), indices


class _FakeFaiss:
    IndexFlatIP = _FakeIndex

    @staticmethod
    def normalize_L2(vectors: np.ndarray) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors /= norms

    @staticmethod
    def write_index(index: _FakeIndex, path: str) -> None:
        Path(path).write_bytes(b"fake-index")

    @staticmethod
    def read_index(_path: str) -> _FakeIndex:
        raise OSError("fake persistence is intentionally not implemented")


class _StaticEmbedding:
    dimensions = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def runtime_metadata(self) -> dict[str, str]:
        return {"provider": "test", "name": "static", "dimensions": "2"}


@unittest.skipIf(np is None, "vector extra is not installed")
class FaissRetrievalTests(unittest.TestCase):
    def test_configuration_parses_faiss_backend_and_embedding_secret(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "model.local.json"
            config_path.write_text(
                '{"retrieval":{"backend":"faiss","faiss":{"index_path":"cache.faiss",'
                '"embedding":{"provider":"openai-compatible","api_key_env":"EMBED_KEY",'
                '"base_url":"https://example.invalid/v1","model":"embed-model",'
                '"dimensions":32}}}}',
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {
                    "MEDICAL_AGENT_CONFIG": str(config_path),
                    "EMBED_KEY": "embedding-secret",
                }
            )
        self.assertEqual(configuration.retrieval.backend, "faiss")
        self.assertEqual(configuration.retrieval.index_path, "cache.faiss")
        self.assertEqual(configuration.retrieval.embedding.provider, "openai-compatible")
        self.assertEqual(configuration.retrieval.embedding.api_key, "embedding-secret")

    def test_configuration_parses_embedding_cache_controls(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "model.local.json"
            config_path.write_text(
                '{"retrieval":{"backend":"faiss","embedding":{"cache_size":9,"cache_ttl_seconds":17}}}',
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {"MEDICAL_AGENT_CONFIG": str(config_path)}
            )
        self.assertEqual(configuration.retrieval.embedding.cache_size, 9)
        self.assertEqual(configuration.retrieval.embedding.cache_ttl_seconds, 17)

    def test_faiss_decorator_searches_and_invalidates_after_import(self) -> None:
        source = JsonKnowledgeBase(
            [
                {"id": "a", "title": "Alpha", "text": "alpha", "priority": 1},
                {"id": "b", "title": "Beta", "text": "beta", "priority": 1},
            ]
        )
        provider = _StaticEmbedding(
            {
                "Alpha alpha": [1.0, 0.0],
                "Beta beta": [0.0, 1.0],
                "alpha": [1.0, 0.0],
                "beta": [0.0, 1.0],
            }
        )
        with TemporaryDirectory() as directory:
            knowledge = FaissKnowledgeBase(
                source,
                embedding_provider=provider,
                index_path=Path(directory) / "index.faiss",
                faiss_module=_FakeFaiss,
                numpy_module=np,
            )
            results = knowledge.search_many(["beta"], limit=1)
            self.assertEqual(results[0]["id"], "b")
            self.assertEqual(results[0]["retrieval_method"], "rrf_faiss")
            knowledge.import_text(name="new.txt", content="gamma")
            self.assertEqual(len(knowledge.documents), 3)

    def test_search_many_batches_query_embeddings(self) -> None:
        source = JsonKnowledgeBase(
            [
                {"id": "a", "title": "Alpha", "text": "alpha", "priority": 1},
                {"id": "b", "title": "Beta", "text": "beta", "priority": 1},
            ]
        )

        class _BatchEmbedding(_StaticEmbedding):
            def __init__(self, vectors):
                super().__init__(vectors)
                self.calls: list[list[str]] = []

            def embed(self, texts):
                values = list(texts)
                self.calls.append(values)
                return super().embed(values)

        provider = _BatchEmbedding(
            {
                "Alpha alpha": [1.0, 0.0],
                "Beta beta": [0.0, 1.0],
                "alpha": [1.0, 0.0],
                "beta": [0.0, 1.0],
            }
        )
        with TemporaryDirectory() as directory:
            knowledge = FaissKnowledgeBase(
                source,
                embedding_provider=provider,
                index_path=Path(directory) / "index.faiss",
                faiss_module=_FakeFaiss,
                numpy_module=np,
            )
            results = knowledge.search_many(["alpha", "beta"], limit=2)

        self.assertEqual([len(call) for call in provider.calls], [2, 2])
        self.assertEqual({item["id"] for item in results}, {"a", "b"})

    def test_missing_faiss_dependency_is_actionable(self) -> None:
        with patch(
            "medical_agent.retrieval.vector._load_faiss",
            side_effect=RetrievalBackendUnavailable("faiss missing"),
        ):
            with self.assertRaises(RetrievalBackendUnavailable):
                FaissKnowledgeBase(
                    JsonKnowledgeBase([]),
                    embedding_provider=HashEmbeddingProvider(),
                    numpy_module=np,
                )

    def test_invalid_embedding_shape_is_rejected(self) -> None:
        source = JsonKnowledgeBase([{"id": "a", "title": "A", "text": "a"}])
        provider = _StaticEmbedding({"A a": [1.0, 0.0]})
        provider.embed = lambda _texts: []  # type: ignore[method-assign]
        with self.assertRaises(EmbeddingProviderError):
            FaissKnowledgeBase(
                source,
                embedding_provider=provider,
                faiss_module=_FakeFaiss,
                numpy_module=np,
            ).search("a")

    def test_index_embedding_is_batched_for_large_document_pools(self) -> None:
        documents = [
            {"id": f"doc-{index}", "title": f"Doc {index}", "text": f"text {index}"}
            for index in range(65)
        ]

        class _BatchLimitedEmbedding:
            dimensions = 2

            def __init__(self) -> None:
                self.calls: list[int] = []

            def embed(self, texts):
                values = list(texts)
                if len(values) > 64:
                    raise EmbeddingProviderError("batch too large")
                self.calls.append(len(values))
                return [[1.0, 0.0] for _ in values]

            def runtime_metadata(self):
                return {"provider": "test", "name": "batch-limited", "dimensions": "2"}

        provider = _BatchLimitedEmbedding()
        knowledge = FaissKnowledgeBase(
            JsonKnowledgeBase(documents),
            embedding_provider=provider,
            faiss_module=_FakeFaiss,
            numpy_module=np,
        )

        knowledge.search("text 1", limit=1)

        self.assertEqual(provider.calls, [64, 1, 1])

    def test_long_documents_are_chunked_before_embedding(self) -> None:
        documents = [
            {
                "id": "article-1",
                "document_id": "article-1",
                "title": "Long article",
                "text": "needle " + ("background " * 2_000),
            }
        ]

        class _LengthLimitedEmbedding:
            dimensions = 2

            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def embed(self, texts):
                values = list(texts)
                self.calls.append(values)
                if any(len(value) > 8_000 for value in values):
                    raise EmbeddingProviderError("long input")
                return [[1.0, 0.0] for _ in values]

            def runtime_metadata(self):
                return {"provider": "test", "name": "length-limited", "dimensions": "2"}

        provider = _LengthLimitedEmbedding()
        knowledge = FaissKnowledgeBase(
            JsonKnowledgeBase(documents),
            embedding_provider=provider,
            faiss_module=_FakeFaiss,
            numpy_module=np,
        )

        results = knowledge.search("needle", limit=1)

        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "article-1")
        self.assertNotIn("_faiss_chunk_id", results[0])
        self.assertGreater(len(provider.calls), 1)
        self.assertLessEqual(
            max(len(value) for batch in provider.calls for value in batch), 8_000
        )

    def test_cached_embedding_provider_batches_misses_and_reports_safe_usage(self) -> None:
        class _CountingProvider:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def embed(self, texts):
                values = list(texts)
                self.calls.append(values)
                return [[float(len(value)), 1.0] for value in values]

            def runtime_metadata(self):
                return {"provider": "test", "name": "counting", "dimensions": "2"}

        source = _CountingProvider()
        provider = CachedEmbeddingProvider(source, cache_size=4, cache_ttl_seconds=60)
        first = provider.embed(["same", "same", "new"])
        second = provider.embed(["same", "new"])

        self.assertEqual(source.calls, [["same", "new"]])
        self.assertEqual(first, [[4.0, 1.0], [4.0, 1.0], [3.0, 1.0]])
        self.assertEqual(second, [[4.0, 1.0], [3.0, 1.0]])
        usage = provider.drain_usage()
        self.assertEqual(usage["provider_calls"], 1)
        self.assertEqual(usage["provider_texts"], 2)
        self.assertEqual(usage["cache_hits"], 2)
        self.assertNotIn("same", str(usage))


if __name__ == "__main__":
    unittest.main()
