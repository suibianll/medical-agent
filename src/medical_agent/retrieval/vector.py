"""Optional FAISS-backed knowledge retrieval and embedding providers.

The module has no hard dependency on FAISS. Importing the application remains
safe on a minimal installation; selecting the ``faiss`` backend produces an
actionable configuration error unless the optional vector dependencies are
installed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock
from time import monotonic, perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..infrastructure.openai_client import normalize_base_url
from .fusion import fuse_ranked_results, normalize_queries


MAX_EMBEDDING_BATCH = 64
# Hosted embedding APIs commonly enforce a request-level token budget in
# addition to the number-of-items limit.  Keeping batches bounded by text size
# avoids HTTP 400 responses when a small number of long full-text articles are
# indexed together.  Long documents are split below so no single input is
# larger than the provider-safe window.
MAX_DOCUMENT_EMBED_CHARS = 8_000
MAX_EMBEDDING_BATCH_CHARS = 24_000
DOCUMENT_CHUNK_OVERLAP_CHARS = 400
MAX_PROVIDER_RESPONSE_BYTES = 4_000_000
MAX_EMBEDDING_CACHE_ENTRIES = 2_048
MAX_EMBEDDING_CACHE_TTL_SECONDS = 86_400


class RetrievalBackendUnavailable(RuntimeError):
    """Raised when a configured retrieval backend cannot be initialized."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider returns an invalid response."""


class HashEmbeddingProvider:
    """Deterministic local embeddings for tests and offline demonstrations.

    This is a stable feature-hashing fallback, not a semantic embedding model.
    Production deployments should configure an external embedding endpoint.
    """

    def __init__(self, *, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("本地哈希向量维度必须至少为 8。")
        self.dimensions = int(dimensions)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        compact = re.sub(r"\s+", "", str(value or "").lower())
        latin = set(re.findall(r"[a-z0-9._-]{2,}", compact))
        bigrams = {compact[index : index + 2] for index in range(len(compact) - 1)}
        return latin | bigrams

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in self._tokens(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                offset = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[offset] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "provider": "local-hash",
            "name": "feature-hash",
            "dimensions": str(self.dimensions),
        }


class CachedEmbeddingProvider:
    """Bounded text-embedding cache with safe usage counters.

    FAISS already caches its document index.  This decorator targets repeated
    query embeddings across tasks and repair rounds, batching only cache
    misses so provider calls and latency stay proportional to new inputs.
    """

    def __init__(
        self,
        provider: Any,
        *,
        cache_size: int = 256,
        cache_ttl_seconds: int = 600,
    ) -> None:
        if cache_size < 0:
            raise ValueError("embedding 缓存大小不能小于 0。")
        if cache_ttl_seconds < 0:
            raise ValueError("embedding 缓存 TTL 不能小于 0。")
        self._provider = provider
        self.cache_size = min(int(cache_size), MAX_EMBEDDING_CACHE_ENTRIES)
        self.cache_ttl_seconds = min(int(cache_ttl_seconds), MAX_EMBEDDING_CACHE_TTL_SECONDS)
        self._cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._lock = RLock()
        self._batches = 0
        self._provider_calls = 0
        self._requested_texts = 0
        self._provider_texts = 0
        self._cache_hits = 0
        self._failures = 0
        self._latency_ms = 0

    @staticmethod
    def _key(text: str) -> str:
        normalized = " ".join(str(text).split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def runtime_metadata(self) -> dict[str, str]:
        metadata_method = getattr(self._provider, "runtime_metadata", None)
        raw = metadata_method() if callable(metadata_method) else {}
        if not isinstance(raw, dict):
            raw = {}
        metadata = {
            str(key): str(value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, (str, int, float, bool))
        }
        metadata.update(
            {
                "cache": "bounded",
                "cache_size": str(self.cache_size),
                "cache_ttl_seconds": str(self.cache_ttl_seconds),
            }
        )
        return metadata

    def _get_cached(self, key: str) -> list[float] | None:
        if self.cache_size <= 0 or self.cache_ttl_seconds <= 0:
            return None
        now = monotonic()
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                return None
            created, vector = value
            if now - created > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return list(vector)

    def _put_cached(self, key: str, vector: list[float]) -> None:
        if self.cache_size <= 0 or self.cache_ttl_seconds <= 0:
            return
        with self._lock:
            self._cache[key] = (monotonic(), list(vector))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text) for text in texts]
        if not values:
            return []
        if len(values) > MAX_EMBEDDING_BATCH:
            raise EmbeddingProviderError(
                f"单次 embedding 请求最多支持 {MAX_EMBEDDING_BATCH} 个文本。"
            )
        with self._lock:
            self._batches += 1
            self._requested_texts += len(values)
        keys = [self._key(value) for value in values]
        result: list[list[float] | None] = [None] * len(values)
        misses: list[tuple[int, str, str]] = []
        missing_keys: set[str] = set()
        for index, (value, key) in enumerate(zip(values, keys, strict=False)):
            cached = self._get_cached(key)
            if cached is not None:
                result[index] = cached
                continue
            if key not in missing_keys:
                missing_keys.add(key)
                misses.append((index, key, value))

        if misses:
            started = perf_counter()
            with self._lock:
                self._provider_calls += 1
                self._provider_texts += len(misses)
            try:
                vectors = self._provider.embed([item[2] for item in misses])
                if len(vectors) != len(misses):
                    raise EmbeddingProviderError("embedding provider 返回的向量数量无效。")
                generated: dict[str, list[float]] = {}
                for (_index, key, _value), vector in zip(misses, vectors, strict=False):
                    if not isinstance(vector, list) or not vector:
                        raise EmbeddingProviderError("embedding provider 返回了空向量。")
                    converted = [float(value) for value in vector]
                    if any(not math.isfinite(value) for value in converted):
                        raise EmbeddingProviderError("embedding provider 返回了非法向量值。")
                    generated[key] = converted
                    self._put_cached(key, converted)
            except Exception:
                with self._lock:
                    self._failures += 1
                raise
            finally:
                with self._lock:
                    self._latency_ms += max(0, int(round((perf_counter() - started) * 1000)))
            for index, key, _value in misses:
                result[index] = list(generated[key])
            for index, key in enumerate(keys):
                if result[index] is None and key in generated:
                    result[index] = list(generated[key])

        if any(vector is None for vector in result):
            raise EmbeddingProviderError("embedding 缓存未能生成完整结果。")
        return [list(vector) for vector in result if vector is not None]

    def drain_usage(self) -> dict[str, int]:
        with self._lock:
            usage = {
                "batches": self._batches,
                "provider_calls": self._provider_calls,
                "requested_texts": self._requested_texts,
                "provider_texts": self._provider_texts,
                "cache_hits": self._cache_hits,
                "failures": self._failures,
                "latency_ms": self._latency_ms,
                "cache_size": self.cache_size,
            }
            self._batches = 0
            self._provider_calls = 0
            self._requested_texts = 0
            self._provider_texts = 0
            self._cache_hits = 0
            self._failures = 0
            self._latency_ms = 0
            return usage


class OpenAICompatibleEmbeddingProvider:
    """Transport-only client for OpenAI-compatible ``/embeddings`` APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "openai-compatible",
        dimensions: int | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key.strip():
            raise ValueError("缺少 embedding API Key。")
        if not model.strip():
            raise ValueError("缺少 embedding 模型名称。")
        if timeout_seconds <= 0:
            raise ValueError("embedding 服务超时时间必须大于 0。")
        self._api_key = api_key.strip()
        self.provider = provider.strip() or "openai-compatible"
        self.base_url = normalize_base_url(base_url, self.provider)
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.dimensions = int(dimensions) if dimensions else None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text) for text in texts]
        if not values:
            return []
        if len(values) > MAX_EMBEDDING_BATCH:
            raise EmbeddingProviderError(
                f"单次 embedding 请求最多支持 {MAX_EMBEDDING_BATCH} 个文本。"
            )
        payload = {"model": self.model, "input": values}
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                encoded = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                if len(encoded) > MAX_PROVIDER_RESPONSE_BYTES:
                    raise EmbeddingProviderError("embedding 服务响应超过大小限制。")
                raw = encoded.decode("utf-8")
        except HTTPError as exc:
            raise EmbeddingProviderError(f"embedding 服务返回 HTTP {exc.code}。") from None
        except URLError as exc:
            raise EmbeddingProviderError("无法连接到 embedding 服务。") from exc
        except TimeoutError as exc:
            raise EmbeddingProviderError("embedding 服务请求超时。") from exc

        try:
            response_payload: Any = json.loads(raw)
            raw_data = response_payload["data"]
            if not isinstance(raw_data, list):
                raise TypeError
            ordered = sorted(
                raw_data,
                key=lambda item: int(item.get("index", 0))
                if isinstance(item, dict)
                else 0,
            )
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise EmbeddingProviderError("embedding 服务返回了无法识别的响应格式。") from None

        if len(vectors) != len(values) or any(
            not isinstance(vector, list) or not vector for vector in vectors
        ):
            raise EmbeddingProviderError("embedding 服务返回的向量数量无效。")
        normalized: list[list[float]] = []
        for vector in vectors:
            try:
                converted = [float(value) for value in vector]
            except (TypeError, ValueError):
                raise EmbeddingProviderError("embedding 服务返回了非数值向量。") from None
            if any(not math.isfinite(value) for value in converted):
                raise EmbeddingProviderError("embedding 服务返回了非法向量值。")
            normalized.append(converted)
        dimension = len(normalized[0])
        if any(len(vector) != dimension for vector in normalized):
            raise EmbeddingProviderError("embedding 服务返回的向量维度不一致。")
        if self.dimensions and dimension != self.dimensions:
            raise EmbeddingProviderError("embedding 服务返回的向量维度与配置不一致。")
        self.dimensions = dimension
        return normalized

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "name": self.model,
            "dimensions": str(self.dimensions or "unknown"),
        }


def _load_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RetrievalBackendUnavailable(
            "FAISS 检索需要安装可选依赖 faiss-cpu；请安装项目的 vector extra。"
        ) from exc
    return faiss


def _load_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RetrievalBackendUnavailable(
            "FAISS 检索需要安装可选依赖 numpy。"
        ) from exc
    return np


class FaissKnowledgeBase:
    """Knowledge-base decorator that replaces lexical ranking with FAISS."""

    def __init__(
        self,
        source: Any,
        *,
        embedding_provider: Any,
        index_path: Path | None = None,
        faiss_module: Any | None = None,
        numpy_module: Any | None = None,
    ) -> None:
        self._source = source
        self._embedding_provider = embedding_provider
        self._faiss = faiss_module or _load_faiss()
        self._numpy = numpy_module or _load_numpy()
        self._index_path = Path(index_path).expanduser() if index_path else None
        self._lock = RLock()
        self._index: Any | None = None
        self._indexed_fingerprint = ""
        self._indexed_documents: list[dict[str, Any]] = []

    @property
    def documents(self) -> list[dict[str, Any]]:
        return self._source.documents

    def import_text(self, *, name: str, content: str) -> dict[str, Any]:
        result = self._source.import_text(name=name, content=content)
        with self._lock:
            self._index = None
            self._indexed_fingerprint = ""
            self._indexed_documents = []
        return result

    def list_documents(self) -> list[dict[str, Any]]:
        return self._source.list_documents()

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        return f"{document.get('title', '')} {document.get('text', '')}"

    @staticmethod
    def _index_document_id(document: dict[str, Any]) -> str:
        """Return a stable ID for persisted index validation.

        Chunked entries intentionally keep their public ``id`` equal to the
        source document ID so benchmark gold IDs and evidence citations remain
        unchanged.  The private chunk ID is only used to validate a persisted
        FAISS index against the exact vector rows it contains.
        """

        chunk_id = document.get("_faiss_chunk_id")
        if isinstance(chunk_id, str) and chunk_id.strip():
            return chunk_id.strip()
        return str(document.get("id", ""))

    @classmethod
    def _prepare_index_documents(
        cls, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Split oversized documents while preserving source-level IDs.

        The evaluator and application consume source document IDs.  Chunks
        therefore inherit ``id``/``document_id`` and carry a private chunk ID;
        fusion deduplicates them back to one source document in the returned
        ranking.  Titles are repeated in each chunk to retain high-signal
        metadata when the body starts far from the article heading.
        """

        prepared: list[dict[str, Any]] = []
        overlap = min(DOCUMENT_CHUNK_OVERLAP_CHARS, MAX_DOCUMENT_EMBED_CHARS // 4)
        for document in documents:
            title = str(document.get("title", "")).strip()
            body = str(document.get("text", "")).strip()
            prefix = f"{title} " if title else ""
            if len(prefix) + len(body) <= MAX_DOCUMENT_EMBED_CHARS:
                prepared.append(dict(document))
                continue

            chunk_size = max(1, MAX_DOCUMENT_EMBED_CHARS - len(prefix))
            step = max(1, chunk_size - overlap)
            chunks = max(1, (len(body) + step - 1) // step)
            source_id = str(document.get("id", ""))
            for chunk_index, start in enumerate(range(0, len(body), step)):
                chunk_text = body[start : start + chunk_size]
                if not chunk_text:
                    continue
                chunk = dict(document)
                chunk["text"] = chunk_text
                chunk["_faiss_chunk_id"] = f"{source_id}::chunk-{chunk_index}"
                chunk["_faiss_chunk_index"] = chunk_index
                chunk["_faiss_chunk_count"] = chunks
                prepared.append(chunk)
        return prepared

    def _fingerprint(self, documents: list[dict[str, Any]]) -> str:
        metadata_method = getattr(self._embedding_provider, "runtime_metadata", None)
        embedding_metadata = metadata_method() if callable(metadata_method) else {}
        if not isinstance(embedding_metadata, dict):
            embedding_metadata = {}
        payload = {
            "documents": [
                {
                    "id": document.get("id"),
                    "document_id": document.get("document_id"),
                    "title": document.get("title"),
                    "text": document.get("text"),
                    "version": document.get("version"),
                    "source_type": document.get("source_type", "built_in"),
                    "status": document.get("status", document.get("publication_status", "")),
                    "priority": document.get("priority", 0),
                    "synthetic": document.get("synthetic", False),
                    "governance": document.get("governance", {}),
                }
                for document in documents
            ],
            "embedding": {
                key: value
                for key, value in embedding_metadata.items()
                if key not in {"cache", "cache_size", "cache_ttl_seconds"}
            },
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _metadata_path(self) -> Path | None:
        if self._index_path is None:
            return None
        return self._index_path.with_name(f"{self._index_path.name}.meta.json")

    def _load_persisted(
        self, documents: list[dict[str, Any]], fingerprint: str
    ) -> Any | None:
        if self._index_path is None or not self._index_path.is_file():
            return None
        metadata_path = self._metadata_path()
        if metadata_path is None or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ids = [self._index_document_id(document) for document in documents]
            if metadata.get("fingerprint") != fingerprint or metadata.get("ids") != ids:
                return None
            return self._faiss.read_index(str(self._index_path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _persist(self, index: Any, documents: list[dict[str, Any]], fingerprint: str) -> None:
        if self._index_path is None:
            return
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(index, str(self._index_path))
            metadata_path = self._metadata_path()
            if metadata_path is not None:
                metadata_path.write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "ids": [
                                self._index_document_id(document)
                                for document in documents
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except (OSError, TypeError, ValueError):
            # A cache write must never make an otherwise valid run fail.
            return

    def _ensure_index(
        self, documents: list[dict[str, Any]]
    ) -> tuple[Any, list[dict[str, Any]]]:
        index_documents = self._prepare_index_documents(documents)
        fingerprint = self._fingerprint(index_documents)
        with self._lock:
            if self._index is not None and self._indexed_fingerprint == fingerprint:
                return self._index, list(self._indexed_documents)
            persisted = self._load_persisted(index_documents, fingerprint)
            if persisted is not None:
                self._index = persisted
                self._indexed_fingerprint = fingerprint
                self._indexed_documents = index_documents
                return persisted, list(index_documents)

            document_texts = [self._document_text(document) for document in index_documents]
            vectors: list[list[float]] = []
            batch: list[str] = []
            batch_chars = 0
            for text in document_texts:
                text_chars = len(text)
                if batch and (
                    len(batch) >= MAX_EMBEDDING_BATCH
                    or batch_chars + text_chars > MAX_EMBEDDING_BATCH_CHARS
                ):
                    vectors.extend(self._embedding_provider.embed(batch))
                    batch = []
                    batch_chars = 0
                batch.append(text)
                batch_chars += text_chars
            if batch:
                vectors.extend(self._embedding_provider.embed(batch))
            if len(vectors) != len(index_documents) or not vectors:
                raise EmbeddingProviderError("embedding provider 返回的向量数量无效。")
            matrix = self._numpy.asarray(vectors, dtype="float32")
            if len(matrix.shape) != 2 or matrix.shape[0] != len(index_documents):
                raise EmbeddingProviderError("embedding provider 返回了无效矩阵。")
            self._faiss.normalize_L2(matrix)
            index = self._faiss.IndexFlatIP(int(matrix.shape[1]))
            index.add(matrix)
            self._persist(index, index_documents, fingerprint)
            self._index = index
            self._indexed_fingerprint = fingerprint
            self._indexed_documents = index_documents
            return index, list(index_documents)

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        return self.search_many([query], limit=limit, max_per_document=1)

    def search_many(
        self,
        queries: list[str],
        *,
        limit: int = 8,
        source_types: set[str] | None = None,
        max_per_document: int = 2,
    ) -> list[dict[str, Any]]:
        normalized_queries = normalize_queries(queries)
        if not normalized_queries:
            return []
        eligible_documents = getattr(self._source, "eligible_documents", None)
        documents = (
            eligible_documents()
            if callable(eligible_documents)
            else self.documents
        )
        if not documents:
            return []
        index, index_documents = self._ensure_index(documents)
        per_query_limit = min(max(int(limit) * 2, 8), len(index_documents))
        query_vectors = self._embedding_provider.embed(normalized_queries)
        if len(query_vectors) != len(normalized_queries):
            raise EmbeddingProviderError("embedding provider 返回的查询向量数量无效。")
        query_matrix = self._numpy.asarray(query_vectors, dtype="float32")
        if len(query_matrix.shape) != 2 or query_matrix.shape[0] != len(normalized_queries):
            raise EmbeddingProviderError("embedding provider 返回了无效查询矩阵。")
        self._faiss.normalize_L2(query_matrix)
        scores, indices = index.search(query_matrix, per_query_limit)
        query_results: list[tuple[str, list[dict[str, Any]]]] = []
        for query_index, query in enumerate(normalized_queries):
            results: list[dict[str, Any]] = []
            for similarity, document_index in zip(
                scores[query_index], indices[query_index], strict=False
            ):
                index_value = int(document_index)
                if index_value < 0 or index_value >= len(index_documents):
                    continue
                document = index_documents[index_value]
                public_document = {
                    key: value
                    for key, value in document.items()
                    if not key.startswith("_faiss_")
                }
                source_type = str(public_document.get("source_type", "built_in"))
                if source_types and source_type not in source_types:
                    continue
                results.append(
                    {
                        **public_document,
                        "score": float(similarity),
                        "vector_score": float(similarity),
                        "relevance_score": max(0.0, min(float(similarity), 1.0)),
                    }
                )
            query_results.append((query, results))
        return fuse_ranked_results(
            query_results,
            limit=limit,
            max_per_document=max_per_document,
            method="rrf_faiss",
        )

    def retrieval_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "backend": "faiss",
            "embedding": self._embedding_provider.runtime_metadata(),
            "index_path": str(self._index_path) if self._index_path else "",
        }
        source_metadata_method = getattr(self._source, "retrieval_metadata", None)
        try:
            source_metadata = source_metadata_method() if callable(source_metadata_method) else {}
        except Exception:  # noqa: BLE001 - diagnostics must never break retrieval
            source_metadata = {}
        if isinstance(source_metadata, dict) and isinstance(
            source_metadata.get("governance"), dict
        ):
            metadata["governance"] = source_metadata["governance"]
        return metadata

    def drain_embedding_usage(self) -> dict[str, int]:
        """Drain provider counters without exposing text or vector values."""

        drain = getattr(self._embedding_provider, "drain_usage", None)
        if not callable(drain):
            return {}
        try:
            usage = drain()
        except Exception:  # noqa: BLE001 - metrics are best effort
            return {}
        if not isinstance(usage, dict):
            return {}
        return {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(key, str) and isinstance(value, int) and value >= 0
        }
