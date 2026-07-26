"""Optional FAISS-backed knowledge retrieval and embedding providers.

The module has no hard dependency on FAISS. Importing the application remains
safe on a minimal installation; selecting the ``faiss`` backend produces an
actionable configuration error unless the optional vector dependencies are
installed.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..infrastructure.openai_client import normalize_base_url
from .fusion import fuse_ranked_results, normalize_queries


MAX_EMBEDDING_BATCH = 64
MAX_PROVIDER_RESPONSE_BYTES = 4_000_000


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

    def _fingerprint(self, documents: list[dict[str, Any]]) -> str:
        payload = {
            "documents": [
                {
                    "id": document.get("id"),
                    "document_id": document.get("document_id"),
                    "title": document.get("title"),
                    "text": document.get("text"),
                    "version": document.get("version"),
                }
                for document in documents
            ],
            "embedding": self._embedding_provider.runtime_metadata(),
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
            ids = [str(document.get("id", "")) for document in documents]
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
                            "ids": [str(document.get("id", "")) for document in documents],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except (OSError, TypeError, ValueError):
            # A cache write must never make an otherwise valid run fail.
            return

    def _ensure_index(self, documents: list[dict[str, Any]]) -> Any:
        fingerprint = self._fingerprint(documents)
        with self._lock:
            if self._index is not None and self._indexed_fingerprint == fingerprint:
                return self._index
            persisted = self._load_persisted(documents, fingerprint)
            if persisted is not None:
                self._index = persisted
                self._indexed_fingerprint = fingerprint
                self._indexed_documents = documents
                return persisted

            vectors = self._embedding_provider.embed(
                [self._document_text(document) for document in documents]
            )
            if len(vectors) != len(documents) or not vectors:
                raise EmbeddingProviderError("embedding provider 返回的向量数量无效。")
            matrix = self._numpy.asarray(vectors, dtype="float32")
            if len(matrix.shape) != 2 or matrix.shape[0] != len(documents):
                raise EmbeddingProviderError("embedding provider 返回了无效矩阵。")
            self._faiss.normalize_L2(matrix)
            index = self._faiss.IndexFlatIP(int(matrix.shape[1]))
            index.add(matrix)
            self._persist(index, documents, fingerprint)
            self._index = index
            self._indexed_fingerprint = fingerprint
            self._indexed_documents = documents
            return index

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
        documents = self.documents
        if not documents:
            return []
        index = self._ensure_index(documents)
        per_query_limit = min(max(int(limit) * 2, 8), len(documents))
        query_results: list[tuple[str, list[dict[str, Any]]]] = []
        for query in normalized_queries:
            vectors = self._embedding_provider.embed([query])
            if len(vectors) != 1:
                raise EmbeddingProviderError("embedding provider 未返回查询向量。")
            matrix = self._numpy.asarray(vectors, dtype="float32")
            self._faiss.normalize_L2(matrix)
            scores, indices = index.search(matrix, per_query_limit)
            results: list[dict[str, Any]] = []
            for similarity, document_index in zip(scores[0], indices[0], strict=False):
                index_value = int(document_index)
                if index_value < 0 or index_value >= len(documents):
                    continue
                document = documents[index_value]
                source_type = str(document.get("source_type", "built_in"))
                if source_types and source_type not in source_types:
                    continue
                results.append(
                    {
                        **document,
                        "score": float(similarity),
                        "vector_score": float(similarity),
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
        return {
            "backend": "faiss",
            "embedding": self._embedding_provider.runtime_metadata(),
            "index_path": str(self._index_path) if self._index_path else "",
        }
