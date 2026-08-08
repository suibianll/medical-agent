"""Sparse/dense retrieval fusion with a single knowledge-base contract.

The runtime deliberately keeps lexical and vector retrieval as independent
backends.  This decorator combines their bounded ranked lists at the
composition boundary, so the task workflow does not need to know which
retrievers are enabled and either backend can still be tested in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .fusion import normalize_queries


class HybridKnowledgeBase:
    """Fuse sparse and dense candidates with weighted reciprocal-rank fusion."""

    def __init__(
        self,
        sparse_backend: Any,
        dense_backend: Any,
        *,
        sparse_weight: float = 0.45,
        dense_weight: float = 0.55,
        rrf_k: int = 60,
    ) -> None:
        if sparse_backend is None or dense_backend is None:
            raise ValueError("hybrid 检索必须同时配置稀疏和稠密后端。")
        try:
            sparse_value = float(sparse_weight)
            dense_value = float(dense_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("hybrid 检索权重必须是数字。") from exc
        if sparse_value < 0 or dense_value < 0 or sparse_value + dense_value <= 0:
            raise ValueError("hybrid 检索至少需要一个大于 0 的后端权重。")
        self._sparse = sparse_backend
        self._dense = dense_backend
        self.sparse_weight = sparse_value / (sparse_value + dense_value)
        self.dense_weight = dense_value / (sparse_value + dense_value)
        self.rrf_k = max(1, min(int(rrf_k), 1_000))

    @property
    def documents(self) -> list[dict[str, Any]]:
        documents = getattr(self._sparse, "documents", [])
        return list(documents) if isinstance(documents, (list, tuple)) else []

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        return self.search_many([query], limit=limit, max_per_document=1)

    @staticmethod
    def _backend_search(
        backend: Any,
        queries: list[str],
        *,
        limit: int,
        source_types: set[str] | None,
        max_per_document: int,
    ) -> list[dict[str, Any]]:
        search_many = getattr(backend, "search_many", None)
        if callable(search_many):
            try:
                values = search_many(
                    queries,
                    limit=limit,
                    source_types=source_types,
                    max_per_document=max_per_document,
                )
            except TypeError:
                # Keep compatibility with small custom ports that only accept
                # the original search_many(query, limit, max_per_document)
                # signature.
                try:
                    values = search_many(
                        queries,
                        limit=limit,
                        max_per_document=max_per_document,
                    )
                except TypeError:
                    values = search_many(queries, limit=limit)
            if not isinstance(values, (list, tuple)):
                return []
            return [value for value in values if isinstance(value, dict)]
        search = getattr(backend, "search", None)
        if not callable(search):
            return []
        results: list[dict[str, Any]] = []
        for query in queries:
            values = search(query, limit=limit)
            if not isinstance(values, (list, tuple)):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                if source_types and str(value.get("source_type", "built_in")) not in source_types:
                    continue
                results.append(value)
        return results

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
        bounded_limit = max(1, min(int(limit), 128))
        bounded_per_document = max(1, min(int(max_per_document), bounded_limit))
        per_backend_limit = min(max(bounded_limit * 2, 8), 128)
        sparse_results = self._backend_search(
            self._sparse,
            normalized_queries,
            limit=per_backend_limit,
            source_types=source_types,
            max_per_document=bounded_per_document,
        )
        dense_results = self._backend_search(
            self._dense,
            normalized_queries,
            limit=per_backend_limit,
            source_types=source_types,
            max_per_document=bounded_per_document,
        )

        candidates: dict[str, dict[str, Any]] = {}
        backend_results: tuple[str, Sequence[dict[str, Any]], float] = (
            "sparse",
            sparse_results,
            self.sparse_weight,
        )
        for backend_name, results, weight in (
            backend_results,
            ("dense", dense_results, self.dense_weight),
        ):
            for rank, document in enumerate(results, start=1):
                key = str(document.get("id", ""))
                if not key:
                    continue
                item = candidates.setdefault(key, {**document})
                if backend_name == "dense":
                    # Sparse metadata is preferred for human-facing source
                    # fields; dense-only fields are still copied in.
                    for field, value in document.items():
                        item.setdefault(field, value)
                try:
                    backend_score = float(
                        document.get("retrieval_score", document.get("score", 0.0)) or 0.0
                    )
                except (TypeError, ValueError):
                    backend_score = 0.0
                item[f"{backend_name}_score"] = backend_score
                item.setdefault("hybrid_backend_ranks", {})[backend_name] = rank
                sources = item.setdefault("retrieval_sources", [])
                if not isinstance(sources, list):
                    sources = []
                    item["retrieval_sources"] = sources
                if backend_name not in sources:
                    sources.append(backend_name)
                item["retrieval_score"] = float(item.get("retrieval_score", 0.0)) + weight / (
                    self.rrf_k + rank
                )
                queries_seen = item.setdefault("retrieval_queries", [])
                if not isinstance(queries_seen, list):
                    queries_seen = []
                    item["retrieval_queries"] = queries_seen
                raw_queries = document.get("retrieval_queries", normalized_queries)
                if not isinstance(raw_queries, (list, tuple)):
                    raw_queries = normalized_queries
                for query in raw_queries:
                    if query not in queries_seen:
                        queries_seen.append(query)

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                float(item.get("retrieval_score", 0.0)),
                float(item.get("dense_score", 0.0) or 0.0),
                float(item.get("sparse_score", 0.0) or 0.0),
                int(item.get("priority", 0) or 0),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        document_counts: dict[str, int] = {}
        for item in ranked:
            document_id = str(item.get("document_id", item.get("id", "")))
            if document_counts.get(document_id, 0) >= bounded_per_document:
                continue
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            selected.append(item)
            if len(selected) >= bounded_limit:
                break
        for rank, item in enumerate(selected, start=1):
            item["retrieval_method"] = "hybrid_rrf"
            item["retrieval_rank"] = rank
            item["score"] = round(float(item.get("retrieval_score", 0.0)), 6)
        return selected

    def import_text(self, *, name: str, content: str) -> dict[str, Any]:
        importer = getattr(self._sparse, "import_text", None)
        if not callable(importer):
            raise AttributeError("稀疏检索后端不支持知识导入。")
        # The dense backend is built over the same source in the composition
        # root; forwarding twice would duplicate imported chunks.
        return importer(name=name, content=content)

    def list_documents(self) -> list[dict[str, Any]]:
        lister = getattr(self._sparse, "list_documents", None)
        values = lister() if callable(lister) else []
        return list(values) if isinstance(values, (list, tuple)) else []

    def retrieval_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "backend": "hybrid",
            "fusion": "weighted_rrf",
            "sparse_weight": round(self.sparse_weight, 4),
            "dense_weight": round(self.dense_weight, 4),
            "rrf_k": self.rrf_k,
        }
        for key, backend in (("sparse", self._sparse), ("dense", self._dense)):
            getter = getattr(backend, "retrieval_metadata", None)
            try:
                value = getter() if callable(getter) else {}
            except Exception:  # noqa: BLE001 - diagnostics must never break retrieval
                value = {}
            if isinstance(value, dict):
                safe = {
                    field: value[field]
                    for field in ("backend", "provider", "name", "dimensions")
                    if field in value
                }
                if isinstance(value.get("embedding"), dict):
                    safe["embedding"] = {
                        field: value["embedding"][field]
                        for field in ("provider", "name", "dimensions")
                        if field in value["embedding"]
                    }
                metadata[key] = safe
                if key == "sparse" and isinstance(value.get("governance"), dict):
                    metadata["governance"] = value["governance"]
        return metadata

    def drain_embedding_usage(self) -> dict[str, int]:
        drain = getattr(self._dense, "drain_embedding_usage", None)
        if not callable(drain):
            return {}
        values = drain()
        return values if isinstance(values, dict) else {}
