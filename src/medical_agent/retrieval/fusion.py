"""Rank fusion helpers shared by lexical and vector retrieval backends."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def normalize_queries(queries: Iterable[Any], *, max_queries: int = 3) -> list[str]:
    """Normalize and de-duplicate query views within a bounded budget."""

    if max_queries < 1:
        return []
    normalized: list[str] = []
    for query in queries:
        value = " ".join(str(query).split())
        if value and value not in normalized:
            normalized.append(value)
        if len(normalized) >= max_queries:
            break
    return normalized


def fuse_ranked_results(
    query_results: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    limit: int = 8,
    max_per_document: int = 2,
    method: str = "rrf",
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse per-query ranked documents with reciprocal-rank fusion.

    Every backend supplies the same bounded result contract. The helper keeps
    provenance fields from the first result while adding query/rank metadata,
    so the application layer does not need to know which index produced it.
    """

    bounded_limit = max(1, min(int(limit), 128))
    bounded_per_document = max(1, min(int(max_per_document), bounded_limit))
    bounded_rrf_k = max(1, int(rrf_k))
    candidates: dict[str, dict[str, Any]] = {}
    for query, documents in query_results:
        for rank, document in enumerate(documents, start=1):
            if not isinstance(document, dict):
                continue
            key = str(document.get("id", ""))
            if not key:
                continue
            item = candidates.setdefault(
                key,
                {
                    **document,
                    "retrieval_score": 0.0,
                    "retrieval_queries": [],
                    "retrieval_ranks": {},
                },
            )
            item["retrieval_score"] += 1.0 / (bounded_rrf_k + rank)
            if query not in item["retrieval_queries"]:
                item["retrieval_queries"].append(query)
            item["retrieval_ranks"][query] = rank

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            float(item.get("retrieval_score", 0.0)),
            float(item.get("score", 0.0) or 0.0),
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
        item["retrieval_method"] = method
        item["retrieval_rank"] = rank
        item["score"] = round(float(item.get("retrieval_score", 0.0)), 6)
    return selected
