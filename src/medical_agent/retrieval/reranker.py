"""External reranker adapter with a bounded, provenance-preserving contract."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from time import monotonic, perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_RERANK_CANDIDATES = 64
MAX_PROVIDER_RESPONSE_BYTES = 2_000_000
MAX_RERANK_CACHE_ENTRIES = 1_024
MAX_RERANK_CACHE_TTL_SECONDS = 86_400


class RerankerError(RuntimeError):
    """Raised when an external reranker cannot produce a valid ranking."""


def _candidate_key(document: dict[str, Any], index: int) -> str:
    """Build a non-reversible cache identity for one candidate.

    The cache must not use raw medical text as a telemetry key.  Including a
    stable identifier and a content digest makes changes invalidate old
    rankings while keeping the key opaque and bounded.
    """

    stable_id = str(
        document.get("id")
        or document.get("document_id")
        or document.get("locator")
        or f"candidate-{index}"
    )
    payload = {
        "id": stable_id[:240],
        "title": str(document.get("title", ""))[:500],
        "text_sha256": sha256(str(document.get("text", "")).encode("utf-8")).hexdigest(),
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class RerankerRun:
    """Per-workflow budget and cache around a reranker adapter.

    A workflow can execute tasks concurrently and may rerun failed tasks.  The
    adapter itself remains stateless while this object provides a thread-safe
    run-level budget, short-lived ranking cache and safe cost telemetry.
    """

    def __init__(
        self,
        reranker: Any,
        *,
        max_calls: int = 8,
        min_candidates: int = 2,
        cache_size: int = 128,
        cache_ttl_seconds: int = 300,
    ) -> None:
        if max_calls < 0:
            raise ValueError("reranker 每轮最大调用次数不能小于 0。")
        if min_candidates < 1:
            raise ValueError("reranker 最小候选数必须大于 0。")
        if cache_size < 0:
            raise ValueError("reranker 缓存大小不能小于 0。")
        if cache_ttl_seconds < 0:
            raise ValueError("reranker 缓存 TTL 不能小于 0。")
        self._reranker = reranker
        self.max_calls = int(max_calls)
        self.min_candidates = int(min_candidates)
        self.cache_size = min(int(cache_size), MAX_RERANK_CACHE_ENTRIES)
        self.cache_ttl_seconds = min(int(cache_ttl_seconds), MAX_RERANK_CACHE_TTL_SECONDS)
        self._cache: OrderedDict[str, tuple[float, list[tuple[str, float]]]] = OrderedDict()
        self._lock = RLock()
        self._attempts = 0
        self._external_calls = 0
        self._cache_hits = 0
        self._skipped_budget = 0
        self._skipped_min_candidates = 0
        self._failures = 0
        self._candidate_count = 0
        self._latency_ms = 0

    def runtime_metadata(self) -> dict[str, str]:
        metadata = getattr(self._reranker, "runtime_metadata", None)
        raw = metadata() if callable(metadata) else {}
        if not isinstance(raw, dict):
            raw = {}
        result = {
            str(key): str(value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, (str, int, float, bool))
        }
        result.update({
            "max_calls_per_run": str(self.max_calls),
            "min_candidates": str(self.min_candidates),
        })
        return result

    @staticmethod
    def _cache_key(query: str, candidate_keys: list[str], limit: int) -> str:
        payload = {
            "query": " ".join(str(query).split())[:1_000],
            "candidates": candidate_keys,
            "limit": limit,
        }
        return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_get(
        self, key: str, candidates: list[dict[str, Any]], candidate_keys: list[str]
    ) -> list[dict[str, Any]] | None:
        if self.cache_size <= 0 or self.cache_ttl_seconds <= 0:
            return None
        now = monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            created, ranked = cached
            if now - created > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            by_key = {candidate_key: document for candidate_key, document in zip(candidate_keys, candidates, strict=False)}
            if any(candidate_key not in by_key for candidate_key, _score in ranked):
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            self._cache_hits += 1
        return [
            {
                **by_key[candidate_key],
                "rerank_score": round(score, 6),
                "retrieval_method": "external_reranker",
                "rerank_cached": True,
            }
            for candidate_key, score in ranked
        ]

    def _cache_put(self, key: str, ranked: list[tuple[str, float]]) -> None:
        if self.cache_size <= 0 or self.cache_ttl_seconds <= 0 or not ranked:
            return
        with self._lock:
            self._cache[key] = (monotonic(), list(ranked))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def rerank(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        candidates = [document for document in documents if isinstance(document, dict)][:MAX_RERANK_CANDIDATES]
        if len(candidates) < self.min_candidates:
            with self._lock:
                self._skipped_min_candidates += 1
            return []
        bounded_limit = max(1, min(int(limit), len(candidates)))
        candidate_keys = [_candidate_key(document, index) for index, document in enumerate(candidates)]
        cache_key = self._cache_key(query, candidate_keys, bounded_limit)
        cached = self._cache_get(cache_key, candidates, candidate_keys)
        if cached is not None:
            return cached
        with self._lock:
            if self._attempts >= self.max_calls:
                self._skipped_budget += 1
                return []
            self._attempts += 1
            self._candidate_count += len(candidates)

        started = perf_counter()
        with self._lock:
            self._external_calls += 1
        try:
            result = self._reranker.rerank(
                query=query,
                documents=candidates,
                limit=bounded_limit,
            )
            if not isinstance(result, list):
                raise RerankerError("reranker 适配器返回了无效结果。")
            ranked: list[tuple[str, float]] = []
            normalized: list[dict[str, Any]] = []
            by_key = {
                candidate_key: candidate
                for candidate_key, candidate in zip(candidate_keys, candidates, strict=False)
            }
            seen_keys: set[str] = set()
            for item in result[:bounded_limit]:
                if not isinstance(item, dict):
                    continue
                matched_key = next(
                    (
                        candidate_key
                        for candidate_key, candidate in zip(candidate_keys, candidates, strict=False)
                        if candidate.get("id") == item.get("id")
                        and (
                            "document_id" not in item
                            or item.get("document_id") == candidate.get("document_id")
                        )
                        and (
                            "text" not in item
                            or item.get("text") == candidate.get("text")
                        )
                    ),
                    None,
                )
                if matched_key is None or matched_key not in by_key:
                    continue
                try:
                    score = float(item.get("rerank_score", item.get("score", 0.0)))
                except (TypeError, ValueError):
                    score = 0.0
                if math.isfinite(score):
                    if matched_key in seen_keys:
                        continue
                    seen_keys.add(matched_key)
                    ranked.append((matched_key, score))
                    normalized_item = {
                        **by_key[matched_key],
                        "rerank_score": round(score, 6),
                        "retrieval_method": "external_reranker",
                        "rerank_cached": False,
                    }
                    for field in ("rerank_provider", "rerank_model"):
                        value = item.get(field)
                        if isinstance(value, str) and value.strip():
                            normalized_item[field] = value[:120]
                    normalized.append(normalized_item)
            if ranked:
                self._cache_put(cache_key, ranked)
            return normalized
        except Exception:
            with self._lock:
                self._failures += 1
            raise
        finally:
            elapsed = max(0, int(round((perf_counter() - started) * 1000)))
            with self._lock:
                self._latency_ms += elapsed

    def usage(self) -> dict[str, Any]:
        with self._lock:
            return {
                "reranker": {
                    "attempts": self._attempts,
                    "external_calls": self._external_calls,
                    "cache_hits": self._cache_hits,
                    "skipped_budget": self._skipped_budget,
                    "skipped_min_candidates": self._skipped_min_candidates,
                    "failures": self._failures,
                    "candidate_count": self._candidate_count,
                    "latency_ms": self._latency_ms,
                    "max_calls_per_run": self.max_calls,
                    "cache_size": self.cache_size,
                }
            }


class ExternalApiReranker:
    """Call a Cohere/Jina-compatible rerank endpoint.

    The transport accepts any API that returns ``results`` containing a
    document ``index`` (or ``document_index``) and a numeric
    ``relevance_score`` (or ``score``). Unknown fields are ignored.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        model: str = "",
        provider: str = "generic",
        timeout_seconds: int = 30,
        top_n: int = 8,
        auth_header: str = "",
    ) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint.startswith(("https://", "http://")):
            raise ValueError("reranker endpoint 必须以 http:// 或 https:// 开头。")
        if timeout_seconds <= 0:
            raise ValueError("reranker 服务超时时间必须大于 0。")
        if top_n < 1:
            raise ValueError("reranker top_n 必须大于 0。")
        self.endpoint = endpoint
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.provider = provider.strip().lower() or "generic"
        self.timeout_seconds = timeout_seconds
        self.top_n = min(int(top_n), MAX_RERANK_CANDIDATES)
        self.auth_header = auth_header.strip()

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "name": self.model or "external-reranker",
            "mode": "external",
        }

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        if not self._api_key:
            return headers
        header = self.auth_header
        if not header:
            header = "X-API-Key" if self.provider in {"cohere", "jina"} else "Authorization"
        value = self._api_key if header.lower() == "x-api-key" else f"Bearer {self._api_key}"
        headers[header] = value
        return headers

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        title = str(document.get("title", "")).strip()
        text = str(document.get("text", "")).strip()
        return f"{title}\n{text}".strip()

    @staticmethod
    def _result_index(item: dict[str, Any]) -> int | None:
        raw_index = item.get("index", item.get("document_index"))
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    @staticmethod
    def _result_score(item: dict[str, Any]) -> float | None:
        raw_score = item.get("relevance_score", item.get("score"))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def rerank(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        normalized_query = " ".join(str(query).split())
        if not normalized_query or not documents:
            return []
        candidates = [document for document in documents[:MAX_RERANK_CANDIDATES] if document]
        if not candidates:
            return []
        bounded_limit = max(1, min(int(limit), self.top_n, len(candidates)))
        payload: dict[str, Any] = {
            "query": normalized_query,
            "documents": [self._document_text(document) for document in candidates],
            "top_n": bounded_limit,
            "return_documents": False,
        }
        if self.model:
            payload["model"] = self.model
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                encoded = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                if len(encoded) > MAX_PROVIDER_RESPONSE_BYTES:
                    raise RerankerError("reranker 服务响应超过大小限制。")
                raw = encoded.decode("utf-8")
        except HTTPError as exc:
            raise RerankerError(f"reranker 服务返回 HTTP {exc.code}。") from None
        except URLError as exc:
            raise RerankerError("无法连接到 reranker 服务。") from exc
        except TimeoutError as exc:
            raise RerankerError("reranker 服务请求超时。") from exc

        try:
            response_payload: Any = json.loads(raw)
            raw_results = response_payload.get("results", response_payload.get("data"))
            if not isinstance(raw_results, list):
                raise TypeError
        except (AttributeError, TypeError, json.JSONDecodeError):
            raise RerankerError("reranker 服务返回了无法识别的响应格式。") from None

        ranked: list[tuple[int, float]] = []
        seen_indices: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            index = self._result_index(item)
            score = self._result_score(item)
            if index is None or score is None or index >= len(candidates) or index in seen_indices:
                continue
            seen_indices.add(index)
            ranked.append((index, score))
        if not ranked:
            raise RerankerError("reranker 服务未返回有效排序结果。")
        ranked.sort(key=lambda item: (-item[1], item[0]))
        result: list[dict[str, Any]] = []
        for index, score in ranked[:bounded_limit]:
            result.append(
                {
                    **candidates[index],
                    "rerank_score": round(score, 6),
                    "rerank_provider": self.provider,
                    "rerank_model": self.model,
                    "retrieval_method": "external_reranker",
                }
            )
        return result
