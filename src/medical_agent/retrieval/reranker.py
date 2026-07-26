"""External reranker adapter with a bounded, provenance-preserving contract."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_RERANK_CANDIDATES = 64
MAX_PROVIDER_RESPONSE_BYTES = 2_000_000


class RerankerError(RuntimeError):
    """Raised when an external reranker cannot produce a valid ranking."""


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
