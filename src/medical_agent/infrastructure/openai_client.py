"""Minimal synchronous client for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import json
import random
from http.client import HTTPResponse, RemoteDisconnected
from threading import local
from time import monotonic, perf_counter, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_PROVIDER_RESPONSE_BYTES = 2_000_000
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


class ModelProviderError(RuntimeError):
    """Sanitized provider error that deliberately never includes credentials."""


def normalize_base_url(base_url: str, provider: str) -> str:
    """Normalize a provider base URL without rewriting an existing v1 path."""

    value = base_url.strip().rstrip("/")
    if not value.startswith(("https://", "http://")):
        raise ValueError("模型服务 URL 必须以 http:// 或 https:// 开头。")
    if value.endswith(("/compatible-mode/v1", "/v1")):
        return value
    suffix = "/compatible-mode/v1" if provider == "aliyun-model-studio" else "/v1"
    return f"{value}{suffix}"


class OpenAIChatClient:
    """Transport-only client; it owns no medical prompts or workflow logic."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str,
        timeout_seconds: int = 90,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        stream: bool = False,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        retry_max_backoff_seconds: float = 4.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("缺少模型 API Key。")
        if not model or not model.strip():
            raise ValueError("缺少模型名称。")
        self._api_key = api_key.strip()
        if not provider or not provider.strip():
            raise ValueError("缺少模型供应商名称。")
        if timeout_seconds <= 0:
            raise ValueError("模型服务超时时间必须大于 0。")
        if max_retries < 0:
            raise ValueError("模型服务重试次数不能小于 0。")
        if retry_backoff_seconds < 0:
            raise ValueError("模型服务重试退避不能小于 0。")
        if retry_max_backoff_seconds < retry_backoff_seconds:
            raise ValueError("模型服务最大重试退避不能小于初始退避。")
        self.provider = provider.strip()
        self.base_url = normalize_base_url(base_url, self.provider)
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.stream = bool(stream)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.retry_max_backoff_seconds = float(retry_max_backoff_seconds)
        # A client may be shared by concurrent runs. Thread-local storage
        # keeps a worker's metrics attached to the call that produced them.
        self._thread_state = local()

    def _record_metrics(
        self,
        *,
        stage: str,
        started: float,
        success: bool,
        usage: Any = None,
        retry_count: int = 0,
        error_type: str | None = None,
    ) -> None:
        usage = usage if isinstance(usage, dict) else {}

        def nonnegative_int(value: Any) -> int | None:
            return value if isinstance(value, int) and value >= 0 else None

        prompt_tokens = nonnegative_int(usage.get("prompt_tokens"))
        output_tokens = nonnegative_int(usage.get("completion_tokens"))
        total_tokens = nonnegative_int(usage.get("total_tokens"))
        details = usage.get("prompt_tokens_details")
        cached_tokens = nonnegative_int(
            details.get("cached_tokens") if isinstance(details, dict) else None
        )
        completion_details = usage.get("completion_tokens_details")
        reasoning_tokens = nonnegative_int(
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        )
        metric: dict[str, Any] = {
            "stage": stage,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": max(0, round((perf_counter() - started) * 1000)),
            "success": bool(success),
            "retry_count": max(0, int(retry_count)),
            "attempts": max(1, int(retry_count) + 1),
        }
        if error_type:
            metric["error_type"] = str(error_type)[:80]
        for key, value in (
            ("input_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            ("total_tokens", total_tokens),
            ("cached_tokens", cached_tokens),
            ("reasoning_tokens", reasoning_tokens),
        ):
            if value is not None:
                metric[key] = value
        metrics = getattr(self._thread_state, "metrics", None)
        if not isinstance(metrics, list):
            metrics = []
            self._thread_state.metrics = metrics
        metrics.append(metric)

    def drain_call_metrics(self) -> list[dict[str, Any]]:
        metrics = getattr(self._thread_state, "metrics", [])
        self._thread_state.metrics = []
        return [dict(item) for item in metrics if isinstance(item, dict)]

    @staticmethod
    def _read_response_bytes(response: Any, *, deadline: float) -> bytes:
        """Read a bounded JSON body with a wall-clock deadline.

        ``HTTPResponse.read(limit)`` can wait indefinitely when a provider
        trickles bytes before the socket idle timeout.  Reading bounded chunks
        and refreshing the socket timeout with the remaining deadline makes
        the configured request timeout a true upper bound.  Lightweight test
        doubles keep the legacy single-read path.
        """

        if not isinstance(response, HTTPResponse):
            encoded = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
            if len(encoded) > MAX_PROVIDER_RESPONSE_BYTES:
                raise ModelProviderError("模型服务响应超过大小限制。")
            return encoded
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("provider response deadline exceeded")
            raw_socket = getattr(response, "_sock", None)
            if raw_socket is not None and hasattr(raw_socket, "settimeout"):
                raw_socket.settimeout(max(0.1, remaining))
            chunk = response.read(min(64 * 1024, MAX_PROVIDER_RESPONSE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PROVIDER_RESPONSE_BYTES:
                raise ModelProviderError("模型服务响应超过大小限制。")
        return b"".join(chunks)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1200,
        temperature: float = 0.1,
        stage: str = "unknown",
        enable_thinking_override: bool | None = None,
        thinking_budget_override: int | None = None,
    ) -> str:
        started = perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.stream,
        }
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        effective_thinking = (
            self.enable_thinking
            if enable_thinking_override is None
            else enable_thinking_override
        )
        effective_budget = (
            self.thinking_budget
            if thinking_budget_override is None
            else thinking_budget_override
        )
        if self.provider == "openrouter":
            reasoning: dict[str, Any] = {}
            if effective_thinking is not None:
                reasoning["enabled"] = effective_thinking
            if effective_budget is not None and (
                effective_thinking or enable_thinking_override is None
            ):
                reasoning["max_tokens"] = effective_budget
            if effective_thinking:
                # The workflow never consumes or persists hidden chain-of-thought.
                reasoning["exclude"] = True
            if reasoning:
                payload["reasoning"] = reasoning
        else:
            if effective_thinking is not None:
                payload["enable_thinking"] = effective_thinking
            if effective_budget is not None and (
                effective_thinking or enable_thinking_override is None
            ):
                payload["thinking_budget"] = effective_budget
        deadline = monotonic() + self.timeout_seconds
        retry_count = 0
        response_payload: dict[str, Any] = {}
        content = ""
        for attempt in range(self.max_retries + 1):
            remaining = deadline - monotonic()
            if remaining <= 0:
                self._record_metrics(
                    stage=stage,
                    started=started,
                    success=False,
                    retry_count=retry_count,
                    error_type="timeout",
                )
                raise ModelProviderError("模型服务请求超时。") from None
            request = Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=remaining) as response:  # noqa: S310
                    if self.stream:
                        content, response_usage = self._read_stream_response(
                            response,
                            deadline=deadline,
                        )
                        response_payload = {"usage": response_usage}
                    else:
                        encoded = self._read_response_bytes(
                            response,
                            deadline=deadline,
                        )
                        raw = encoded.decode("utf-8")
                break
            except ModelProviderError:
                self._record_metrics(
                    stage=stage,
                    started=started,
                    success=False,
                    retry_count=retry_count,
                    error_type="response",
                )
                raise
            except HTTPError as exc:
                if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < self.max_retries:
                    retry_count += 1
                    delay = min(
                        self.retry_max_backoff_seconds,
                        self.retry_backoff_seconds * (2 ** (retry_count - 1)),
                    )
                    delay *= random.uniform(0.5, 1.5)
                    remaining = deadline - monotonic()
                    if delay > 0 and remaining > 0:
                        sleep(min(delay, remaining))
                    continue
                self._record_metrics(
                    stage=stage,
                    started=started,
                    success=False,
                    retry_count=retry_count,
                    error_type=f"http_{exc.code}",
                )
                raise ModelProviderError(f"模型服务返回 HTTP {exc.code}。") from None
            except TimeoutError as exc:
                if attempt < self.max_retries and monotonic() < deadline:
                    retry_count += 1
                    delay = min(
                        self.retry_max_backoff_seconds,
                        self.retry_backoff_seconds * (2 ** (retry_count - 1)),
                    )
                    delay *= random.uniform(0.5, 1.5)
                    remaining = deadline - monotonic()
                    if delay > 0 and remaining > 0:
                        sleep(min(delay, remaining))
                    continue
                self._record_metrics(
                    stage=stage,
                    started=started,
                    success=False,
                    retry_count=retry_count,
                    error_type="timeout",
                )
                raise ModelProviderError("模型服务请求超时。") from exc
            except (RemoteDisconnected, ConnectionError, URLError) as exc:
                if attempt < self.max_retries and monotonic() < deadline:
                    retry_count += 1
                    delay = min(
                        self.retry_max_backoff_seconds,
                        self.retry_backoff_seconds * (2 ** (retry_count - 1)),
                    )
                    delay *= random.uniform(0.5, 1.5)
                    remaining = deadline - monotonic()
                    if delay > 0 and remaining > 0:
                        sleep(min(delay, remaining))
                    continue
                self._record_metrics(
                    stage=stage,
                    started=started,
                    success=False,
                    retry_count=retry_count,
                    error_type="connection",
                )
                raise ModelProviderError("无法连接到模型服务。") from exc
        else:  # pragma: no cover - loop always returns or raises
            self._record_metrics(
                stage=stage,
                started=started,
                success=False,
                retry_count=retry_count,
                error_type="retry_exhausted",
            )
            raise ModelProviderError("模型服务请求失败。") from None

        if not self.stream:
            try:
                response_payload = json.loads(raw)
                message = response_payload["choices"][0]["message"]
                content = message.get("content")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                self._record_metrics(
                    stage=stage,
                    started=started,
                    success=False,
                    retry_count=retry_count,
                    error_type="invalid_json",
                )
                raise ModelProviderError("模型服务返回了无法识别的响应格式。") from None

        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            self._record_metrics(
                stage=stage,
                started=started,
                success=False,
                usage=response_payload.get("usage"),
                retry_count=retry_count,
                error_type="empty_content",
            )
            raise ModelProviderError("模型响应中缺少可用的最终文本。")
        self._record_metrics(
            stage=stage,
            started=started,
            success=True,
            usage=response_payload.get("usage"),
            retry_count=retry_count,
        )
        return content.strip()

    @staticmethod
    def _read_stream_response(
        response: Any, *, deadline: float | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Collect SSE text deltas without retaining reasoning content."""

        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        content_bytes = 0
        try:
            lines = iter(response)
        except TypeError as exc:
            raise ModelProviderError("模型服务流式响应格式无效。") from exc
        for raw_line in lines:
            if deadline is not None and monotonic() >= deadline:
                raise ModelProviderError("模型服务流式响应超时。")
            if isinstance(raw_line, str):
                line = raw_line
            else:
                line = raw_line.decode("utf-8", errors="replace")
            if len(line.encode("utf-8")) > MAX_PROVIDER_RESPONSE_BYTES:
                raise ModelProviderError("模型服务响应单行超过大小限制。")
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            raw_usage = chunk.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
            choices = chunk.get("choices", [])
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta", choice.get("message", {}))
                if not isinstance(delta, dict):
                    continue
                value = delta.get("content")
                if isinstance(value, str):
                    content_bytes += len(value.encode("utf-8"))
                    if content_bytes > MAX_PROVIDER_RESPONSE_BYTES:
                        raise ModelProviderError("模型最终文本超过大小限制。")
                    content_parts.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                            continue
                        text = item["text"]
                        content_bytes += len(text.encode("utf-8"))
                        if content_bytes > MAX_PROVIDER_RESPONSE_BYTES:
                            raise ModelProviderError("模型最终文本超过大小限制。")
                        content_parts.append(text)
        return "".join(content_parts), usage
