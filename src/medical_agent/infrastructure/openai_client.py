"""Minimal synchronous client for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("缺少模型 API Key。")
        if not model or not model.strip():
            raise ValueError("缺少模型名称。")
        self._api_key = api_key.strip()
        if not provider or not provider.strip():
            raise ValueError("缺少模型供应商名称。")
        self.provider = provider.strip()
        self.base_url = normalize_base_url(base_url, self.provider)
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
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
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise ModelProviderError(f"模型服务返回 HTTP {exc.code}。") from None
        except URLError as exc:
            raise ModelProviderError("无法连接到模型服务。") from exc
        except TimeoutError as exc:
            raise ModelProviderError("模型服务请求超时。") from exc

        try:
            response_payload: Any = json.loads(raw)
            message = response_payload["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise ModelProviderError("模型服务返回了无法识别的响应格式。") from None

        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ModelProviderError("模型响应中缺少可用的最终文本。")
        return content.strip()
