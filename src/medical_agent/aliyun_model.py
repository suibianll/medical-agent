"""OpenAI-compatible model adapter used by Model Studio and OpenRouter.

No API key is stored in this module.  Supply it at runtime through
``MEDICAL_AGENT_API_KEY`` (or ``DASHSCOPE_API_KEY``) only.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model_adapter import ModelAdapter


class ModelProviderError(RuntimeError):
    """Sanitized provider error that deliberately never includes credentials."""


def normalize_base_url(
    base_url: str, provider: str = "aliyun-model-studio"
) -> str:
    """Normalize a provider base URL without rewriting an existing v1 path."""

    value = base_url.strip().rstrip("/")
    if not value.startswith(("https://", "http://")):
        raise ValueError("模型服务 URL 必须以 http:// 或 https:// 开头。")
    if value.endswith(("/compatible-mode/v1", "/v1")):
        return value
    if provider == "aliyun-model-studio":
        value = f"{value}/compatible-mode/v1"
    else:
        value = f"{value}/v1"
    return value


def _compact(value: Any, limit: int = 10_000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}\n[已截断]"


def _extract_json(content: str) -> dict[str, Any]:
    """Parse a JSON object from a compliant or lightly fenced chat response."""

    source = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", source, flags=re.DOTALL | re.I)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(source)
    object_start = source.find("{")
    if object_start >= 0:
        candidates.append(source[object_start:])

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ModelProviderError("模型未返回有效的 JSON 对象。")


class AliyunCompatibleModelAdapter(ModelAdapter):
    """Synchronous OpenAI-compatible adapter for Model Studio chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "aliyun-model-studio",
        timeout_seconds: int = 90,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("缺少模型 API Key。")
        if not model or not model.strip():
            raise ValueError("缺少模型名称。")
        self._api_key = api_key.strip()
        self.provider = provider.strip() or "openai-compatible"
        self.base_url = normalize_base_url(base_url, self.provider)
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    def runtime_metadata(self) -> dict[str, str]:
        """Expose only the selected provider and model, never credentials."""

        return {
            "mode": "real",
            "provider": self.provider,
            "name": self.model,
        }

    def _chat(self, *, system: str, user: str, max_tokens: int = 1200) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - user-supplied configured endpoint
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            # Do not include response body: it can contain provider diagnostics
            # and would make accidental logging of request context easier.
            raise ModelProviderError(f"模型服务返回 HTTP {exc.code}。") from None
        except URLError as exc:
            raise ModelProviderError("无法连接到模型服务。") from exc
        except TimeoutError as exc:
            raise ModelProviderError("模型服务请求超时。") from exc

        try:
            payload = json.loads(raw)
            message = payload["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise ModelProviderError("模型服务返回了无法识别的响应格式。") from None

        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            # Reasoning content intentionally is not treated as a final answer.
            raise ModelProviderError("模型响应中缺少可用的最终文本。")
        return content.strip()

    def _chat_json(self, *, task: str, payload: dict[str, Any], max_tokens: int = 1200) -> dict[str, Any]:
        system = (
            "你是医疗信息系统中的受限组件。只处理给定数据，不输出诊断、处方、剂量或完整思维链。"
            "必须严格按用户给定的 JSON 模板返回一个 JSON 对象，禁止 Markdown、解释文字和额外字段。"
            "引用只能使用输入中已有的证据 ID。"
        )
        user = f"任务：{task}\n\n输入：\n{json.dumps(payload, ensure_ascii=False)}"
        return _extract_json(self._chat(system=system, user=user, max_tokens=max_tokens))

    def plan(self, request: str, patient_record: str) -> dict[str, Any]:
        patient_instruction = (
            "已提供患者病历：可以设置患者事实提取任务。"
            if patient_record.strip()
            else "未提供患者病历：只规划一般医学知识检索与回答任务，不要虚构患者事实。"
        )
        return self._chat_json(
            task=(
                "将请求拆为 2 到 6 个可执行任务。返回模板："
                '{"tasks":[{"id":1,"goal":"一句话任务目标","deps":[]}]}。'
                "id 必须从 1 连续递增；deps 只能包含小于当前 id 的整数；"
                f"保留必要依赖，不要生成报告任务。{patient_instruction}"
            ),
            payload={"request": _compact(request), "patient_record": _compact(patient_record)},
            max_tokens=900,
        )

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> dict[str, Any]:
        upstream_claims = {
            str(task_id): [
                {"text": _compact(claim.get("text"), 300), "refs": claim.get("refs", [])}
                for claim in (result or {}).get("claims", [])[:3]
            ]
            for task_id, result in upstream.items()
        }
        return self._chat_json(
            task='为当前子任务生成 1 到 3 条简短检索查询。返回模板：{"queries":["查询"]}。',
            payload={
                "task": task,
                "request": _compact(request, 2000),
                "patient_record": _compact(patient_record, 4000),
                "upstream_claims": upstream_claims,
            },
            max_tokens=500,
        )

    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> dict[str, Any]:
        return self._chat_json(
            task=(
                "仅从输入证据中提取最多 6 条关键事实。返回模板："
                '{"facts":[{"text":"原子事实","ref":"输入中的单个证据ID"}]}。'
                "不得推断、合并多个来源或编造引用。"
            ),
            payload={"task": task, "evidence": evidence[:12]},
            max_tokens=900,
        )

    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
        evidence: list[dict[str, str]],
        upstream: dict[int, Any],
    ) -> dict[str, Any]:
        patient_grounding_required = bool(task.get("patient_grounding_required", True))
        citation_policy = (
            "涉及个体患者分析、风险或建议时，同时引用一个 P# 患者事实和一个 K# 知识库证据。"
            if patient_grounding_required
            else "当前没有患者病历；每条结论至少引用一个 K# 知识库证据，且不得声称适用于某个具体患者。"
        )
        return self._chat_json(
            task=(
                "根据事实回答当前子任务。返回模板："
                '{"claims":[{"text":"一条原子、审慎的结论","refs":["P1","K1"]}],"unknowns":["缺失信息"]}。'
                f"每条可验证结论必须给出至少一个真实 refs；{citation_policy}"
                "证据不足时写入 unknowns。"
            ),
            payload={
                "task": task,
                "request": _compact(request, 2000),
                "facts": facts[:8],
                "evidence": evidence[:12],
            },
            max_tokens=1000,
        )

    def judge_claim(self, *, claim: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        response = self._chat(
            system=(
                "你是引用核验器。不要解释，也不要输出思维过程。"
                "仅输出一个大写词：SUPPORTED、NOT_SUPPORTED 或 UNCERTAIN。"
            ),
            user=(
                "判断给定证据是否直接支持结论。\n"
                f"结论：{_compact(claim.get('text'), 1200)}\n"
                f"证据：{json.dumps(evidence, ensure_ascii=False)}"
            ),
            max_tokens=20,
        ).upper()
        match = re.fullmatch(
            r"\s*(SUPPORTED|NOT_SUPPORTED|UNCERTAIN)\s*[.!。]?\s*", response
        )
        if match:
            return match.group(1)
        return "UNCERTAIN"


# The original public name remains available for compatibility.  New code can
# use the provider-neutral alias without forcing downstream imports to change.
OpenAICompatibleModelAdapter = AliyunCompatibleModelAdapter
