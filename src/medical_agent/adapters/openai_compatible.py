"""Medical workflow adapter for any OpenAI-compatible chat provider."""

from __future__ import annotations

import json
import re
from threading import local
from typing import Any, cast

from ..contracts import FactPayload, PlanPayload, QueryPayload, SynthesisPayload
from ..infrastructure.openai_client import ModelProviderError, OpenAIChatClient
from ..prompting import (
    ChatPrompt,
    JsonPrompt,
    build_claim_batch_judge_prompt,
    build_fact_extraction_prompt,
    build_plan_prompt,
    build_query_prompt,
    build_synthesis_prompt,
    render_json_prompt,
)


def _extract_json_object(content: str) -> dict[str, Any]:
    source = content.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", source, flags=re.DOTALL | re.I
    )
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


class OpenAICompatibleModelAdapter:
    """Translate the domain model contract into prompt and transport calls."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str,
        timeout_seconds: int = 90,
    ) -> None:
        self._client = OpenAIChatClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            timeout_seconds=timeout_seconds,
        )
        self._thread_state = local()

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "mode": "real",
            "provider": self._client.provider,
            "name": self._client.model,
        }

    def _complete(self, prompt: ChatPrompt) -> str:
        return self._client.complete(
            system=prompt.system,
            user=prompt.user,
            max_tokens=prompt.max_tokens,
            stage=str(getattr(self._thread_state, "stage", "unknown")),
        )

    def _complete_json(self, prompt: JsonPrompt) -> dict[str, Any]:
        return _extract_json_object(self._complete(render_json_prompt(prompt)))

    def _call_json(self, stage: str, prompt: JsonPrompt) -> dict[str, Any]:
        self._thread_state.stage = stage
        try:
            return self._complete_json(prompt)
        finally:
            self._thread_state.stage = "unknown"

    def drain_call_metrics(self) -> list[dict[str, Any]]:
        return self._client.drain_call_metrics()

    def plan(self, request: str, patient_record: str) -> PlanPayload:
        return cast(
            PlanPayload,
            self._call_json("plan", build_plan_prompt(request, patient_record)),
        )

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> QueryPayload:
        return cast(
            QueryPayload,
            self._call_json(
                "query",
                build_query_prompt(
                    task=task,
                    request=request,
                    patient_record=patient_record,
                    upstream=upstream,
                ),
            ),
        )

    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> FactPayload:
        return cast(
            FactPayload,
            self._call_json(
                "extract", build_fact_extraction_prompt(task=task, evidence=evidence)
            ),
        )

    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
    ) -> SynthesisPayload:
        return cast(
            SynthesisPayload,
            self._call_json(
                "synthesize",
                build_synthesis_prompt(task=task, request=request, facts=facts),
            ),
        )

    def judge_claims(self, items: list[dict[str, Any]]) -> dict[str, str]:
        """Evaluate claims in bounded batches instead of one provider call each."""

        verdicts: dict[str, str] = {}
        for offset in range(0, len(items), 8):
            batch = items[offset : offset + 8]
            payload = self._call_json("judge", build_claim_batch_judge_prompt(batch))
            raw_verdicts = payload.get("verdicts", [])
            if isinstance(raw_verdicts, list):
                for item in raw_verdicts:
                    if not isinstance(item, dict):
                        continue
                    claim_id = str(item.get("id", ""))
                    verdict = str(item.get("verdict", "")).upper()
                    if verdict in {
                        "SUPPORTED",
                        "PARTIALLY_SUPPORTED",
                        "CONTRADICTED",
                        "INSUFFICIENT",
                        "NOT_SUPPORTED",
                        "UNCERTAIN",
                    }:
                        verdicts[claim_id] = verdict
            for item in batch:
                claim_id = str(item.get("id", ""))
                verdicts.setdefault(claim_id, "UNCERTAIN")
        return verdicts
