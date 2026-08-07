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
        max_output_tokens: int | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        stream: bool = False,
        thinking_stages: tuple[str, ...] | None = None,
    ) -> None:
        self._client = OpenAIChatClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            timeout_seconds=timeout_seconds,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            stream=stream,
        )
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("模型输出 token 预算必须大于 0。")
        self._max_output_tokens = max_output_tokens
        self._thinking_stages = (
            frozenset(thinking_stages) if thinking_stages is not None else None
        )
        self._thread_state = local()

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "mode": "real",
            "provider": self._client.provider,
            "name": self._client.model,
        }

    def _complete(self, prompt: ChatPrompt) -> str:
        max_tokens = prompt.max_tokens
        if self._max_output_tokens is not None:
            max_tokens = max(max_tokens, self._max_output_tokens)
        stage = str(getattr(self._thread_state, "stage", "unknown"))
        thinking_override: bool | None = None
        thinking_budget_override: int | None = None
        if self._thinking_stages is not None:
            thinking_override = stage in self._thinking_stages
            thinking_budget_override = (
                self._client.thinking_budget if thinking_override else None
            )
        return self._client.complete(
            system=prompt.system,
            user=prompt.user,
            max_tokens=max_tokens,
            stage=stage,
            enable_thinking_override=thinking_override,
            thinking_budget_override=thinking_budget_override,
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

    def judge_claims(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate claims in bounded batches instead of one provider call each."""

        verdicts: dict[str, Any] = {}
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
                        raw_edges = item.get(
                            "evidence_verdicts", item.get("edges", item.get("evidence"))
                        )
                        edge_values: list[dict[str, str]] = []
                        if isinstance(raw_edges, dict):
                            edge_items = [
                                {"evidence_id": key, "verdict": value}
                                for key, value in raw_edges.items()
                            ]
                        elif isinstance(raw_edges, list):
                            edge_items = raw_edges
                        else:
                            edge_items = []
                        for edge in edge_items:
                            if not isinstance(edge, dict):
                                continue
                            evidence_id = str(
                                edge.get("evidence_id")
                                or edge.get("id")
                                or edge.get("ref")
                                or ""
                            )
                            raw_edge_verdict = edge.get("verdict")
                            if isinstance(raw_edge_verdict, dict):
                                raw_edge_verdict = raw_edge_verdict.get("verdict")
                            edge_verdict = str(raw_edge_verdict or "").upper()
                            if evidence_id and edge_verdict in {
                                "SUPPORTED",
                                "PARTIALLY_SUPPORTED",
                                "CONTRADICTED",
                                "INSUFFICIENT",
                                "NOT_SUPPORTED",
                                "UNCERTAIN",
                            }:
                                edge_values.append(
                                    {
                                        "evidence_id": evidence_id,
                                        "verdict": edge_verdict,
                                    }
                                )
                        verdicts[claim_id] = (
                            {"verdict": verdict, "evidence_verdicts": edge_values}
                            if edge_values
                            else verdict
                        )
            for item in batch:
                claim_id = str(item.get("id", ""))
                verdicts.setdefault(claim_id, "UNCERTAIN")
        return verdicts
