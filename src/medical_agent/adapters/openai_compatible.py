"""Medical workflow adapter for any OpenAI-compatible chat provider."""

from __future__ import annotations

from typing import Any, cast

from ..contracts import FactPayload, PlanPayload, QueryPayload, SynthesisPayload
from ..infrastructure.openai_client import ModelProviderError, OpenAIChatClient
from ..model_adapter import ModelAdapter
from ..prompts.common import render_json_prompt
from ..prompts.evaluation import build_claim_batch_judge_prompt
from ..prompts.extraction import build_fact_extraction_prompt
from ..prompts.planning import build_plan_prompt
from ..prompts.retrieval import build_query_prompt
from ..prompts.synthesis import build_synthesis_prompt
from ..prompts.types import ChatPrompt, JsonPrompt
from ..utils.json_tools import extract_json_object


class OpenAICompatibleModelAdapter(ModelAdapter):
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
        # Preserve public attributes used by existing integrations.
        self.provider = self._client.provider
        self.base_url = self._client.base_url
        self.model = self._client.model
        self.timeout_seconds = self._client.timeout_seconds

    def runtime_metadata(self) -> dict[str, str]:
        return {"mode": "real", "provider": self.provider, "name": self.model}

    def _chat(self, *, system: str, user: str, max_tokens: int = 1200) -> str:
        """Compatibility hook retained for tests and custom subclasses."""

        return self._client.complete(system=system, user=user, max_tokens=max_tokens)

    def _complete(self, prompt: ChatPrompt) -> str:
        return self._chat(
            system=prompt.system, user=prompt.user, max_tokens=prompt.max_tokens
        )

    def _complete_json(self, prompt: JsonPrompt) -> dict[str, Any]:
        return extract_json_object(
            self._complete(render_json_prompt(prompt)), error_type=ModelProviderError
        )

    def plan(self, request: str, patient_record: str) -> PlanPayload:
        return cast(PlanPayload, self._complete_json(build_plan_prompt(request, patient_record)))

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> QueryPayload:
        return cast(QueryPayload, self._complete_json(
            build_query_prompt(
                task=task,
                request=request,
                patient_record=patient_record,
                upstream=upstream,
            )
        ))

    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> FactPayload:
        return cast(FactPayload, self._complete_json(
            build_fact_extraction_prompt(task=task, evidence=evidence)
        ))

    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
    ) -> SynthesisPayload:
        return cast(SynthesisPayload, self._complete_json(
            build_synthesis_prompt(
                task=task, request=request, facts=facts
            )
        ))

    def judge_claims(self, items: list[dict[str, Any]]) -> dict[str, str]:
        """Evaluate claims in bounded batches instead of one provider call each."""

        verdicts: dict[str, str] = {}
        for offset in range(0, len(items), 8):
            batch = items[offset : offset + 8]
            payload = self._complete_json(build_claim_batch_judge_prompt(batch))
            raw_verdicts = payload.get("verdicts", [])
            if isinstance(raw_verdicts, list):
                for item in raw_verdicts:
                    if not isinstance(item, dict):
                        continue
                    claim_id = str(item.get("id", ""))
                    verdict = str(item.get("verdict", "")).upper()
                    if verdict in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
                        verdicts[claim_id] = verdict
            for item in batch:
                claim_id = str(item.get("id", ""))
                verdicts.setdefault(claim_id, "UNCERTAIN")
        return verdicts
