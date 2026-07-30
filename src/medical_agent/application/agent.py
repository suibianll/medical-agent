"""Core application API for the medical agent."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..contracts import Claim, ModelMetadata, ModelProfileMetadata, RunResult
from ..observability.progress import audit_text
from ..ports import (
    AuditEventSink,
    DecisionRouter,
    KnowledgeBasePort,
    ModelAdapter,
    RerankerPort,
    RunArchivePort,
)
from ..report import render_cited_claim
from .workflow import MedicalWorkflow


ProgressCallback = Callable[[dict[str, Any]], None]


class MedicalAgent:
    """Coordinate models, knowledge and the execution workflow."""

    def __init__(
        self,
        *,
        model_profiles: Mapping[str, ModelAdapter],
        knowledge_base: KnowledgeBasePort,
        run_archive: RunArchivePort,
        audit_logger: AuditEventSink,
        verifier_model: ModelAdapter | None = None,
        model_profile_labels: Mapping[str, str] | None = None,
        default_model_profile: str | None = None,
        max_repair_rounds: int = 2,
        max_workers: int = 3,
        retrieval_limit: int = 8,
        retrieval_candidate_budget: int = 12,
        retrieval_max_per_document: int = 2,
        retrieval_max_rounds: int = 2,
        retrieval_refine_on_empty: bool = True,
        retrieval_refine_min_candidates: int = 1,
        reranker: RerankerPort | None = None,
        reranker_max_calls_per_run: int = 8,
        reranker_min_candidates: int = 2,
        reranker_cache_size: int = 128,
        reranker_cache_ttl_seconds: int = 300,
        decision_router: DecisionRouter | None = None,
    ) -> None:
        profiles = dict(model_profiles)
        if not profiles:
            raise ValueError("至少需要一个模型配置。")
        default_profile = str(default_model_profile or "").strip() or next(iter(profiles))
        if default_profile not in profiles:
            raise ValueError("默认模型配置不存在。")

        self.default_model_profile = default_profile
        self.model_profiles = profiles
        self.model_profile_labels = {
            profile_id: audit_text(
                (model_profile_labels or {}).get(profile_id, profile_id), 80
            )
            for profile_id in profiles
        }
        self.knowledge_base = knowledge_base
        self.run_archive = run_archive
        self.workflow = MedicalWorkflow(
            knowledge_base=knowledge_base,
            run_archive=run_archive,
            audit_logger=audit_logger,
            verifier_model=verifier_model,
            max_repair_rounds=max_repair_rounds,
            max_workers=max_workers,
            retrieval_limit=retrieval_limit,
            retrieval_candidate_budget=retrieval_candidate_budget,
            retrieval_max_per_document=retrieval_max_per_document,
            retrieval_max_rounds=retrieval_max_rounds,
            retrieval_refine_on_empty=retrieval_refine_on_empty,
            retrieval_refine_min_candidates=retrieval_refine_min_candidates,
            reranker=reranker,
            reranker_max_calls_per_run=reranker_max_calls_per_run,
            reranker_min_candidates=reranker_min_candidates,
            reranker_cache_size=reranker_cache_size,
            reranker_cache_ttl_seconds=reranker_cache_ttl_seconds,
            decision_router=decision_router,
        )

    @staticmethod
    def _model_metadata(model: ModelAdapter) -> ModelMetadata:
        try:
            raw = model.runtime_metadata()
        except Exception:  # noqa: BLE001 - health checks must stay available
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        mode = str(raw.get("mode", "demo"))
        return {
            "mode": mode if mode in {"real", "demo"} else "demo",
            "provider": audit_text(raw.get("provider", "local-demo"), 80),
            "name": audit_text(raw.get("name", "demo"), 120),
        }

    def model_metadata(self) -> ModelMetadata:
        return self._model_metadata(self.model_profiles[self.default_model_profile])

    @staticmethod
    def _safe_component_metadata(value: Any, allowed: set[str]) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key in allowed:
            raw = value.get(key)
            if isinstance(raw, bool):
                result[key] = "true" if raw else "false"
            elif isinstance(raw, (int, float)):
                result[key] = str(raw)
            elif isinstance(raw, str):
                result[key] = audit_text(raw, 120)
        return result

    def runtime_metadata(self) -> dict[str, Any]:
        """Return non-secret backend identities for health and diagnostics."""

        retrieval_method = getattr(self.knowledge_base, "retrieval_metadata", None)
        try:
            retrieval_raw = retrieval_method() if callable(retrieval_method) else {}
        except Exception:  # noqa: BLE001 - diagnostics must never break health
            retrieval_raw = {}
        retrieval = self._safe_component_metadata(
            retrieval_raw, {"backend", "provider", "name", "dimensions"}
        )
        if not retrieval:
            retrieval = {"backend": "lexical"}
        embedding_raw = (
            retrieval_raw.get("embedding") if isinstance(retrieval_raw, dict) else {}
        )
        embedding = self._safe_component_metadata(
            embedding_raw, {"provider", "name", "dimensions"}
        )
        if embedding:
            retrieval["embedding"] = embedding
        governance_raw = (
            retrieval_raw.get("governance") if isinstance(retrieval_raw, dict) else {}
        )
        governance = self._safe_component_metadata(
            governance_raw,
            {
                "enabled",
                "blocked_status_count",
                "min_priority",
                "require_version",
                "allow_synthetic",
                "max_age_days",
                "reject_unknown_date",
            },
        )
        if isinstance(governance_raw, dict) and isinstance(
            governance_raw.get("allowed_source_types"), list
        ):
            governance["allowed_source_types"] = [
                audit_text(value, 80)
                for value in governance_raw["allowed_source_types"][:16]
                if isinstance(value, str)
            ]
        if governance:
            retrieval["governance"] = governance

        reranker = {"enabled": False}
        reranker_object = getattr(self.workflow, "reranker", None)
        if reranker_object is not None:
            reranker_raw_method = getattr(reranker_object, "runtime_metadata", None)
            try:
                reranker_raw = (
                    reranker_raw_method() if callable(reranker_raw_method) else {}
                )
            except Exception:  # noqa: BLE001 - diagnostics are best effort
                reranker_raw = {}
            reranker = {
                "enabled": True,
                **self._safe_component_metadata(
                    reranker_raw, {"provider", "name", "mode", "max_calls_per_run", "min_candidates"}
                ),
            }

        router_object = getattr(self.workflow, "decision_router", None)
        router_raw_method = getattr(router_object, "runtime_metadata", None)
        try:
            router_raw = router_raw_method() if callable(router_raw_method) else {}
        except Exception:  # noqa: BLE001 - diagnostics are best effort
            router_raw = {}
        routing = self._safe_component_metadata(router_raw, {"mode", "provider", "name"})
        if not routing:
            routing = {"mode": type(router_object).__name__ if router_object else "evidence"}
        return {"retrieval": retrieval, "reranker": reranker, "routing": routing}

    def model_catalog(self) -> dict[str, Any]:
        profiles = [
            {
                "id": profile_id,
                "label": self.model_profile_labels[profile_id],
                **self._model_metadata(model),
            }
            for profile_id, model in self.model_profiles.items()
        ]
        return {
            "default": self.default_model_profile,
            "profiles": profiles,
            "runtime": self.runtime_metadata(),
        }

    def _select_model(self, profile_id: Any = None) -> tuple[str, ModelAdapter]:
        selected = str(profile_id or self.default_model_profile).strip()
        if selected not in self.model_profiles:
            raise ValueError("所选模型配置不存在或未完整配置。")
        return selected, self.model_profiles[selected]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.run_archive.get_result(run_id)

    def get_run_events(self, run_id: str) -> dict[str, Any] | None:
        return self.run_archive.get_events(run_id)

    def import_knowledge(self, *, name: str, content: str) -> dict[str, Any]:
        return self.knowledge_base.import_text(name=name, content=content)

    def list_knowledge(self) -> list[dict[str, Any]]:
        return self.knowledge_base.list_documents()

    def chat(
        self,
        *,
        message: str,
        patient_record: str = "",
        history: list[dict[str, Any]] | None = None,
        report_template: Any = None,
        model_profile: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> RunResult:
        result = self.run(
            request=message,
            patient_record=patient_record,
            allow_general=True,
            conversation_history=history,
            report_template=report_template,
            model_profile=model_profile,
            on_progress=on_progress,
            archive_result=False,
        )
        claims: list[Claim] = result.get("claims", [])
        decision = result.get("run", {}).get("decision", {})
        outcome = decision.get("outcome") if isinstance(decision, dict) else ""
        supported_claims = [
            claim for claim in claims if claim.get("status") == "supported"
        ]
        if (
            result.get("status") == "passed"
            and outcome == "answer"
            and supported_claims
        ):
            answer = "\n".join(
                str(claim.get("cited_text") or render_cited_claim(claim))
                for claim in supported_claims
            )
        elif outcome == "emergency_escalation":
            answer = "当前信息触发紧急升级信号，请立即联系当地急救或具备资质的医疗专业人员。"
        elif result.get("status") == "needs_human_review" or outcome == "defer":
            answer = "当前证据链未通过自动核验，建议转人工审核或补充资料。"
        elif outcome == "ask_clarification":
            answer = "现有信息不足以形成可核验结论，请补充问题、病历或知识库资料。"
        else:
            answer = "未检索到足以形成可引用结论的证据，请补充问题或知识库资料。"
        result["answer"] = answer
        result["mode"] = "patient" if patient_record.strip() else "general"
        self.workflow.archive_result(result)
        return result

    def run(
        self,
        request: str,
        patient_record: str = "",
        plan: Any | None = None,
        allow_general: bool = False,
        conversation_history: list[dict[str, Any]] | None = None,
        report_template: Any = None,
        model_profile: str | None = None,
        on_progress: ProgressCallback | None = None,
        archive_result: bool = True,
    ) -> RunResult:
        profile_id, model = self._select_model(model_profile)
        metadata: ModelProfileMetadata = {
            "profile": profile_id,
            **self._model_metadata(model),
        }
        return self.workflow.run(
            request=request,
            patient_record=patient_record,
            selected_model=model,
            selected_model_metadata=metadata,
            plan=plan,
            allow_general=allow_general,
            conversation_history=conversation_history,
            report_template=report_template,
            on_progress=on_progress,
            archive_result=archive_result,
        )
