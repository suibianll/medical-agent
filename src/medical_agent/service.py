"""Application facade for medical-agent use cases."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from .application.workflow import MedicalWorkflow
from .contracts import Claim, ModelMetadata, ModelProfileMetadata, RunResult
from .model_adapter import ModelAdapter
from .observability.progress import audit_text
from .ports import AuditEventSink, KnowledgeBasePort, RunArchivePort
from .report import render_cited_claim


ProgressCallback = Callable[[dict[str, Any]], None]


class MedicalAgentService:
    """Stable application API with all concrete dependencies injected."""

    def __init__(
        self,
        *,
        model_profiles: Mapping[str, ModelAdapter],
        knowledge_base: KnowledgeBasePort,
        run_archive: RunArchivePort,
        audit_logger: AuditEventSink,
        model_profile_labels: Mapping[str, str] | None = None,
        default_model_profile: str | None = None,
        max_repair_rounds: int = 2,
        max_workers: int = 3,
    ) -> None:
        profiles = dict(model_profiles)
        if not profiles:
            raise ValueError("至少需要一个模型配置。")
        requested_default = str(default_model_profile or "").strip()
        if requested_default and requested_default not in profiles:
            raise ValueError("默认模型配置不存在。")
        self.default_model_profile = requested_default or next(iter(profiles))
        self.model_profiles = profiles
        self.model_profile_labels = {
            profile_id: audit_text(
                (model_profile_labels or {}).get(profile_id, profile_id), 80
            )
            for profile_id in profiles
        }
        self.knowledge_base = knowledge_base
        self.run_archive = run_archive
        self._workflow = MedicalWorkflow(
            knowledge_base=knowledge_base,
            run_archive=run_archive,
            audit_logger=audit_logger,
            max_repair_rounds=max_repair_rounds,
            max_workers=max_workers,
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.run_archive.get_result(run_id)

    def get_run_events(self, run_id: str) -> dict[str, Any] | None:
        payload = self.run_archive.get_events(run_id)
        if payload is None:
            return None
        public_payload = deepcopy(payload)
        for event in public_payload.get("events", []):
            if not isinstance(event, dict) or not isinstance(event.get("counts"), dict):
                continue
            counts = event["counts"]
            if "claims" in counts:
                event["counts"] = {
                    **{key: value for key, value in counts.items() if key != "claims"},
                    "claim_count": counts["claims"],
                }
        return public_payload

    @staticmethod
    def _model_metadata_for(model: ModelAdapter) -> ModelMetadata:
        try:
            raw = model.runtime_metadata()
        except Exception:  # noqa: BLE001 - health checks must stay available
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        mode = str(raw.get("mode", "demo"))
        if mode not in {"real", "demo"}:
            mode = "demo"
        return {
            "mode": mode,
            "provider": audit_text(raw.get("provider", "local-demo"), 80),
            "name": audit_text(raw.get("name", "demo"), 120),
        }

    def model_metadata(self) -> ModelMetadata:
        return self._model_metadata_for(self.model_profiles[self.default_model_profile])

    def model_catalog(self) -> dict[str, Any]:
        profiles: list[dict[str, str]] = []
        for profile_id, model in self.model_profiles.items():
            profiles.append(
                {
                    "id": profile_id,
                    "label": self.model_profile_labels.get(profile_id, profile_id),
                    **self._model_metadata_for(model),
                }
            )
        return {"default": self.default_model_profile, "profiles": profiles}

    def _select_model(self, profile_id: Any = None) -> tuple[str, ModelAdapter]:
        selected = str(profile_id or self.default_model_profile).strip()
        if selected not in self.model_profiles:
            raise ValueError("所选模型配置不存在或未完整配置。")
        return selected, self.model_profiles[selected]

    @staticmethod
    def _chat_answer(claims: list[Claim], status: str) -> str:
        if claims:
            return "\n".join(
                str(claim.get("cited_text") or render_cited_claim(claim))
                for claim in claims
            )
        if status == "needs_human_review":
            return "当前证据链未通过自动核验，建议转人工审核或补充资料。"
        return "未检索到足以形成可引用结论的证据，请补充问题或知识库资料。"

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
        result["answer"] = self._chat_answer(result.get("claims", []), result["status"])
        result["mode"] = "patient" if patient_record.strip() else "general"
        self._workflow.archive_result(result)
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
        selected_profile, selected_model = self._select_model(model_profile)
        metadata: ModelProfileMetadata = {
            "profile": selected_profile,
            **self._model_metadata_for(selected_model),
        }
        return self._workflow.run(
            request=request,
            patient_record=patient_record,
            selected_model=selected_model,
            selected_model_metadata=metadata,
            plan=plan,
            allow_general=allow_general,
            conversation_history=conversation_history,
            report_template=report_template,
            on_progress=on_progress,
            archive_result=archive_result,
        )
