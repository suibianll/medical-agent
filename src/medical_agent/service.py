"""Application service that coordinates planning, execution, evaluation and repair."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Callable

from .adapters.model_factory import create_model_runtime
from .agent_pipeline import ThreeStageTaskAgent
from .audit_log import AuditEventSink, SafeAuditLogger
from .dag_scheduler import execute_dag
from .demo_model import DemoModelAdapter
from .evaluator import evaluate_claims
from .evidence import EvidenceRegistry
from .graph import build_evidence_graph
from .model_adapter import ModelAdapter
from .observability.progress import audit_text, make_progress_emitter
from .plan_validator import validate_plan
from .prompts.conversation import build_contextual_request
from .repair import build_repair_plan
from .report import render_cited_claim, render_report
from .retrieval import JsonKnowledgeBase, PatientRecordRetriever
from .run_archive import InMemoryRunArchive
from .infrastructure.model_config import load_model_configuration


ProgressCallback = Callable[[dict[str, Any]], None]


class MedicalAgentService:
    """Local MVP service with a replaceable model and knowledge-base adapter."""

    def __init__(
        self,
        *,
        model_profiles: dict[str, ModelAdapter] | None = None,
        model_profile_labels: dict[str, str] | None = None,
        default_model_profile: str | None = None,
        knowledge_base: JsonKnowledgeBase | None = None,
        max_repair_rounds: int = 2,
        max_workers: int = 3,
        run_archive: InMemoryRunArchive | None = None,
        archive_max_runs: int = 24,
        archive_ttl_seconds: int = 30 * 60,
        audit_logger: AuditEventSink | None = None,
    ) -> None:
        profiles = dict(model_profiles or {})
        if not profiles:
            profiles = {"demo": DemoModelAdapter()}
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
        self.knowledge_base = knowledge_base or JsonKnowledgeBase.demo()
        self.max_repair_rounds = max_repair_rounds
        self.max_workers = max_workers
        # Complete results remain only in this process for a bounded TTL so a
        # local evidence view can be reopened.  Audit events are redacted by
        # the archive and no archive data is written to logs or disk.
        self.run_archive = run_archive or InMemoryRunArchive(
            max_runs=archive_max_runs,
            ttl_seconds=archive_ttl_seconds,
        )
        self.audit_logger = audit_logger or SafeAuditLogger()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a short-lived in-memory full result, if still retained."""

        return self.run_archive.get_result(run_id)

    def get_run_events(self, run_id: str) -> dict[str, Any] | None:
        """Return a public redacted audit timeline for one retained run.

        The archive's internal counter map keeps its natural ``claims`` key,
        while the HTTP-facing view renames it to ``claim_count`` so consumers
        never confuse an execution count with stored claim content.
        """

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

    def _archive_result(self, result: dict[str, Any]) -> None:
        """Archival is best-effort and must never affect a medical response."""

        try:
            self.run_archive.finalize(result)
        except Exception:  # noqa: BLE001 - temporary navigation is optional
            return

    def _record_audit_event(self, event: dict[str, Any]) -> None:
        """Fan out a safe event to the redacted archive and runtime logger."""

        try:
            self.run_archive.append_event(event)
        except Exception:  # noqa: BLE001 - auditing cannot interrupt a run
            pass
        try:
            self.audit_logger.record(event)
        except Exception:  # noqa: BLE001 - handlers are optional/best effort
            pass

    def model_metadata(self) -> dict[str, str]:
        """Return safe model identity for clients without exposing secrets."""

        return self._model_metadata_for(
            self.model_profiles[self.default_model_profile]
        )

    def _model_metadata_for(self, model: ModelAdapter) -> dict[str, str]:
        """Normalize one adapter's safe public runtime identity."""

        try:
            raw = model.runtime_metadata()
        except Exception:  # noqa: BLE001 - health checks must stay available
            raw = {}
        if not isinstance(raw, dict):
            raw = {}

        mode = str(raw.get("mode", "demo"))
        # The public API intentionally has only these two states.  Unknown
        # injected adapters fall back to demo rather than claiming a provider
        # is configured when it is not.
        if mode not in {"real", "demo"}:
            mode = "demo"
        provider = audit_text(raw.get("provider", "local-demo"), 80)
        name = audit_text(raw.get("name", "demo"), 120)
        return {"mode": mode, "provider": provider, "name": name}

    def model_catalog(self) -> dict[str, Any]:
        """Return selectable model identities without URLs or credentials."""

        profiles: list[dict[str, str]] = []
        for profile_id, model in self.model_profiles.items():
            try:
                raw = model.runtime_metadata()
            except Exception:  # noqa: BLE001 - catalog must remain available
                raw = {}
            mode = str(raw.get("mode", "demo")) if isinstance(raw, dict) else "demo"
            if mode not in {"real", "demo"}:
                mode = "demo"
            profiles.append(
                {
                    "id": profile_id,
                    "label": self.model_profile_labels.get(profile_id, profile_id),
                    "mode": mode,
                    "provider": audit_text(
                        raw.get("provider", "local-demo") if isinstance(raw, dict) else "local-demo",
                        80,
                    ),
                    "name": audit_text(
                        raw.get("name", "demo") if isinstance(raw, dict) else "demo",
                        120,
                    ),
                }
            )
        return {"default": self.default_model_profile, "profiles": profiles}

    def _select_model(self, profile_id: Any = None) -> tuple[str, ModelAdapter]:
        selected = str(profile_id or self.default_model_profile).strip()
        if selected not in self.model_profiles:
            raise ValueError("所选模型配置不存在或未完整配置。")
        return selected, self.model_profiles[selected]

    @classmethod
    def from_environment(cls) -> "MedicalAgentService":
        """Build adapters from external configuration through the bootstrap layer."""

        runtime = create_model_runtime(load_model_configuration())
        return cls(
            model_profiles=runtime.profiles,
            model_profile_labels=runtime.labels,
            default_model_profile=runtime.default_profile,
            # A little lower than local demo concurrency to be friendlier to
            # provider quotas while retaining parallel DAG behavior.
            max_workers=2 if runtime.default_profile != "demo" else 3,
        )

    @staticmethod
    def _collect_claims(task_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        counter = 0
        for task_id, state in sorted(task_states.items()):
            if state.get("status") != "completed":
                continue
            for raw_claim in (state.get("result") or {}).get("claims", []):
                counter += 1
                claim = deepcopy(raw_claim)
                claim["id"] = f"C{counter}"
                claim["task_id"] = task_id
                claims.append(claim)
        return claims

    @staticmethod
    def _annotate_claim_status(
        claims: list[dict[str, Any]], evaluation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        issues_by_claim: dict[str, list[str]] = {}
        for issue in evaluation.get("issues", []):
            issues_by_claim.setdefault(issue["claim"], []).append(issue["code"])
        annotated = []
        for claim in claims:
            clone = deepcopy(claim)
            clone["status"] = "supported" if clone["id"] not in issues_by_claim else "needs_repair"
            clone["issues"] = issues_by_claim.get(clone["id"], [])
            # Reuses the report renderer so API claims and conversational
            # answers display the original refs after every sentence.
            clone["cited_text"] = render_cited_claim(clone)
            annotated.append(clone)
        return annotated

    def _execute(
        self,
        *,
        tasks: list[dict[str, Any]],
        agent: ThreeStageTaskAgent,
        prior_states: dict[int, dict[str, Any]] | None = None,
        rerun_task_ids: set[int] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def worker(task: dict[str, Any], upstream: dict[int, Any]) -> dict[str, Any]:
            return agent.run(task, upstream)

        task_by_id = {task["id"]: task for task in tasks}

        def on_status(task_id: int, status: str) -> None:
            if on_progress is None:
                return
            task = task_by_id.get(task_id)
            if status == "running":
                on_progress(
                    {
                        "stage": "task_started",
                        "message": "子任务已开始执行",
                        "task_id": task_id,
                        "task": task,
                        "status": status,
                    }
                )
            elif status == "completed":
                on_progress(
                    {
                        "stage": "task_completed",
                        "message": "子任务已完成",
                        "task_id": task_id,
                        "task": task,
                        "status": status,
                    }
                )
            elif status in {"failed", "blocked"}:
                on_progress(
                    {
                        "stage": "error",
                        "message": (
                            "子任务执行失败"
                            if status == "failed"
                            else "子任务因依赖未完成而被阻塞"
                        ),
                        "task_id": task_id,
                        "task": task,
                        "status": status,
                    }
                )

        return execute_dag(
            tasks,
            worker,
            max_workers=self.max_workers,
            initial_task_states=prior_states,
            rerun_task_ids=rerun_task_ids,
            on_status=on_status,
        )

    @staticmethod
    def _chat_answer(claims: list[dict[str, Any]], status: str) -> str:
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
    ) -> dict[str, Any]:
        """Answer a conversational turn with the same evidence guarantees as a run."""

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
        # Archive once, after chat-specific fields are complete, so a refreshed
        # evidence page receives exactly the payload returned to the caller.
        self._archive_result(result)
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
    ) -> dict[str, Any]:
        """Run a full plan-execute-evaluate cycle with explicit arguments."""

        request = str(request or "").strip()
        patient_record = str(patient_record or "").strip()
        selected_profile, selected_model = self._select_model(model_profile)
        selected_model_metadata = {
            "profile": selected_profile,
            **self._model_metadata_for(selected_model),
        }
        run_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            self.run_archive.start(run_id, created_at)
        except Exception:  # noqa: BLE001 - archive navigation is optional
            pass
        emit_progress = make_progress_emitter(
            on_progress,
            run_id,
            audit_callback=self._record_audit_event,
        )

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            if archive_result:
                self._archive_result(result)
            return result

        if not request:
            emit_progress(
                {
                    "stage": "error",
                    "message": "缺少用户问题，无法开始任务规划",
                    "status": "rejected",
                }
            )
            return finish({
                "status": "rejected",
                "run": {"id": run_id, "created_at": created_at, "model": selected_model_metadata},
                "errors": [{"code": "REQUEST_REQUIRED", "message": "请提供用户请求。"}],
            })
        if not patient_record and not allow_general:
            emit_progress(
                {
                    "stage": "error",
                    "message": "缺少患者病历，无法执行个体化分析",
                    "status": "rejected",
                }
            )
            return finish({
                "status": "rejected",
                "run": {"id": run_id, "created_at": created_at, "model": selected_model_metadata},
                "errors": [{"code": "PATIENT_RECORD_REQUIRED", "message": "请提供患者病历。"}],
            })

        original_request = request
        model_request = build_contextual_request(request, conversation_history)
        emit_progress(
            {
                "stage": "planning",
                "message": "正在生成任务计划",
                "status": "started",
            }
        )
        try:
            raw_plan = plan if plan is not None else selected_model.plan(model_request, patient_record)
        except Exception:
            emit_progress(
                {
                    "stage": "error",
                    "message": "任务规划调用失败",
                    "status": "failed",
                }
            )
            self.run_archive.mark_failed(run_id)
            raise
        validation = validate_plan(raw_plan)
        if not validation["valid"]:
            emit_progress(
                {
                    "stage": "error",
                    "message": "任务计划未通过结构校验",
                    "status": "plan_rejected",
                }
            )
            return finish({
                "status": "plan_rejected",
                "run": {
                    "id": run_id,
                    "created_at": created_at,
                    "model": selected_model_metadata,
                    "plan": raw_plan,
                },
                "errors": validation["errors"],
            })

        tasks = validation["tasks"]
        emit_progress(
            {
                "stage": "planning",
                "message": "任务计划已生成",
                "status": "completed",
                "tasks": tasks,
            }
        )
        registry = EvidenceRegistry()
        agent = ThreeStageTaskAgent(
            model=selected_model,
            patient_retriever=PatientRecordRetriever(patient_record),
            knowledge_base=self.knowledge_base,
            registry=registry,
            patient_record=patient_record,
            request=model_request,
            patient_grounding_required=bool(patient_record),
            on_progress=emit_progress,
        )

        execution = self._execute(tasks=tasks, agent=agent, on_progress=emit_progress)
        task_states = execution["tasks"]
        repair_history: list[dict[str, Any]] = []
        semantic_cache: dict[tuple[str, tuple[str, ...]], str] = {}

        for repair_round in range(self.max_repair_rounds + 1):
            claims = self._collect_claims(task_states)
            execution_issues = []
            for task_id, state in sorted(task_states.items()):
                if state.get("status") == "failed":
                    execution_issues.append(
                        {
                            "claim": f"T{task_id}",
                            "task_id": task_id,
                            "code": "TASK_EXECUTION_FAILED",
                        }
                    )
                elif state.get("status") == "blocked":
                    execution_issues.append(
                        {
                            "claim": f"T{task_id}",
                            "task_id": task_id,
                            "code": "TASK_BLOCKED",
                        }
                    )
            emit_progress(
                {
                    "stage": "evaluate",
                    "message": "正在核验结论与证据链",
                    "status": "started",
                    "round": repair_round,
                }
            )
            evaluation = (
                {"pass": False, "issues": execution_issues, "judgements": []}
                if execution_issues
                else evaluate_claims(
                    claims,
                    registry.as_map(),
                    selected_model,
                    semantic_cache=semantic_cache,
                )
            )
            issue_codes = [
                issue.get("code", "")
                for issue in evaluation.get("issues", [])
                if isinstance(issue, dict) and isinstance(issue.get("code"), str)
            ]
            emit_progress(
                {
                    "stage": "evaluate",
                    "message": "证据链核验完成",
                    "status": "passed" if evaluation["pass"] else "needs_repair",
                    "round": repair_round,
                    "evaluation": {
                        "pass": evaluation["pass"],
                        "issue_count": len(evaluation.get("issues", [])),
                        "judgement_count": len(evaluation.get("judgements", [])),
                        "issue_codes": issue_codes,
                    },
                }
            )
            if evaluation["pass"]:
                status = "passed"
                break

            if repair_round >= self.max_repair_rounds:
                status = "needs_human_review"
                break

            repair = build_repair_plan(
                issues=evaluation["issues"], claims=claims, tasks=tasks
            )
            repair["round"] = repair_round + 1
            repair_history.append(repair)
            if not repair["repairable"]:
                emit_progress(
                    {
                        "stage": "repair",
                        "message": "当前问题无法通过自动重试修复",
                        "status": "not_repairable",
                        "round": repair["round"],
                    }
                )
                status = "needs_human_review"
                break

            repair_tasks = [
                task
                for task in tasks
                if task["id"] in set(repair.get("rerun_tasks", []))
            ]
            emit_progress(
                {
                    "stage": "repair",
                    "message": "证据不足，正在重试相关子任务",
                    "status": "scheduled",
                    "round": repair["round"],
                    "tasks": repair_tasks,
                }
            )

            agent.set_repair_directives(repair.get("task_directives", []))
            execution = self._execute(
                tasks=tasks,
                agent=agent,
                prior_states=task_states,
                rerun_task_ids=set(repair["rerun_tasks"]),
                on_progress=emit_progress,
            )
            task_states = execution["tasks"]
        else:  # defensive; the loop always breaks above
            status = "needs_human_review"

        claims = self._annotate_claim_status(claims, evaluation)
        evidence = registry.all()
        graph = build_evidence_graph(
            tasks=tasks,
            task_states=task_states,
            claims=claims,
            evidence=evidence,
        )
        report = render_report(
            request=original_request,
            task_states=task_states,
            claims=claims,
            evidence=evidence,
            evaluation=evaluation,
            status=status,
            template=report_template,
        )

        task_list = [
            {
                "id": task_id,
                "goal": state["task"]["goal"],
                "deps": state["task"]["deps"],
                "status": state["status"],
                "error": state.get("error"),
            }
            for task_id, state in sorted(task_states.items())
        ]
        emit_progress(
            {
                "stage": "completed",
                "message": "本轮任务执行完成",
                "status": status,
                "counts": {
                    "tasks": len(task_list),
                    "claims": len(claims),
                    "evidence": len(evidence),
                },
            }
        )
        return finish({
            "status": status,
            "run": {
                "id": run_id,
                "created_at": created_at,
                "model": selected_model_metadata,
                "plan": {"tasks": tasks},
                "tasks": task_list,
                "waves": execution["waves"],
                "repair_history": repair_history,
                "evaluation": evaluation,
            },
            "report": report,
            "graph": graph,
            "evidence": evidence,
            "claims": claims,
        })
