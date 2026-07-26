"""Plan-execute-evaluate workflow for one medical-agent run."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from ..agent_pipeline import ThreeStageTaskAgent
from ..contracts import Claim, ModelProfileMetadata, RunResult
from ..dag_scheduler import execute_dag
from ..evaluator import evaluate_claims
from ..evidence import EvidenceRegistry
from ..graph import build_evidence_graph
from ..observability.progress import make_progress_emitter
from ..observability.model_metrics import drain_model_metrics, summarize_model_metrics
from ..plan_validator import validate_plan
from ..ports import (
    AuditEventSink,
    KnowledgeBasePort,
    ModelAdapter,
    RerankerPort,
    RunArchivePort,
)
from ..prompting import build_contextual_request
from ..repair import build_repair_plan
from ..report import render_cited_claim, render_report
from ..risk import route_decision
from ..retrieval.patient import PatientRecordRetriever


ProgressCallback = Callable[[dict[str, Any]], None]


class MedicalWorkflow:
    """Execute one run without knowing configuration or concrete I/O classes."""

    def __init__(
        self,
        *,
        knowledge_base: KnowledgeBasePort,
        run_archive: RunArchivePort,
        audit_logger: AuditEventSink,
        verifier_model: ModelAdapter | None = None,
        max_repair_rounds: int = 2,
        max_workers: int = 3,
        retrieval_limit: int = 8,
        retrieval_candidate_budget: int = 12,
        retrieval_max_per_document: int = 2,
        reranker: RerankerPort | None = None,
    ) -> None:
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds 不能小于 0")
        if max_workers < 1:
            raise ValueError("max_workers 必须大于 0")
        self.knowledge_base = knowledge_base
        self.run_archive = run_archive
        self.audit_logger = audit_logger
        self.verifier_model = verifier_model
        self.max_repair_rounds = max_repair_rounds
        self.max_workers = max_workers
        self.retrieval_limit = retrieval_limit
        self.retrieval_candidate_budget = retrieval_candidate_budget
        self.retrieval_max_per_document = retrieval_max_per_document
        self.reranker = reranker

    def archive_result(self, result: RunResult | dict[str, Any]) -> None:
        """Archive best-effort without affecting the medical response."""

        try:
            self.run_archive.finalize(dict(result))
        except Exception:  # noqa: BLE001 - local navigation is optional
            return

    def _record_audit_event(self, event: dict[str, Any]) -> None:
        """Fan out a safe event to the archive and configured audit sink."""

        try:
            self.run_archive.append_event(event)
        except Exception:  # noqa: BLE001 - auditing cannot interrupt a run
            pass
        try:
            self.audit_logger.record(event)
        except Exception:  # noqa: BLE001 - handlers are best effort
            pass

    @staticmethod
    def collect_claims(task_states: dict[int, dict[str, Any]]) -> list[Claim]:
        claims: list[Claim] = []
        counter = 0
        for task_id, state in sorted(task_states.items()):
            if state.get("status") != "completed":
                continue
            for raw_claim in (state.get("result") or {}).get("claims", []):
                counter += 1
                claim: Claim = deepcopy(raw_claim)
                claim["id"] = f"C{counter}"
                claim["task_id"] = task_id
                claims.append(claim)
        return claims

    @staticmethod
    def annotate_claim_status(
        claims: list[Claim], evaluation: dict[str, Any]
    ) -> list[Claim]:
        issues_by_claim: dict[str, list[str]] = {}
        for issue in evaluation.get("issues", []):
            if not isinstance(issue, dict):
                continue
            claim_id = str(issue.get("claim", ""))
            code = str(issue.get("code", ""))
            if claim_id and code:
                issues_by_claim.setdefault(claim_id, []).append(code)
        annotated: list[Claim] = []
        for claim in claims:
            clone: Claim = deepcopy(claim)
            claim_id = str(clone.get("id", ""))
            clone["status"] = "supported" if claim_id not in issues_by_claim else "needs_repair"
            clone["issues"] = issues_by_claim.get(claim_id, [])
            clone["support_edges"] = [
                dict(edge)
                for edge in evaluation.get("support_edges", [])
                if isinstance(edge, dict) and edge.get("claim_id") == claim_id
            ]
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
        on_progress: ProgressCallback | None = None,
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

    def run(
        self,
        *,
        request: str,
        patient_record: str,
        selected_model: ModelAdapter,
        selected_model_metadata: ModelProfileMetadata,
        plan: Any | None = None,
        allow_general: bool = False,
        conversation_history: list[dict[str, Any]] | None = None,
        report_template: Any = None,
        on_progress: ProgressCallback | None = None,
        archive_result: bool = True,
    ) -> RunResult:
        """Run one full plan-execute-evaluate-repair cycle."""

        request = str(request or "").strip()
        patient_record = str(patient_record or "").strip()
        run_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            self.run_archive.start(run_id, created_at)
        except Exception:  # noqa: BLE001 - local navigation is optional
            pass
        model_call_metrics: list[dict[str, Any]] = []

        def observe_progress(event: dict[str, Any]) -> None:
            if event.get("stage") == "model_call" and isinstance(event.get("metrics"), dict):
                model_call_metrics.append(dict(event["metrics"]))
            if on_progress is not None:
                on_progress(event)

        emit_progress = make_progress_emitter(
            observe_progress,
            run_id,
            audit_callback=self._record_audit_event,
        )

        def emit_drained_model_metrics(
            model: ModelAdapter, task_id: int | None = None
        ) -> None:
            for metric in drain_model_metrics(model):
                emit_progress(
                    {
                        "stage": "model_call",
                        "message": "模型调用完成",
                        "task_id": task_id,
                        "metrics": metric,
                    }
                )

        def finish(result: RunResult) -> RunResult:
            if archive_result:
                self.archive_result(result)
            return result

        run_header = {
            "id": run_id,
            "created_at": created_at,
            "model": selected_model_metadata,
        }
        if not request:
            emit_progress(
                {
                    "stage": "error",
                    "message": "缺少用户问题，无法开始任务规划",
                    "status": "rejected",
                }
            )
            return finish(
                {
                    "status": "rejected",
                    "run": run_header,
                    "errors": [{"code": "REQUEST_REQUIRED", "message": "请提供用户请求。"}],
                }
            )
        if not patient_record and not allow_general:
            emit_progress(
                {
                    "stage": "error",
                    "message": "缺少患者病历，无法执行个体化分析",
                    "status": "rejected",
                }
            )
            return finish(
                {
                    "status": "rejected",
                    "run": run_header,
                    "errors": [
                        {"code": "PATIENT_RECORD_REQUIRED", "message": "请提供患者病历。"}
                    ],
                }
            )

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
            emit_drained_model_metrics(selected_model)
        except Exception:
            emit_drained_model_metrics(selected_model)
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
            return finish(
                {
                    "status": "plan_rejected",
                    "run": {**run_header, "plan": raw_plan},
                    "errors": validation["errors"],
                }
            )

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
            retrieval_limit=self.retrieval_limit,
            retrieval_candidate_budget=self.retrieval_candidate_budget,
            retrieval_max_per_document=self.retrieval_max_per_document,
            reranker=self.reranker,
        )

        execution = self._execute(tasks=tasks, agent=agent, on_progress=emit_progress)
        task_states = execution["tasks"]
        repair_history: list[dict[str, Any]] = []
        semantic_cache: dict[tuple[str, tuple[str, ...]], str] = {}
        verifier_model = self.verifier_model or selected_model

        for repair_round in range(self.max_repair_rounds + 1):
            claims = self.collect_claims(task_states)
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
            if claims:
                # Always validate collected claims even when some tasks failed.
                # Deterministic gates (NO_REF / BAD_REF / MISSING_* ) and the
                # semantic hook run regardless, so a needs_human_review report
                # never shows unverified conclusions as "supported".
                evidence_evaluation = evaluate_claims(
                    claims,
                    registry.as_map(),
                    verifier_model,
                    semantic_cache=semantic_cache,
                )
                emit_drained_model_metrics(verifier_model)
                combined_issues = execution_issues + evidence_evaluation["issues"]
                evaluation = {
                    "pass": not combined_issues,
                    "issues": combined_issues,
                    "judgements": evidence_evaluation["judgements"],
                    "support_edges": evidence_evaluation.get("support_edges", []),
                }
            else:
                evaluation = {
                    "pass": not execution_issues,
                    "issues": execution_issues,
                    "judgements": [],
                    "support_edges": [],
                }
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

            rerun_ids = set(repair.get("rerun_tasks", []))
            repair_tasks = [task for task in tasks if task["id"] in rerun_ids]
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
                rerun_task_ids=rerun_ids,
                on_progress=emit_progress,
            )
            task_states = execution["tasks"]
        else:  # pragma: no cover - every branch above breaks
            status = "needs_human_review"

        claims = self.annotate_claim_status(claims, evaluation)
        decision = route_decision(
            request=original_request,
            patient_record=patient_record,
            claims=claims,
            evaluation=evaluation,
        )
        run_header["model_calls"] = model_call_metrics
        run_header["model_usage"] = summarize_model_metrics(model_call_metrics)
        run_header["decision"] = decision
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
                "retrieval": (state.get("result") or {}).get("retrieval", {}),
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
        return finish(
            {
                "status": status,
                "run": {
                    **run_header,
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
            }
        )
