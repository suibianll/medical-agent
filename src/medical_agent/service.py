"""Application service that coordinates planning, execution, evaluation and repair."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from .agent_pipeline import ThreeStageTaskAgent
from .dag_scheduler import execute_dag
from .demo_model import DemoModelAdapter
from .evaluator import evaluate_claims
from .evidence import EvidenceRegistry
from .graph import build_evidence_graph
from .model_adapter import ModelAdapter
from .plan_validator import validate_plan
from .repair import build_repair_plan
from .report import render_report
from .retrieval import JsonKnowledgeBase, PatientRecordRetriever


class MedicalAgentService:
    """Local MVP service with a replaceable model and knowledge-base adapter."""

    def __init__(
        self,
        *,
        model: ModelAdapter | None = None,
        knowledge_base: JsonKnowledgeBase | None = None,
        max_repair_rounds: int = 2,
        max_workers: int = 3,
    ) -> None:
        self.model = model or DemoModelAdapter()
        self.knowledge_base = knowledge_base or JsonKnowledgeBase.demo()
        self.max_repair_rounds = max_repair_rounds
        self.max_workers = max_workers

    @classmethod
    def from_environment(cls) -> "MedicalAgentService":
        """Use the real OpenAI-compatible adapter only when explicitly configured."""

        import os

        api_key = os.getenv("MEDICAL_AGENT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("MEDICAL_AGENT_BASE_URL")
        model_name = os.getenv("MEDICAL_AGENT_MODEL")
        if api_key and base_url and model_name:
            from .aliyun_model import AliyunCompatibleModelAdapter

            return cls(
                model=AliyunCompatibleModelAdapter(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                ),
                # A little lower than local demo concurrency to be friendlier to
                # provider quotas while retaining parallel DAG behavior.
                max_workers=2,
            )
        return cls()

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
            annotated.append(clone)
        return annotated

    def _execute(
        self,
        *,
        tasks: list[dict[str, Any]],
        agent: ThreeStageTaskAgent,
        prior_states: dict[int, dict[str, Any]] | None = None,
        rerun_task_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        def worker(task: dict[str, Any], upstream: dict[int, Any]) -> dict[str, Any]:
            return agent.run(task, upstream)

        return execute_dag(
            tasks,
            worker,
            max_workers=self.max_workers,
            initial_task_states=prior_states,
            rerun_task_ids=rerun_task_ids,
        )

    def run(
        self,
        request: str | dict[str, Any],
        patient_record: str = "",
        plan: Any | None = None,
        **legacy: Any,
    ) -> dict[str, Any]:
        """Run a full plan-execute-evaluate cycle.

        ``record`` is accepted as a compatibility alias for callers that use a
        shorter field name.  The method deliberately returns JSON-safe values.
        """

        if isinstance(request, dict):
            payload = request
            patient_record = patient_record or str(
                payload.get("patientRecord", payload.get("record", ""))
            )
            plan = plan if plan is not None else payload.get("plan")
            request = str(payload.get("request", ""))

        if not patient_record:
            patient_record = str(legacy.get("record", ""))
        request = str(request or "").strip()
        patient_record = str(patient_record or "").strip()
        run_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        if not request:
            return {
                "status": "rejected",
                "run": {"id": run_id, "created_at": created_at},
                "errors": [{"code": "REQUEST_REQUIRED", "message": "请提供用户请求。"}],
            }
        if not patient_record:
            return {
                "status": "rejected",
                "run": {"id": run_id, "created_at": created_at},
                "errors": [{"code": "PATIENT_RECORD_REQUIRED", "message": "请提供患者病历。"}],
            }

        raw_plan = plan if plan is not None else self.model.plan(request, patient_record)
        validation = validate_plan(raw_plan)
        if not validation["valid"]:
            return {
                "status": "plan_rejected",
                "run": {"id": run_id, "created_at": created_at, "plan": raw_plan},
                "errors": validation["errors"],
            }

        tasks = validation["tasks"]
        registry = EvidenceRegistry()
        agent = ThreeStageTaskAgent(
            model=self.model,
            patient_retriever=PatientRecordRetriever(patient_record),
            knowledge_base=self.knowledge_base,
            registry=registry,
            patient_record=patient_record,
            request=request,
        )

        execution = self._execute(tasks=tasks, agent=agent)
        task_states = execution["tasks"]
        repair_history: list[dict[str, Any]] = []

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
            evaluation = (
                {"pass": False, "issues": execution_issues, "judgements": []}
                if execution_issues
                else evaluate_claims(claims, registry.as_map(), self.model)
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
                status = "needs_human_review"
                break

            execution = self._execute(
                tasks=tasks,
                agent=agent,
                prior_states=task_states,
                rerun_task_ids=set(repair["rerun_tasks"]),
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
            request=request,
            task_states=task_states,
            claims=claims,
            evidence=evidence,
            evaluation=evaluation,
            status=status,
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
        return {
            "status": status,
            "run": {
                "id": run_id,
                "created_at": created_at,
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
