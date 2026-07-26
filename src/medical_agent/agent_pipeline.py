"""The three-stage task agent: retrieve -> extract -> synthesize."""

from __future__ import annotations

from typing import Any, Callable

from .evidence import EvidenceRegistry
from .observability.model_metrics import drain_model_metrics
from .ports import KnowledgeBasePort, ModelAdapter
from .prompting import build_repair_context
from .retrieval.patient import PatientRecordRetriever
from .retrieval.state import RetrievalState


class ThreeStageTaskAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        patient_retriever: PatientRecordRetriever,
        knowledge_base: KnowledgeBasePort,
        registry: EvidenceRegistry,
        patient_record: str,
        request: str,
        patient_grounding_required: bool = True,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.patient_retriever = patient_retriever
        self.knowledge_base = knowledge_base
        self.registry = registry
        self.patient_record = patient_record
        self.request = request
        self.patient_grounding_required = patient_grounding_required
        # The callback is deliberately limited to audit events assembled by
        # this class.  It never receives raw model text or model reasoning.
        self.on_progress = on_progress
        self._repair_codes: dict[int, list[str]] = {}

    def set_repair_directives(self, directives: list[dict[str, Any]]) -> None:
        """Attach small code-owned hints for the next targeted rerun only."""

        normalized: dict[int, list[str]] = {}
        for directive in directives:
            if not isinstance(directive, dict) or not isinstance(
                directive.get("task_id"), int
            ):
                continue
            codes = [
                code
                for code in directive.get("codes", [])
                if isinstance(code, str)
            ][:8]
            if codes:
                normalized[directive["task_id"]] = list(dict.fromkeys(codes))
        self._repair_codes = normalized

    def _emit_progress(self, stage: str, message: str, **payload: Any) -> None:
        """Best-effort, presentation-safe task audit event.

        Task workers run concurrently, so a consumer must treat events as an
        interleaved trace.  The application wraps the callback with a lock before
        it reaches this method.  A disconnected stream must not fail a task.
        """

        if self.on_progress is None:
            return

        try:
            self.on_progress({"stage": stage, "message": message, **payload})
        except Exception:  # noqa: BLE001 - observability must not affect care workflow
            return

    def _emit_model_metrics(self, task_id: int) -> None:
        """Expose token/latency facts without exposing prompt or model text."""

        for metric in drain_model_metrics(self.model):
            self._emit_progress(
                "model_call",
                "模型调用完成",
                task_id=task_id,
                metrics=metric,
            )

    @staticmethod
    def _audit_text(value: Any, limit: int = 220) -> str:
        """Compact a visible audit field without returning model raw output."""

        text = " ".join(str(value or "").split())
        return text if len(text) <= limit else f"{text[:limit]}…"

    @staticmethod
    def _normalise_queries(payload: dict[str, Any]) -> list[str]:
        raw_queries = payload.get("queries", []) if isinstance(payload, dict) else []
        if not isinstance(raw_queries, list):
            return []
        queries: list[str] = []
        for item in raw_queries:
            value = " ".join(str(item).split())
            if value and value not in queries:
                queries.append(value)
        return queries[:3]

    def _retrieve(
        self, task: dict[str, Any], queries: list[str]
    ) -> tuple[list[str], dict[str, Any]]:
        retrieval_state = RetrievalState(max_rounds=2, max_candidates=12)
        retrieval_state.begin_round(queries)
        evidence_ids: list[str] = []

        # Patient facts are collected first so a bounded extraction prompt
        # cannot be starved by a highly ranked knowledge result.
        for query in queries:
            query_evidence_ids: list[str] = []
            for fact in self.patient_retriever.search(query):
                item = self.registry.add_patient(
                    fact["text"],
                    locator=fact["locator"],
                    metadata={"retrieval_query": query, "score": fact["score"]},
                )
                evidence_ids.append(item["id"])
                query_evidence_ids.append(item["id"])
            self._emit_progress(
                "retrieve",
                "已完成一条患者事实检索查询",
                task_id=task["id"],
                queries=[self._audit_text(query, 180)],
                evidence_ids=list(dict.fromkeys(query_evidence_ids)),
            )

        # JsonKnowledgeBase exposes a fused multi-query path. Custom knowledge
        # ports retain the old one-query-at-a-time compatibility path.
        search_many = getattr(self.knowledge_base, "search_many", None)
        if callable(search_many):
            documents = search_many(queries, limit=8)
            knowledge_ids: list[str] = []
            for document in documents:
                title = str(document.get("title", "知识库片段"))
                retrieval_queries = document.get("retrieval_queries", queries[:3])
                if not isinstance(retrieval_queries, list):
                    retrieval_queries = queries[:3]
                item = self.registry.add_knowledge(
                    str(document.get("text", "")),
                    source=title,
                    locator=document.get("locator", "知识库片段"),
                    document_id=document.get("document_id", document.get("id", title)),
                    metadata={
                        "retrieval_query": str(retrieval_queries[0]) if retrieval_queries else "",
                        "retrieval_queries": [
                            str(value) for value in retrieval_queries[:3] if str(value).strip()
                        ],
                        "score": document.get("score", 0),
                        "retrieval_score": document.get("retrieval_score", 0),
                        "retrieval_rank": document.get("retrieval_rank"),
                        "retrieval_method": document.get("retrieval_method", "rrf_lexical"),
                        "version": document.get("version", "未标注"),
                        "url": document.get("url", ""),
                        "synthetic": document.get("synthetic", True),
                    },
                )
                knowledge_ids.append(item["id"])
            evidence_ids.extend(knowledge_ids)
            if knowledge_ids:
                self._emit_progress(
                    "retrieve",
                    "已完成多查询融合检索",
                    task_id=task["id"],
                    queries=queries[:3],
                    evidence_ids=knowledge_ids,
                )
        else:
            for query in queries:
                query_evidence_ids = []
                for document in self.knowledge_base.search(query):
                    item = self.registry.add_knowledge(
                        document["text"],
                        source=document["title"],
                        locator=document.get("locator", "知识库片段"),
                        document_id=document.get(
                            "document_id", document.get("id", document["title"])
                        ),
                        metadata={
                            "retrieval_query": query,
                            "score": document["score"],
                            "version": document.get("version", "未标注"),
                            "url": document.get("url", ""),
                            "synthetic": document.get("synthetic", True),
                        },
                    )
                    evidence_ids.append(item["id"])
                    query_evidence_ids.append(item["id"])
                self._emit_progress(
                    "retrieve",
                    "已完成兼容检索查询",
                    task_id=task["id"],
                    queries=[self._audit_text(query, 180)],
                    evidence_ids=list(dict.fromkeys(query_evidence_ids)),
                )

        unique_ids = list(dict.fromkeys(evidence_ids))[: retrieval_state.max_candidates]
        retrieval_state.add_evidence(unique_ids)
        retrieval_state.finish()
        return unique_ids, retrieval_state.as_dict()

    @staticmethod
    def _valid_facts(
        payload: dict[str, Any], available_ids: set[str]
    ) -> list[dict[str, str]]:
        raw_facts = payload.get("facts", []) if isinstance(payload, dict) else []
        result: list[dict[str, str]] = []
        if not isinstance(raw_facts, list):
            return result
        for fact in raw_facts[:8]:
            if not isinstance(fact, dict):
                continue
            text = fact.get("text")
            ref = fact.get("ref")
            if isinstance(text, str) and text.strip() and isinstance(ref, str) and ref in available_ids:
                result.append({"text": text.strip(), "ref": ref})
        return result

    @staticmethod
    def _valid_result(
        payload: dict[str, Any],
        available_ids: set[str],
        task: dict[str, Any],
        patient_grounding_required: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        raw_claims = payload.get("claims", []) if isinstance(payload, dict) else []
        raw_unknowns = payload.get("unknowns", []) if isinstance(payload, dict) else []
        claims: list[dict[str, Any]] = []
        if isinstance(raw_claims, list):
            for raw_claim in raw_claims[:5]:
                if not isinstance(raw_claim, dict):
                    continue
                text = raw_claim.get("text")
                refs = raw_claim.get("refs", [])
                if not isinstance(text, str) or not text.strip() or not isinstance(refs, list):
                    continue
                valid_refs = [
                    ref for ref in refs if isinstance(ref, str) and ref in available_ids
                ]
                claims.append(
                    {
                        "text": text.strip(),
                        "refs": list(dict.fromkeys(valid_refs)),
                        "task_id": task["id"],
                        # Source tasks bypass model synthesis. Every remaining
                        # patient-specific conclusion requires both P* and K*.
                        "requires_dual_support": patient_grounding_required,
                    }
                )

        unknowns = [
            item.strip()
            for item in raw_unknowns[:5]
            if isinstance(item, str) and item.strip()
        ] if isinstance(raw_unknowns, list) else []
        return claims, unknowns

    @staticmethod
    def _source_task_prefix(task: dict[str, Any]) -> str | None:
        """Return the evidence prefix expected by a source task, if applicable."""

        goal = str(task.get("goal", "")).lower()
        source_action = any(
            marker in goal
            for marker in (
                "提取",
                "抽取",
                "检索",
                "查找",
                "extract",
                "retrieve",
                "search",
            )
        )
        if not source_action:
            return None
        has_patient = any(
            marker in goal for marker in ("患者", "病历", "patient", "record")
        )
        has_knowledge = any(
            marker in goal
            for marker in ("知识", "指南", "文献", "依据", "knowledge", "guideline")
        )
        if has_patient and not has_knowledge:
            return "P"
        if has_knowledge:
            return "K"
        return ""

    def run(self, task: dict[str, Any], upstream: dict[int, Any]) -> dict[str, Any]:
        """Run all three stages for one task with compact, validated hand-offs."""

        task_for_model = {
            **task,
            "patient_grounding_required": self.patient_grounding_required,
        }
        repair_codes = self._repair_codes.get(task["id"], [])
        if repair_codes:
            task_for_model["repair"] = build_repair_context(repair_codes)

        # Stage 1: model proposes queries; code performs every actual retrieval.
        self._emit_progress(
            "query",
            "正在生成检索查询",
            task_id=task["id"],
        )
        query_payload = self.model.make_queries(
            task=task_for_model,
            request=self.request,
            patient_record=self.patient_record,
            upstream=upstream,
        )
        self._emit_model_metrics(task["id"])
        queries = self._normalise_queries(query_payload)
        if not queries:
            queries = [task["goal"], self.request]
        repair_queries: list[str] = []
        if "MISSING_PATIENT_REF" in repair_codes:
            repair_queries.append(f"患者病历关键事实 {task['goal']}")
        if "MISSING_KB_REF" in repair_codes:
            repair_queries.append(f"医学知识库直接依据 {task['goal']}")
        if any(code in repair_codes for code in ("NO_REF", "BAD_REF", "NOT_SUPPORTED")):
            repair_queries.append(f"直接支持当前结论的证据 {task['goal']}")
        queries = list(dict.fromkeys(repair_queries + queries))[:3]
        local_evidence_ids, retrieval_state = self._retrieve(task, queries)

        upstream_evidence_ids: list[str] = []
        for result in upstream.values():
            upstream_evidence_ids.extend((result or {}).get("evidence_ids", []))
        ordered_available_ids = list(
            dict.fromkeys(local_evidence_ids + upstream_evidence_ids)
        )
        available_ids = set(ordered_available_ids)
        # Preserve retrieval order. Patient facts are registered before
        # knowledge results for each query, preventing a lexical K-before-P
        # sort from starving the bounded extraction prompt of patient context.
        model_evidence = self.registry.model_view(ordered_available_ids)

        def finish(
            *,
            facts: list[dict[str, str]] | None = None,
            claims: list[dict[str, Any]] | None = None,
            unknowns: list[str] | None = None,
        ) -> dict[str, Any]:
            return {
                "task_id": task["id"],
                "goal": task["goal"],
                "queries": queries,
                "evidence_ids": list(dict.fromkeys(local_evidence_ids)),
                "facts": facts or [],
                "claims": claims or [],
                "unknowns": unknowns or [],
                "retrieval": retrieval_state,
            }

        if not model_evidence:
            # Query generation is required for retrieval, but extraction and
            # synthesis cannot add value without server-registered evidence.
            self._emit_progress(
                "extract",
                "未检索到证据，已跳过关键信息提取",
                task_id=task["id"],
                facts=[],
            )
            self._emit_progress(
                "synthesize",
                "没有可引用事实，已跳过结论生成",
                task_id=task["id"],
                claims=[],
            )
            return finish(unknowns=["未检索到可用于该子任务的证据。"])

        # Stage 2: extract facts, preserving a single source ID for each fact.
        self._emit_progress(
            "extracting",
            "正在提取证据关键信息",
            task_id=task["id"],
        )
        fact_payload = self.model.extract_facts(task=task_for_model, evidence=model_evidence)
        self._emit_model_metrics(task["id"])
        facts = self._valid_facts(fact_payload, available_ids)
        self._emit_progress(
            "extract",
            "已完成证据关键信息提取",
            task_id=task["id"],
            facts=[
                {"ref": fact["ref"], "summary": self._audit_text(fact["text"])}
                for fact in facts
            ],
        )

        if not facts:
            self._emit_progress(
                "synthesize",
                "未抽取到可引用事实，已跳过结论生成",
                task_id=task["id"],
                claims=[],
            )
            return finish(
                unknowns=["证据存在，但未抽取到与该子任务直接相关的事实。"]
            )

        source_prefix = self._source_task_prefix(task)
        if source_prefix is not None:
            # Extraction/retrieval tasks have already produced atomic facts
            # with one validated ref each.  A second model paraphrase would add
            # cost and hallucination risk without adding information.
            source_facts = [
                fact
                for fact in facts
                if not source_prefix or fact["ref"].startswith(source_prefix)
            ]
            if not source_facts:
                self._emit_progress(
                    "synthesize",
                    "未抽取到目标来源事实，已跳过结论生成",
                    task_id=task["id"],
                    claims=[],
                )
                return finish(
                    facts=facts,
                    unknowns=["未抽取到与来源任务类型匹配的可引用事实。"],
                )
            claims = [
                {
                    "text": fact["text"],
                    "refs": [fact["ref"]],
                    "task_id": task["id"],
                    "requires_dual_support": False,
                }
                for fact in source_facts[:5]
            ]
            self._emit_progress(
                "synthesize",
                "来源任务已直接生成带引用的事实摘要",
                task_id=task["id"],
                claims=[
                    {
                        "refs": claim["refs"],
                        "summary": self._audit_text(claim["text"]),
                    }
                    for claim in claims
                ],
            )
            return finish(facts=facts, claims=claims)

        # Stage 3: synthesize claims using IDs already issued by the server.
        self._emit_progress(
            "synthesizing",
            "正在生成带引用的结论",
            task_id=task["id"],
        )
        result_payload = self.model.synthesize(
            task=task_for_model,
            request=self.request,
            facts=facts,
        )
        self._emit_model_metrics(task["id"])
        claims, unknowns = self._valid_result(
            result_payload,
            available_ids,
            task,
            self.patient_grounding_required,
        )
        self._emit_progress(
            "synthesize",
            "已生成带引用的任务结论",
            task_id=task["id"],
            claims=[
                {
                    "refs": list(claim.get("refs", [])),
                    "summary": self._audit_text(claim.get("text", "")),
                }
                for claim in claims
            ],
        )

        return finish(facts=facts, claims=claims, unknowns=unknowns)
