"""The three-stage task agent: retrieve -> extract -> synthesize."""

from __future__ import annotations

from typing import Any

from .evidence import EvidenceRegistry
from .model_adapter import ModelAdapter
from .retrieval import JsonKnowledgeBase, PatientRecordRetriever


class ThreeStageTaskAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        patient_retriever: PatientRecordRetriever,
        knowledge_base: JsonKnowledgeBase,
        registry: EvidenceRegistry,
        patient_record: str,
        request: str,
    ) -> None:
        self.model = model
        self.patient_retriever = patient_retriever
        self.knowledge_base = knowledge_base
        self.registry = registry
        self.patient_record = patient_record
        self.request = request

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

    def _retrieve(self, queries: list[str]) -> list[str]:
        evidence_ids: list[str] = []
        for query in queries:
            for fact in self.patient_retriever.search(query):
                item = self.registry.add_patient(
                    fact["text"],
                    locator=fact["locator"],
                    metadata={"retrieval_query": query, "score": fact["score"]},
                )
                evidence_ids.append(item["id"])

            for document in self.knowledge_base.search(query):
                item = self.registry.add_knowledge(
                    document["text"],
                    source=document["title"],
                    locator=document.get("locator", "知识库片段"),
                    document_id=document.get("id", document["title"]),
                    metadata={
                        "retrieval_query": query,
                        "score": document["score"],
                        "version": document.get("version", "未标注"),
                        "url": document.get("url", ""),
                        "synthetic": document.get("synthetic", True),
                    },
                )
                evidence_ids.append(item["id"])

        # Preserve retrieval order while removing duplicates.
        return list(dict.fromkeys(evidence_ids))

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
        payload: dict[str, Any], available_ids: set[str], task: dict[str, Any]
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
                        # This is assigned by server policy, not authored by the
                        # model. Retrieval/extraction tasks report source facts;
                        # analysis tasks must be grounded in both P* and K*.
                        "requires_dual_support": not any(
                            marker in task["goal"] for marker in ("提取", "检索")
                        ),
                    }
                )

        unknowns = [
            item.strip()
            for item in raw_unknowns[:5]
            if isinstance(item, str) and item.strip()
        ] if isinstance(raw_unknowns, list) else []
        return claims, unknowns

    def run(self, task: dict[str, Any], upstream: dict[int, Any]) -> dict[str, Any]:
        """Run all three stages for one task with compact, validated hand-offs."""

        # Stage 1: model proposes queries; code performs every actual retrieval.
        query_payload = self.model.make_queries(
            task=task,
            request=self.request,
            patient_record=self.patient_record,
            upstream=upstream,
        )
        queries = self._normalise_queries(query_payload)
        if not queries:
            queries = [task["goal"], self.request]
        local_evidence_ids = self._retrieve(queries)

        upstream_evidence_ids: list[str] = []
        for result in upstream.values():
            upstream_evidence_ids.extend((result or {}).get("evidence_ids", []))
        available_ids = set(local_evidence_ids) | set(upstream_evidence_ids)
        model_evidence = self.registry.model_view(sorted(available_ids))

        # Stage 2: extract facts, preserving a single source ID for each fact.
        fact_payload = self.model.extract_facts(task=task, evidence=model_evidence)
        facts = self._valid_facts(fact_payload, available_ids)

        # Stage 3: synthesize claims using IDs already issued by the server.
        result_payload = self.model.synthesize(
            task=task,
            request=self.request,
            facts=facts,
            evidence=model_evidence,
            upstream=upstream,
        )
        claims, unknowns = self._valid_result(result_payload, available_ids, task)

        return {
            "task_id": task["id"],
            "goal": task["goal"],
            "queries": queries,
            "evidence_ids": list(dict.fromkeys(local_evidence_ids)),
            "facts": facts,
            "claims": claims,
            "unknowns": unknowns,
        }
