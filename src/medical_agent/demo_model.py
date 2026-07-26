"""Deterministic adapter used to make the MVP runnable without an LLM API."""

from __future__ import annotations

from typing import Any

from .contracts import FactPayload, PlanPayload, QueryPayload, SynthesisPayload


def _short(value: str, limit: int = 72) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}…"


class DemoModelAdapter:
    """A deliberately conservative stand-in for a weak JSON-capable model."""

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "mode": "demo",
            "provider": "local-demo",
            "name": "demo",
        }

    def plan(self, request: str, patient_record: str) -> PlanPayload:
        if not patient_record.strip():
            return {
                "tasks": [
                    {
                        "id": 1,
                        "goal": "检索与用户问题相关的医学知识依据",
                        "deps": [],
                        "evidence_scope": "knowledge",
                        "analysis_mode": "retrieval",
                    },
                    {
                        "id": 2,
                        "goal": "基于知识库证据回答问题并说明局限",
                        "deps": [1],
                        "evidence_scope": "both",
                        "analysis_mode": "synthesis",
                    },
                ]
            }
        return {
            "tasks": [
                {
                    "id": 1,
                    "goal": "提取与用户请求相关的患者事实",
                    "deps": [],
                    "evidence_scope": "patient",
                    "analysis_mode": "retrieval",
                },
                {
                    "id": 2,
                    "goal": "检索与患者情况相关的医学知识依据",
                    "deps": [1],
                    "evidence_scope": "knowledge",
                    "analysis_mode": "retrieval",
                },
                {
                    "id": 3,
                    "goal": "基于患者事实和医学依据分析请求中的临床问题",
                    "deps": [1, 2],
                    "evidence_scope": "both",
                    "analysis_mode": "analysis",
                },
                {
                    "id": 4,
                    "goal": "识别风险、禁忌和待补充信息",
                    "deps": [1, 2],
                    "evidence_scope": "both",
                    "analysis_mode": "risk_review",
                },
                {
                    "id": 5,
                    "goal": "整合分析，形成需专业人员复核的结论",
                    "deps": [3, 4],
                    "evidence_scope": "both",
                    "analysis_mode": "synthesis",
                },
            ]
        }

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> QueryPayload:
        queries = [task["goal"], request]
        for result in upstream.values():
            for claim in (result or {}).get("claims", [])[:1]:
                queries.append(claim.get("text", ""))
        unique: list[str] = []
        for query in queries:
            query = " ".join(str(query).split())
            if query and query not in unique:
                unique.append(query)
        return {"queries": unique[:3]}

    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> FactPayload:
        facts = [
            {"text": _short(item["text"], 120), "ref": item["id"]}
            for item in evidence[:6]
        ]
        return {"facts": facts}

    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
    ) -> SynthesisPayload:
        patient = next((item for item in facts if item["ref"].startswith("P")), None)
        knowledge = next((item for item in facts if item["ref"].startswith("K")), None)

        if not facts:
            return {
                "claims": [],
                "unknowns": ["没有检索到足以支持该子任务的证据。"],
            }

        if not patient and knowledge:
            return {
                "claims": [
                    {
                        "text": (
                            f"知识库中与“{_short(request, 48)}”相关的资料指出："
                            f"{_short(knowledge['text'])}"
                        ),
                        "refs": [knowledge["ref"]],
                    }
                ],
                "unknowns": [],
            }

        if patient and knowledge:
            analysis_mode = str(task.get("analysis_mode", "general"))
            if analysis_mode == "risk_review":
                text = (
                    "基于现有病历和知识库证据，应由临床专业人员重点核实"
                    "禁忌证、过敏史、肝肾功能及缺失检查结果后再作处置决定。"
                )
            elif analysis_mode == "synthesis":
                text = (
                    "综合现有证据，本结果仅提供可追溯的临床决策支持；"
                    "具体诊疗、用药和剂量必须由具备资质的临床专业人员复核。"
                )
            else:
                text = (
                    f"针对“{_short(request, 48)}”，现有患者事实与医学知识存在相关依据，"
                    "但需要结合完整病史、检查结果和临床判断后才能确定具体处置。"
                )
            return {
                "claims": [
                    {"text": text, "refs": [patient["ref"], knowledge["ref"]]}
                ],
                "unknowns": [],
            }

        return {
            "claims": [],
            "unknowns": ["患者事实或外部医学依据不完整，无法形成双重证据支持的结论。"],
        }

    def judge_claims(self, items: list[dict[str, Any]]) -> dict[str, str]:
        return {
            item["id"]: "SUPPORTED" if item.get("evidence") else "NOT_SUPPORTED"
            for item in items
        }
