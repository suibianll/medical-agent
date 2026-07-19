"""Deterministic adapter used to make the MVP runnable without an LLM API."""

from __future__ import annotations

from typing import Any

from .model_adapter import ModelAdapter


def _short(value: str, limit: int = 72) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}…"


class DemoModelAdapter(ModelAdapter):
    """A deliberately conservative stand-in for a weak JSON-capable model."""

    def plan(self, request: str, patient_record: str) -> dict[str, Any]:
        return {
            "tasks": [
                {"id": 1, "goal": "提取与用户请求相关的患者事实", "deps": []},
                {"id": 2, "goal": "检索与患者情况相关的医学知识依据", "deps": [1]},
                {"id": 3, "goal": "基于患者事实和医学依据分析请求中的临床问题", "deps": [1, 2]},
                {"id": 4, "goal": "识别风险、禁忌和待补充信息", "deps": [1, 2]},
                {"id": 5, "goal": "整合分析，形成需专业人员复核的结论", "deps": [3, 4]},
            ]
        }

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
        evidence: list[dict[str, str]],
        upstream: dict[int, Any],
    ) -> dict[str, Any]:
        patient = next((item for item in evidence if item["id"].startswith("P")), None)
        knowledge = next((item for item in evidence if item["id"].startswith("K")), None)
        task_id = task["id"]

        if not facts:
            return {
                "claims": [],
                "unknowns": ["没有检索到足以支持该子任务的证据。"],
            }

        if task_id == 1 and patient:
            return {
                "claims": [
                    {
                        "text": f"病历中与当前问题相关的信息包括：{_short(patient['text'])}",
                        "refs": [patient["id"]],
                    }
                ],
                "unknowns": [],
            }

        if task_id == 2 and knowledge:
            return {
                "claims": [
                    {
                        "text": f"知识库中检索到的相关依据指出：{_short(knowledge['text'])}",
                        "refs": [knowledge["id"]],
                    }
                ],
                "unknowns": [],
            }

        if patient and knowledge:
            if task_id == 4:
                text = (
                    "基于现有病历和知识库证据，应由临床专业人员重点核实"
                    "禁忌证、过敏史、肝肾功能及缺失检查结果后再作处置决定。"
                )
            elif task_id == 5:
                text = (
                    "综合现有证据，本结果仅提供可追溯的临床决策支持；"
                    "具体诊疗、用药和剂量必须由具备资质的临床专业人员复核。"
                )
            else:
                text = (
                    f"针对“{_short(request, 48)}”，现有患者事实与医学知识存在相关依据，"
                    "但需要结合完整病史、检查结果和临床判断后才能确定具体处置。"
                )
            return {"claims": [{"text": text, "refs": [patient["id"], knowledge["id"]]}], "unknowns": []}

        return {
            "claims": [],
            "unknowns": ["患者事实或外部医学依据不完整，无法形成双重证据支持的结论。"],
        }

    def judge_claim(self, *, claim: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        return "SUPPORTED" if evidence else "NOT_SUPPORTED"
