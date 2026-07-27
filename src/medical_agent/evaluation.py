"""Offline evaluation assets and smoke-run orchestration.

The production workflow deliberately does not download or embed public medical
datasets.  This module keeps the boundary explicit: a small, synthetic fixture
can be run in CI, while real datasets are described by a validated manifest and
loaded by dataset-specific adapters outside the request path.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .evaluator import evaluate_claims
from .quality import (
    evaluate_counterfactual_cases,
    evaluate_evidence_chain,
    evaluate_retrieval_cases,
)
from .retrieval.knowledge import JsonKnowledgeBase


MANIFEST_SCHEMA_VERSION = 1
MAX_DATASETS = 128
MAX_FIXTURE_CASES = 2_048
REQUIRED_DATASET_FIELDS = frozenset(
    {
        "id",
        "name",
        "task",
        "role",
        "access",
        "language",
        "size_note",
        "official_url",
        "citation_url",
        "metrics",
        "notes",
    }
)
VALID_ACCESS = frozenset(
    {
        "downloadable",
        "public_dev_registration_train",
        "registration",
        "credentialed",
        "human_only",
        "paper_access",
        "terms_check",
    }
)


class EvaluationAssetError(ValueError):
    """Raised when a local evaluation manifest or fixture is unsafe/invalid."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise EvaluationAssetError(f"评测文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationAssetError(f"评测文件不是有效 JSON: {path}") from exc
    except OSError as exc:
        raise EvaluationAssetError(f"无法读取评测文件: {path}") from exc


def _bounded_text(value: Any, *, field: str, limit: int = 4_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationAssetError(f"评测字段 {field} 必须是非空字符串")
    return " ".join(value.split())[:limit]


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate dataset metadata without downloading any data.

    The returned records intentionally contain links, access conditions and
    metric mappings only.  Dataset contents, patient text and credentials are
    never accepted as part of this manifest.
    """

    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise EvaluationAssetError("数据集清单顶层必须是对象")
    schema_version = payload.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise EvaluationAssetError(
            f"不支持的数据集清单版本: {schema_version!r}"
        )
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise EvaluationAssetError("数据集清单必须包含非空 datasets 数组")
    if len(raw_datasets) > MAX_DATASETS:
        raise EvaluationAssetError(f"数据集数量不能超过 {MAX_DATASETS}")

    datasets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_datasets, start=1):
        if not isinstance(raw, dict):
            raise EvaluationAssetError(f"第 {index} 个数据集记录必须是对象")
        missing = sorted(REQUIRED_DATASET_FIELDS - raw.keys())
        if missing:
            raise EvaluationAssetError(
                f"数据集记录 {index} 缺少字段: {', '.join(missing)}"
            )
        dataset_id = _bounded_text(raw["id"], field="id", limit=120)
        if dataset_id in seen_ids:
            raise EvaluationAssetError(f"数据集 ID 重复: {dataset_id}")
        seen_ids.add(dataset_id)
        access = _bounded_text(raw["access"], field="access", limit=64)
        if access not in VALID_ACCESS:
            raise EvaluationAssetError(f"未知数据访问级别: {access}")
        metrics = raw["metrics"]
        if not isinstance(metrics, list) or not metrics:
            raise EvaluationAssetError(f"数据集 {dataset_id} 的 metrics 必须非空")
        if any(not isinstance(metric, str) or not metric.strip() for metric in metrics):
            raise EvaluationAssetError(f"数据集 {dataset_id} 的 metrics 含无效值")
        official_url = _bounded_text(raw["official_url"], field="official_url", limit=1_000)
        citation_url = _bounded_text(raw["citation_url"], field="citation_url", limit=1_000)
        if not official_url.startswith(("https://", "http://")):
            raise EvaluationAssetError(f"数据集 {dataset_id} 的 official_url 必须为 URL")
        if not citation_url.startswith(("https://", "http://")):
            raise EvaluationAssetError(f"数据集 {dataset_id} 的 citation_url 必须为 URL")
        normalized = dict(raw)
        for field in ("name", "task", "role", "language", "size_note", "notes"):
            normalized[field] = _bounded_text(raw[field], field=field)
        normalized["id"] = dataset_id
        normalized["access"] = access
        normalized["official_url"] = official_url
        normalized["citation_url"] = citation_url
        normalized["metrics"] = [str(metric).strip()[:120] for metric in metrics[:32]]
        datasets.append(normalized)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "updated_at": _bounded_text(payload.get("updated_at", "unknown"), field="updated_at"),
        "datasets": datasets,
    }


def load_smoke_fixture(path: str | Path) -> dict[str, Any]:
    """Load the small synthetic fixture used for offline regression checks."""

    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise EvaluationAssetError("冒烟 fixture 顶层必须是对象")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EvaluationAssetError("冒烟 fixture 的 schema_version 不受支持")
    if payload.get("synthetic") is not True:
        raise EvaluationAssetError("冒烟 fixture 必须显式标记 synthetic=true")
    retrieval = payload.get("retrieval_cases", [])
    counterfactual = payload.get("counterfactual_cases", [])
    if not isinstance(retrieval, list) or not isinstance(counterfactual, list):
        raise EvaluationAssetError("冒烟 fixture 的 cases 必须是数组")
    if len(retrieval) + len(counterfactual) > MAX_FIXTURE_CASES:
        raise EvaluationAssetError("冒烟 fixture 超出案例数量上限")
    chain = payload.get("evidence_chain")
    if not isinstance(chain, dict):
        raise EvaluationAssetError("冒烟 fixture 缺少 evidence_chain 对象")
    return payload


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    datasets = manifest.get("datasets", [])
    access_counts = Counter(
        str(dataset.get("access", "unknown"))
        for dataset in datasets
        if isinstance(dataset, dict)
    )
    return {
        "dataset_count": len(datasets),
        "access_counts": dict(sorted(access_counts.items())),
        "dataset_ids": [
            str(dataset.get("id"))
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("id")
        ],
    }


def run_local_smoke_evaluation(
    manifest_path: str | Path,
    fixture_path: str | Path,
    *,
    knowledge_base: JsonKnowledgeBase | None = None,
) -> dict[str, Any]:
    """Run all deterministic quality metrics against the synthetic fixture.

    No model, network, patient record or external dataset is used.  The
    callback exercises the same ``search_many`` contract used by the runtime.
    """

    manifest = load_dataset_manifest(manifest_path)
    fixture = load_smoke_fixture(fixture_path)
    knowledge = knowledge_base or JsonKnowledgeBase.demo()

    def search(query: str) -> list[dict[str, Any]]:
        return knowledge.search_many([query], limit=8)

    retrieval_result = evaluate_retrieval_cases(
        fixture.get("retrieval_cases", []),
        search,
        ks=(1, 3, 5),
    )
    chain_fixture = fixture["evidence_chain"]
    claims = chain_fixture.get("claims", [])
    evidence = chain_fixture.get("evidence", [])
    chain_evaluation = chain_fixture.get("evaluation")
    if not isinstance(claims, list) or not isinstance(evidence, list):
        raise EvaluationAssetError("evidence_chain 的 claims/evidence 必须为数组")
    if not isinstance(chain_evaluation, dict):
        chain_evaluation = evaluate_claims(claims, evidence)
    evidence_result = evaluate_evidence_chain(claims, evidence, chain_evaluation)
    counterfactual_result = evaluate_counterfactual_cases(
        fixture.get("counterfactual_cases", [])
    )
    return {
        "manifest": _manifest_summary(manifest),
        "fixture": {
            "synthetic": True,
            "retrieval_cases": retrieval_result["case_count"],
            "counterfactual_cases": counterfactual_result["case_count"],
        },
        "retrieval": retrieval_result,
        "evidence_chain": evidence_result,
        "counterfactual": counterfactual_result,
    }


__all__ = [
    "EvaluationAssetError",
    "load_dataset_manifest",
    "load_smoke_fixture",
    "run_local_smoke_evaluation",
]
