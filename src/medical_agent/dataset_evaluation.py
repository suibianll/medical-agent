"""Dataset adapters and controlled evaluation for the medical agent.

The module keeps benchmark-specific parsing outside the request path.  Every
adapter emits a small, common case contract, while the runner records only
metrics, identifiers, statuses and safe cost telemetry.  Raw questions,
patient text, evidence text and model responses are never written to the
report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from zipfile import ZipFile

from .bootstrap import (
    _build_embedding_provider,
    create_agent,
    create_agent_from_environment,
)
from .demo_model import DemoModelAdapter
from .quality import evaluate_counterfactual_cases, evaluate_retrieval_cases
from .retrieval.knowledge import JsonKnowledgeBase
from .retrieval.vector import FaissKnowledgeBase
from .infrastructure.model_config import load_model_configuration
from .observability.model_metrics import summarize_model_metrics


MAX_CASES = 2_048
MAX_DOCUMENTS_PER_CASE = 512
MAX_TEXT_CHARS = 1_000_000
DEFAULT_MAX_CASES = 8
DEFAULT_TOP_K = 5
DEFAULT_DATA_ROOT = Path("data") / "evaluation"
DEFAULT_MANIFEST = Path("evaluation") / "datasets.json"
SUPPORTED_RETRIEVAL_BACKENDS = ("lexical", "faiss")
SUPPORTED_DATASETS = (
    "pubmedqa",
    "medmcqa",
    "medqa-usmle",
    "evidence-inference-2",
    "evidencebench",
    "faithfulness-qa-2026",
    "ragchecker",
)
REFERENCE_ONLY_DATASET_REASONS = {
    "bioasq": "BioASQ 需要官方 challenge 语料/访问权限，当前未下载可复现的本地 adapter。",
    "mednli": "MedNLI 需要 PhysioNet credential 和 DUA，当前运行环境不含授权文本。",
    "mimic-iv": "MIMIC-IV 需要 PhysioNet credential、培训和 DUA，禁止在仓库外泄露原始文本。",
    "healthsearchqa": "HealthSearchQA 缺少稳定的自动 gold，需要按人工 rubric 评测。",
    "bridge": "BRIDGE 当前仅登记来源和论文，未提供项目可直接消费的本地评测文件。",
    "medqa-cs-2026": "MedQA-CS-2026 当前没有已核验的本地评测文件。",
}
ALL_DATASETS = SUPPORTED_DATASETS + tuple(REFERENCE_ONLY_DATASET_REASONS)


class DatasetEvaluationError(RuntimeError):
    """Base error for malformed or unavailable benchmark assets."""


class DatasetUnavailable(DatasetEvaluationError):
    """Raised when a dataset needs credentials or an optional dependency."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Normalized benchmark case consumed by retrieval and agent evaluators."""

    case_id: str
    query: str
    relevant_ids: tuple[str, ...] = ()
    gold_answer: str | None = None
    answer_type: str = "none"
    options: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    variants: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = ()
    expected_variant_change: bool | None = None


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A bounded, split-specific collection of cases and retrievable docs."""

    dataset_id: str
    split: str
    cases: tuple[EvaluationCase, ...]
    documents: tuple[dict[str, Any], ...]
    answer_supported: bool = False
    notes: tuple[str, ...] = ()


class ModelCallCounter:
    """Count adapter method calls even when a provider emits no telemetry."""

    def __init__(self) -> None:
        self.calls = 0
        self.by_stage: Counter[str] = Counter()

    def mark(self, stage: str) -> None:
        self.calls += 1
        self.by_stage[stage] += 1


class CountingModelAdapter:
    """Transparent model proxy used only by the evaluator."""

    def __init__(self, delegate: Any, counter: ModelCallCounter) -> None:
        self._delegate = delegate
        self._counter = counter

    def runtime_metadata(self) -> dict[str, str]:
        method = getattr(self._delegate, "runtime_metadata", None)
        value = method() if callable(method) else {}
        return value if isinstance(value, dict) else {}

    def drain_call_metrics(self) -> list[dict[str, Any]]:
        method = getattr(self._delegate, "drain_call_metrics", None)
        values = method() if callable(method) else []
        return values if isinstance(values, list) else []

    def plan(self, request: str, patient_record: str) -> Any:
        self._counter.mark("plan")
        return self._delegate.plan(request, patient_record)

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> Any:
        self._counter.mark("make_queries")
        return self._delegate.make_queries(
            task=task,
            request=request,
            patient_record=patient_record,
            upstream=upstream,
        )

    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> Any:
        self._counter.mark("extract_facts")
        return self._delegate.extract_facts(task=task, evidence=evidence)

    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
    ) -> Any:
        self._counter.mark("synthesize")
        return self._delegate.synthesize(task=task, request=request, facts=facts)

    def judge_claims(self, items: list[dict[str, Any]]) -> Any:
        self._counter.mark("judge_claims")
        return self._delegate.judge_claims(items)


def _bounded_text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return text[:limit]


def _safe_id(value: Any, *, prefix: str = "case") -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip())
    text = text.strip("_")[:160]
    return text or f"{prefix}-{hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:12]}"


def _document(
    document_id: str,
    text: str,
    *,
    title: str = "评测语料片段",
    locator: str = "dataset",
) -> dict[str, Any]:
    content = _bounded_text(text)
    return {
        "id": document_id,
        "document_id": document_id,
        "title": _bounded_text(title, limit=240) or "评测语料片段",
        "text": content,
        "locator": _bounded_text(locator, limit=240),
        "version": "dataset-local",
        "keywords": [],
        "priority": 50,
        "synthetic": False,
        "source_type": "dataset",
        "url": "",
    }


def _evaluation_index_path(
    data_root: str | Path,
    dataset_id: str,
    split: str,
    suffix: str,
) -> Path:
    index_root = Path(data_root) / ".runtime-indexes"
    safe_dataset = _safe_id(dataset_id, prefix="dataset")
    safe_split = _safe_id(split, prefix="split")
    safe_suffix = _safe_id(suffix, prefix="case")
    return index_root / f"{safe_dataset}-{safe_split}-{safe_suffix}.faiss"


def _build_evaluation_knowledge_base(
    source: JsonKnowledgeBase,
    *,
    retrieval_backend: str,
    data_root: str | Path,
    dataset_id: str,
    split: str,
    suffix: str = "base",
) -> Any:
    if retrieval_backend == "lexical":
        return source
    configuration = load_model_configuration()
    retrieval = configuration.retrieval
    if retrieval.backend != "faiss":
        raise DatasetEvaluationError(
            "评测请求使用 FAISS，但当前模型配置 retrieval.backend 不是 faiss。"
        )
    return FaissKnowledgeBase(
        source,
        embedding_provider=_build_embedding_provider(retrieval.embedding),
        index_path=_evaluation_index_path(data_root, dataset_id, split, suffix),
    )


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise DatasetUnavailable(f"数据文件不存在: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetEvaluationError(f"无法读取 JSON 数据: {path}") from exc


def _read_jsonl(path: Path, *, max_cases: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetUnavailable(f"数据文件不存在: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= max_cases:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetEvaluationError(f"无法读取 JSONL 数据: {path}") from exc
    return rows


def _read_jsonl_zip(path: Path, *, member: str, max_cases: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetUnavailable(f"数据文件不存在: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            candidate = next((name for name in names if name.endswith(member)), None)
            if candidate is None:
                raise DatasetEvaluationError(f"压缩包缺少数据文件: {member}")
            with archive.open(candidate) as raw:
                for line in raw:
                    if len(rows) >= max_cases:
                        break
                    try:
                        value = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
    except DatasetEvaluationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise DatasetEvaluationError(f"无法读取压缩 JSONL: {path}") from exc
    return rows


def _bounded_cases(cases: Iterable[EvaluationCase], max_cases: int) -> tuple[EvaluationCase, ...]:
    try:
        limit = max(1, min(int(max_cases), MAX_CASES))
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_CASES
    return tuple(case for index, case in enumerate(cases) if index < limit)


def _pubmedqa(root: Path, split: str, max_cases: int) -> DatasetBundle:
    payload = _read_json(root / "pubmedqa" / "ori_pqal.json")
    if not isinstance(payload, dict):
        raise DatasetEvaluationError("PubMedQA ori_pqal.json 必须是对象")
    gold_path = root / "pubmedqa" / "test_ground_truth.json"
    gold = _read_json(gold_path) if gold_path.is_file() else {}
    gold = gold if isinstance(gold, dict) else {}
    identifiers = list(payload)
    if split == "test" and gold:
        identifiers = [identifier for identifier in gold if identifier in payload]
    elif split == "train" and gold:
        test_ids = set(gold)
        identifiers = [identifier for identifier in identifiers if identifier not in test_ids]
    elif split not in {"all", "test", "train"}:
        raise DatasetEvaluationError("PubMedQA split 只能是 train、test 或 all")

    cases: list[EvaluationCase] = []
    documents: list[dict[str, Any]] = []
    for identifier in identifiers[: max_cases]:
        row = payload.get(identifier)
        if not isinstance(row, dict):
            continue
        case_id = _safe_id(identifier, prefix="pubmedqa")
        contexts = row.get("CONTEXTS", [])
        if not isinstance(contexts, list):
            contexts = []
        relevant: list[str] = []
        for index, context in enumerate(contexts[:MAX_DOCUMENTS_PER_CASE]):
            document_id = f"{case_id}-ctx-{index}"
            documents.append(
                _document(
                    document_id,
                    str(context),
                    title=f"PubMedQA {identifier}",
                    locator=f"context-{index}",
                )
            )
            relevant.append(document_id)
        answer = gold.get(identifier, row.get("final_decision"))
        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=_bounded_text(row.get("QUESTION", ""), limit=16_000),
                relevant_ids=tuple(relevant),
                gold_answer=_bounded_text(answer, limit=80).lower() or None,
                answer_type="label",
                metadata={"source_id": str(identifier), "year": str(row.get("YEAR", ""))},
            )
        )
    return DatasetBundle(
        "pubmedqa",
        split,
        _bounded_cases(cases, max_cases),
        tuple(documents),
        answer_supported=True,
        notes=("PQA-L；test split 使用官方 test_ground_truth。",),
    )


def _evidencebench(root: Path, split: str, max_cases: int) -> DatasetBundle:
    if split not in {"train", "dev", "test"}:
        raise DatasetEvaluationError("EvidenceBench split 只能是 train、dev 或 test")
    payload = _read_json(root / "evidencebench" / f"evidencebench_{split}_set.json")
    if not isinstance(payload, dict):
        raise DatasetEvaluationError("EvidenceBench split 文件必须是对象")
    cases: list[EvaluationCase] = []
    documents: list[dict[str, Any]] = []
    for raw_id, row in list(payload.items())[:max_cases]:
        if not isinstance(row, dict):
            continue
        case_id = _safe_id(raw_id, prefix="evidencebench")
        pool = row.get("paper_as_candidate_pool", [])
        if not isinstance(pool, list):
            pool = []
        indices: set[int] = set()
        aspect_map = row.get("aspect2sentence_indices", {})
        result_aspects = row.get("results_aspect_list_ids", row.get("aspect_list_ids", []))
        if isinstance(aspect_map, dict):
            selected_aspects = (
                result_aspects if isinstance(result_aspects, list) else list(aspect_map)
            )
            for aspect_id in selected_aspects:
                values = aspect_map.get(aspect_id, [])
                if isinstance(values, list):
                    indices.update(value for value in values if isinstance(value, int))
        relevant: list[str] = []
        for index, sentence in enumerate(pool[:MAX_DOCUMENTS_PER_CASE]):
            document_id = f"{case_id}-sent-{index}"
            documents.append(
                _document(
                    document_id,
                    str(sentence),
                    title=f"EvidenceBench {raw_id}",
                    locator=f"sentence-{index}",
                )
            )
            if index in indices:
                relevant.append(document_id)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=_bounded_text(row.get("hypothesis", ""), limit=16_000),
                relevant_ids=tuple(relevant),
                metadata={"source_id": str(raw_id), "paper_id": str(row.get("paper_id", ""))},
            )
        )
    return DatasetBundle(
        "evidencebench",
        split,
        _bounded_cases(cases, max_cases),
        tuple(documents),
        notes=("相关证据按 results_aspect_list_ids 映射到候选句子。",),
    )


def _normalise_choice(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"0", "a"}:
        return "A"
    if text in {"1", "b"}:
        return "B"
    if text in {"2", "c"}:
        return "C"
    if text in {"3", "d"}:
        return "D"
    return text.upper()[:8]


def _medqa(root: Path, split: str, max_cases: int) -> DatasetBundle:
    candidates = [
        root / "medqa-usmle-hf-qa-mirror" / f"phrases_no_exclude_{split}.jsonl",
        root / "medqa-usmle-hf-mirror" / "questions" / f"{split}.jsonl",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise DatasetUnavailable(f"MedQA split 未下载: {split}")
    rows = _read_jsonl(path, max_cases=max_cases)
    cases: list[EvaluationCase] = []
    for index, row in enumerate(rows):
        options = row.get("options", {})
        options = options if isinstance(options, dict) else {}
        normalized_options = {
            _normalise_choice(key): _bounded_text(value, limit=2_000)
            for key, value in options.items()
            if str(key).strip()
        }
        answer = _normalise_choice(row.get("answer_idx", row.get("answer", "")))
        cases.append(
            EvaluationCase(
                case_id=_safe_id(row.get("id", index), prefix="medqa"),
                query=_bounded_text(row.get("question", ""), limit=16_000),
                gold_answer=answer or None,
                answer_type="choice",
                options=normalized_options,
                metadata={
                    "split": split,
                    "meta_info": _bounded_text(row.get("meta_info", ""), limit=80),
                },
            )
        )
    return DatasetBundle(
        "medqa-usmle",
        split,
        _bounded_cases(cases, max_cases),
        (),
        answer_supported=True,
        notes=("当前下载的是 QA-only 镜像，未把 textbook corpus 当作检索证据。",),
    )


def _medmcqa(root: Path, split: str, max_cases: int) -> DatasetBundle:
    path = root / "medmcqa-hf-mirror" / f"{split}.parquet"
    if not path.is_file():
        raise DatasetUnavailable(f"MedMCQA split 未下载: {split}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise DatasetUnavailable(
            "MedMCQA Parquet 评测需要 pyarrow；请安装项目 evaluation 可选依赖。"
        ) from exc
    cases: list[EvaluationCase] = []
    try:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=min(max_cases, 512)):
            for index, row in enumerate(batch.to_pylist()):
                if len(cases) >= max_cases:
                    break
                options = {
                    letter: _bounded_text(row.get(column, ""), limit=2_000)
                    for letter, column in zip("ABCD", ("opa", "opb", "opc", "opd"))
                }
                cases.append(
                    EvaluationCase(
                        case_id=_safe_id(row.get("id", len(cases)), prefix="medmcqa"),
                        query=_bounded_text(row.get("question", ""), limit=16_000),
                        gold_answer=_normalise_choice(row.get("cop", "")) or None,
                        answer_type="choice",
                        options=options,
                        metadata={"subject": _bounded_text(row.get("subject_name", ""), limit=80)},
                    )
                )
            if len(cases) >= max_cases:
                break
    except (OSError, ValueError, RuntimeError) as exc:
        raise DatasetEvaluationError(f"无法读取 MedMCQA Parquet: {path}") from exc
    return DatasetBundle(
        "medmcqa",
        split,
        _bounded_cases(cases, max_cases),
        (),
        answer_supported=True,
        notes=("当前 Parquet 镜像不把 explanation 字段注入知识库，避免答案泄漏。",),
    )


def _faithfulness(root: Path, split: str, max_cases: int) -> DatasetBundle:
    directory = root / "faithfulness-qa-2026" / "data"
    jsonl_path = directory / f"faithfulness_qa_squad_{split}.jsonl"
    if jsonl_path.is_file():
        rows = _read_jsonl(jsonl_path, max_cases=max_cases)
    else:
        zip_path = directory / f"faithfulness_qa_squad_{split}.jsonl.zip"
        rows = _read_jsonl_zip(
            zip_path,
            member=f"faithfulness_qa_squad_{split}.jsonl",
            max_cases=max_cases,
        )
    cases: list[EvaluationCase] = []
    documents: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        case_id = _safe_id(row.get("id", index), prefix="faithfulness")
        original_id = f"{case_id}-original"
        modified_id = f"{case_id}-modified"
        original_doc = _document(
            original_id,
            row.get("original_context", ""),
            title=f"Faithfulness-QA {case_id}",
        )
        modified_doc = _document(
            modified_id,
            row.get("modified_context", ""),
            title=f"Faithfulness-QA {case_id}",
        )
        documents.append(original_doc)
        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=_bounded_text(row.get("question", ""), limit=16_000),
                relevant_ids=(original_id,),
                gold_answer=_bounded_text(row.get("original_answer", ""), limit=4_000) or None,
                answer_type="text",
                metadata={"source": _bounded_text(row.get("source", ""), limit=80)},
                variants=(
                    ("original", (original_doc,)),
                    ("modified", (modified_doc,)),
                ),
                expected_variant_change=(
                    _bounded_text(row.get("original_answer", ""), limit=4_000).strip().lower()
                    != _bounded_text(row.get("faithful_answer", ""), limit=4_000).strip().lower()
                ),
            )
        )
    return DatasetBundle(
        "faithfulness-qa-2026",
        split,
        _bounded_cases(cases, max_cases),
        tuple(documents),
        notes=("原始/修改上下文作为隔离变体，评估证据变化后的响应性。",),
    )


def _evidence_inference(root: Path, split: str, max_cases: int) -> DatasetBundle:
    if split not in {"train", "validation", "test"}:
        raise DatasetEvaluationError("Evidence Inference split 只能是 train、validation 或 test")
    directory = root / "evidence-inference-2"
    prompt_path = directory / "prompts_merged.csv"
    annotation_path = directory / "annotations_merged.csv"
    if not prompt_path.is_file() or not annotation_path.is_file():
        raise DatasetUnavailable("Evidence Inference 2.0 CSV 未下载")
    split_file = directory / "splits" / f"{split}_article_ids.txt"
    if not split_file.is_file() and split == "validation":
        split_file = directory / "splits" / "ev2_validation_article_ids.txt"
    if not split_file.is_file():
        raise DatasetUnavailable(f"Evidence Inference split 文件不存在: {split_file}")
    article_ids = {
        line.strip().replace("PMC", "")
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("PMCID", "") in article_ids:
                grouped[row.get("PromptID", "")].append(row)
    label_by_code = {
        "-1": "significantly decreased",
        "0": "no significant difference",
        "1": "significantly increased",
    }
    cases: list[EvaluationCase] = []
    documents: list[dict[str, Any]] = []
    with prompt_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if len(cases) >= max_cases:
                break
            pmcid = row.get("PMCID", "")
            prompt_id = row.get("PromptID", "")
            if pmcid not in article_ids or prompt_id not in grouped:
                continue
            labels = [
                item.get("Label Code", "")
                for item in grouped[prompt_id]
                if item.get("Valid Label", "").lower() == "true"
            ]
            if not labels:
                continue
            gold_code = Counter(labels).most_common(1)[0][0]
            document_id = f"ei-{_safe_id(pmcid, prefix='pmc')}"
            text_path = directory / "txt_files" / f"PMC{pmcid}.txt"
            if not text_path.is_file():
                continue
            if not any(document.get("id") == document_id for document in documents):
                document_text = text_path.read_text(encoding="utf-8", errors="replace")
                if not document_text.strip():
                    continue
                documents.append(
                    _document(
                        document_id,
                        document_text,
                        title=f"Evidence Inference PMC{pmcid}",
                    )
                )
            query = (
                f"Outcome: {row.get('Outcome', '')}; "
                f"Intervention: {row.get('Intervention', '')}; "
                f"Comparator: {row.get('Comparator', '')}"
            )
            cases.append(
                EvaluationCase(
                    case_id=f"ei-{_safe_id(prompt_id, prefix='prompt')}",
                    query=_bounded_text(query, limit=16_000),
                    relevant_ids=(document_id,),
                    gold_answer=label_by_code.get(gold_code),
                    answer_type="label",
                    metadata={"pmcid": pmcid, "prompt_id": prompt_id},
                )
            )
    return DatasetBundle(
        "evidence-inference-2",
        split,
        _bounded_cases(cases, max_cases),
        tuple(documents),
        answer_supported=True,
        notes=("标签按 Valid Label 标注的多数投票聚合；证据跨度暂用于后续 span adapter。",),
    )


def _ragchecker(root: Path, split: str, max_cases: int) -> DatasetBundle:
    raise DatasetUnavailable(
        "RAGChecker 当前下载内容是 meta-evaluation/reference predictions，缺少统一输入语料；"
        "请先将项目检索输出转换为其官方评测格式，再运行 RAGChecker 原生脚本。"
    )


DATASET_LOADERS: dict[str, Callable[[Path, str, int], DatasetBundle]] = {
    "pubmedqa": _pubmedqa,
    "medmcqa": _medmcqa,
    "medqa-usmle": _medqa,
    "evidence-inference-2": _evidence_inference,
    "evidencebench": _evidencebench,
    "faithfulness-qa-2026": _faithfulness,
    "ragchecker": _ragchecker,
}


def load_dataset_bundle(
    dataset_id: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    max_cases: int = DEFAULT_MAX_CASES,
) -> DatasetBundle:
    """Load one split through a registered adapter."""

    normalized = str(dataset_id).strip().lower()
    loader = DATASET_LOADERS.get(normalized)
    if loader is None:
        if normalized in REFERENCE_ONLY_DATASET_REASONS:
            raise DatasetUnavailable(REFERENCE_ONLY_DATASET_REASONS[normalized])
        raise DatasetEvaluationError(f"不支持的数据集: {dataset_id}")
    try:
        bounded_max_cases = max(1, min(int(max_cases), MAX_CASES))
    except (TypeError, ValueError):
        bounded_max_cases = DEFAULT_MAX_CASES
    return loader(Path(data_root), str(split).strip().lower(), bounded_max_cases)


def _normalize_answer_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _result_text(result: Mapping[str, Any]) -> str:
    claims = result.get("claims", [])
    texts: list[str] = []
    if isinstance(claims, list):
        for claim in claims[:16]:
            if isinstance(claim, Mapping) and isinstance(claim.get("text"), str):
                texts.append(claim["text"])
    report = result.get("report")
    if isinstance(report, Mapping):
        summary = report.get("summary")
        if isinstance(summary, str):
            texts.append(summary)
    return "\n".join(texts)


def _benchmark_request(case: EvaluationCase) -> str:
    """Add a gold-free, machine-readable answer contract for benchmark runs."""

    request = case.query.strip()
    if case.answer_type == "choice" and case.options:
        rendered_options = "\n".join(
            f"{letter}. {_bounded_text(text, limit=2_000)}"
            for letter, text in case.options.items()
        )
        contract = (
            "Benchmark options:\n"
            f"{rendered_options}\n"
            "When the evidence is sufficient, include one cited atomic claim whose text "
            'is exactly "Final answer: X", where X is one of A, B, C, or D. '
            "If evidence is insufficient, abstain rather than guessing."
        )
    elif case.answer_type == "label":
        expected = _normalize_answer_text(case.gold_answer)
        labels = (
            ("yes", "no", "maybe")
            if expected in {"yes", "no", "maybe"}
            else (
                "significantly increased",
                "significantly decreased",
                "no significant difference",
            )
        )
        contract = (
            "When the evidence is sufficient, include one cited atomic claim whose text "
            f'is exactly "Final answer: <label>", using one of: {", ".join(labels)}. '
            "If evidence is insufficient, abstain rather than guessing."
        )
    elif case.answer_type == "text":
        contract = (
            "When the evidence is sufficient, include one cited atomic claim whose text "
            'starts exactly with "Final answer: " followed by the shortest supported answer. '
            "If evidence is insufficient, abstain rather than guessing."
        )
    else:
        return request
    return _bounded_text(f"{request}\n\n{contract}", limit=24_000)


def _extract_prediction(text: str, case: EvaluationCase) -> str | None:
    normalized = _normalize_answer_text(text)
    if not normalized:
        return None
    if case.answer_type == "choice":
        explicit = re.findall(
            r"(?:answer|option|choice|selected|答案|选项)\s*(?:is|:|：)?\s*\(?([abcd])\b",
            text,
            flags=re.IGNORECASE,
        )
        if explicit:
            return explicit[-1].upper()
        for letter, option in case.options.items():
            option_text = _normalize_answer_text(option)
            if option_text and option_text in normalized:
                return letter.upper()
        return None
    if case.answer_type == "label":
        labels = [
            "significantly increased",
            "significantly decreased",
            "no significant difference",
            "yes",
            "no",
            "maybe",
        ]
        explicit_pattern = r"(?:answer|decision|verdict|final|答案|结论)[^\n.!?]{0,80}"
        matches: list[str] = []
        for label in labels:
            if re.search(explicit_pattern + re.escape(label), text, flags=re.IGNORECASE):
                matches.append(label)
        if matches:
            return matches[-1]
        return None
    if case.answer_type == "text":
        matches = re.findall(
            r"(?:final answer|答案)\s*[:：]\s*([^\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        if matches:
            return _normalize_answer_text(matches[-1]) or None
    return None


def _answer_score(case: EvaluationCase, result_text: str) -> dict[str, Any]:
    if not case.gold_answer or case.answer_type not in {"choice", "label", "text"}:
        return {
            "scorable": False,
            "evaluated": False,
            "predicted": None,
            "correct": None,
        }
    prediction = _extract_prediction(result_text, case)
    expected = _normalize_answer_text(case.gold_answer)
    return {
        "scorable": True,
        "evaluated": prediction is not None,
        "predicted": prediction,
        "expected": expected,
        "correct": bool(prediction and _normalize_answer_text(prediction) == expected),
    }


def _verify_benchmark_answer_provenance(
    payload: Any,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a benchmark answer to descend from already supported claims.

    The benchmark label is generated after the normal workflow because it is a
    machine-readable adapter concern.  This gate closes the provenance gap by
    requiring exactly one final-answer claim, valid evidence references, and a
    reference overlap with claims that the workflow verifier marked supported.
    It deliberately does not infer medical truth from the label.
    """

    issues: list[str] = []
    raw_claims = payload.get("claims", []) if isinstance(payload, Mapping) else []
    claims = [item for item in raw_claims if isinstance(item, Mapping)]
    final_claims = [
        item
        for item in claims
        if isinstance(item.get("text"), str)
        and item["text"].strip().lower().startswith("final answer:")
    ]
    if len(final_claims) != 1:
        issues.append("FINAL_ANSWER_CLAIM_COUNT")
        return {
            "passed": False,
            "issues": issues,
            "final_claim_count": len(final_claims),
            "final_refs": [],
            "supported_source_claims": [],
        }

    final_claim = final_claims[0]
    raw_refs = final_claim.get("refs", [])
    refs = [ref for ref in raw_refs if isinstance(ref, str) and ref]
    if not refs:
        issues.append("FINAL_ANSWER_NO_REF")

    evidence_items = result.get("evidence", [])
    evidence_ids = {
        str(item.get("id"))
        for item in evidence_items
        if isinstance(item, Mapping) and item.get("id")
    }
    invalid_refs = sorted(set(refs) - evidence_ids)
    if invalid_refs:
        issues.append("FINAL_ANSWER_BAD_REF")

    workflow_claims = result.get("claims", [])
    supported_source_claims: list[str] = []
    supported_refs: set[str] = set()
    for claim in workflow_claims if isinstance(workflow_claims, list) else []:
        if not isinstance(claim, Mapping) or claim.get("status") != "supported":
            continue
        claim_id = str(claim.get("id", ""))
        claim_refs = {
            str(ref)
            for ref in claim.get("refs", [])
            if isinstance(ref, str) and ref
        }
        if claim_id and claim_refs.intersection(refs):
            supported_source_claims.append(claim_id)
            supported_refs.update(claim_refs)
    if not supported_source_claims:
        issues.append("FINAL_ANSWER_UNSUPPORTED_SOURCE")
    if set(refs) - supported_refs:
        issues.append("FINAL_ANSWER_REF_NOT_VERIFIED")

    return {
        "passed": not issues,
        "issues": issues,
        "final_claim_count": len(final_claims),
        "final_refs": refs,
        "supported_source_claims": supported_source_claims,
    }


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def _merge_model_usage(
    base_usage: Mapping[str, Any] | None,
    metrics: list[dict[str, Any]],
) -> dict[str, int]:
    extra_usage = summarize_model_metrics(metrics) if metrics else {}
    base = base_usage if isinstance(base_usage, Mapping) else {}
    return {
        key: int(base.get(key, 0)) + int(extra_usage.get(key, 0))
        for key in (
            "calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "latency_ms",
        )
    }


def _build_agent(
    knowledge_base: JsonKnowledgeBase,
    *,
    model_source: str,
    model_profile: str | None,
    max_repair_rounds: int,
) -> tuple[Any, ModelCallCounter]:
    counter = ModelCallCounter()
    if model_source == "demo":
        if model_profile and model_profile != "demo":
            raise DatasetEvaluationError("demo 模型来源只提供 profile=demo")
        model = CountingModelAdapter(DemoModelAdapter(), counter)
        agent = create_agent(
            model_profiles={"demo": model},
            default_model_profile="demo",
            knowledge_base=knowledge_base,
            max_repair_rounds=max_repair_rounds,
        )
        return agent, counter
    agent = create_agent_from_environment(
        knowledge_base=knowledge_base,
        max_repair_rounds=max_repair_rounds,
    )
    wrapped_profiles = {
        profile_id: CountingModelAdapter(model, counter)
        for profile_id, model in agent.model_profiles.items()
    }
    agent.model_profiles = wrapped_profiles
    if model_profile and model_profile not in wrapped_profiles:
        raise DatasetEvaluationError(f"模型 profile 不存在: {model_profile}")
    return agent, counter


def _run_agent_case(
    agent: Any,
    counter: ModelCallCounter,
    case: EvaluationCase,
    *,
    model_profile: str | None,
) -> tuple[dict[str, Any], int, dict[str, int]]:
    before = counter.calls
    before_stages = Counter(counter.by_stage)
    started = time.perf_counter()
    try:
        result = agent.run(
            request=_benchmark_request(case),
            patient_record="",
            allow_general=True,
            model_profile=model_profile,
            archive_result=False,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        usage = result.get("run", {}).get("model_usage", {})
        text = _result_text(result)
        answer = _answer_score(case, text)
        answer_stage = "workflow"
        answer_verification: dict[str, Any] = {
            "passed": False,
            "issues": ["FINAL_ANSWER_NOT_VERIFIED"],
            "final_claim_count": 0,
            "final_refs": [],
            "supported_source_claims": [],
        }
        if answer.get("scorable"):
            model_id = model_profile or agent.default_model_profile
            model = agent.model_profiles[model_id]
            labels: tuple[str, ...]
            if case.answer_type == "choice":
                labels = tuple(str(label).upper() for label in case.options)
            elif case.answer_type == "label":
                labels = (
                    ("yes", "no", "maybe")
                    if _normalize_answer_text(case.gold_answer) in {"yes", "no", "maybe"}
                    else (
                        "significantly increased",
                        "significantly decreased",
                        "no significant difference",
                    )
                )
            else:
                labels = ()
            verified_facts = [
                {"text": _bounded_text(claim.get("text", ""), limit=1_200), "ref": ref}
                for claim in result.get("claims", [])[:12]
                if isinstance(claim, Mapping)
                and claim.get("status") == "supported"
                and isinstance(claim.get("refs"), list)
                and claim.get("refs")
                and claim.get("text")
                for ref in claim.get("refs", [])[:4]
                if isinstance(ref, str) and ref
            ]
            benchmark_text = case.answer_type == "text"
            if (labels or benchmark_text) and verified_facts:
                try:
                    synthesis_task = {
                        "id": 0,
                        "goal": "生成一个机器可解析、由已验证事实支持的 benchmark 最终答案。",
                        "deps": [],
                        "patient_grounding_required": False,
                    }
                    if labels:
                        synthesis_task["benchmark_answer_labels"] = list(labels)
                    if benchmark_text:
                        synthesis_task["benchmark_answer_text"] = True
                    payload = model.synthesize(
                        task=synthesis_task,
                        request=case.query,
                        facts=verified_facts,
                    )
                    answer_text = "\n".join(
                        str(claim.get("text", ""))
                        for claim in payload.get("claims", [])[:4]
                        if isinstance(claim, Mapping)
                    ) if isinstance(payload, Mapping) else ""
                    answer = _answer_score(case, answer_text)
                    answer_verification = _verify_benchmark_answer_provenance(
                        payload, result
                    )
                    answer["verification"] = answer_verification
                    if answer_verification["passed"]:
                        answer_stage = "benchmark_synthesis_verified"
                    else:
                        answer["evaluated"] = False
                        answer["correct"] = False
                        answer_stage = "benchmark_synthesis_unverified"
                except Exception as exc:  # noqa: BLE001 - preserve completed workflow
                    answer_verification = {
                        "passed": False,
                        "issues": [f"SYNTHESIS_ERROR:{type(exc).__name__}"],
                        "final_claim_count": 0,
                        "final_refs": [],
                        "supported_source_claims": [],
                    }
                    answer["verification"] = answer_verification
                    answer["evaluated"] = False
                    answer["correct"] = False
                    answer_stage = f"benchmark_synthesis_error:{type(exc).__name__}"
                finally:
                    usage = _merge_model_usage(usage, model.drain_call_metrics())
            elif labels or benchmark_text:
                answer_verification["issues"] = ["NO_VERIFIED_WORKFLOW_FACTS"]
                answer["verification"] = answer_verification
                answer["evaluated"] = False
                answer["correct"] = False
                answer_stage = "benchmark_synthesis_skipped"
        case_result = {
            "id": case.case_id,
            "status": result.get("status", "unknown"),
            "latency_ms": elapsed_ms,
            "answer": answer,
            "answer_stage": answer_stage,
            "answer_verification": answer_verification,
            "quality": result.get("run", {}).get("quality", {}),
            "evidence_count": (
                len(result.get("evidence", []))
                if isinstance(result.get("evidence"), list)
                else 0
            ),
            "claims": (
                len(result.get("claims", []))
                if isinstance(result.get("claims"), list)
                else 0
            ),
            "decision": result.get("run", {}).get("decision", {}).get("outcome", "")
            if isinstance(result.get("run", {}).get("decision", {}), Mapping)
            else "",
            "evidence_ids": [
                str(item.get("id"))
                for item in result.get("evidence", [])[:64]
                if isinstance(item, Mapping) and item.get("id")
            ],
            "evidence_keys": [
                str(
                    item.get("document_id")
                    or item.get("content_hash")
                    or item.get("id")
                )
                for item in result.get("evidence", [])[:64]
                if isinstance(item, Mapping)
                and (
                    item.get("document_id")
                    or item.get("content_hash")
                    or item.get("id")
                )
            ],
            "claim_refs": [
                ref
                for claim in result.get("claims", [])[:16]
                if isinstance(claim, Mapping)
                for ref in claim.get("refs", [])[:16]
                if isinstance(ref, str)
            ],
            "model_usage": usage if isinstance(usage, Mapping) else {},
            "retrieval_usage": (
                result.get("run", {}).get("retrieval_usage", {})
                if isinstance(result.get("run", {}).get("retrieval_usage", {}), Mapping)
                else {}
            ),
        }
        return case_result, counter.calls - before, {
            stage: count - before_stages.get(stage, 0)
            for stage, count in counter.by_stage.items()
            if count > before_stages.get(stage, 0)
        }
    except Exception as exc:  # noqa: BLE001 - isolate one benchmark case
        model_id = model_profile or agent.default_model_profile
        model = agent.model_profiles[model_id]
        usage = _merge_model_usage({}, model.drain_call_metrics())
        return {
            "id": case.case_id,
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": type(exc).__name__,
            "answer": {
                "scorable": bool(case.gold_answer)
                and case.answer_type in {"choice", "label", "text"},
                "evaluated": False,
                "predicted": None,
                "correct": False,
                "verification": {
                    "passed": False,
                    "issues": ["CASE_ERROR"],
                    "final_claim_count": 0,
                    "final_refs": [],
                    "supported_source_claims": [],
                },
            },
            "answer_verification": {
                "passed": False,
                "issues": ["CASE_ERROR"],
                "final_claim_count": 0,
                "final_refs": [],
                "supported_source_claims": [],
            },
            "model_usage": usage,
        }, counter.calls - before, {
            stage: count - before_stages.get(stage, 0)
            for stage, count in counter.by_stage.items()
            if count > before_stages.get(stage, 0)
        }


def _aggregate_agent_results(
    results: list[dict[str, Any]],
    *,
    observed_calls: int,
    stage_counts: Counter[str],
) -> dict[str, Any]:
    answer_rows = [
        row.get("answer", {})
        for row in results
        if isinstance(row.get("answer"), Mapping)
    ]
    evaluated_answers = [row for row in answer_rows if row.get("evaluated")]
    scorable_answers = [row for row in answer_rows if row.get("scorable")]
    correct_answers = [row for row in evaluated_answers if row.get("correct")]
    verified_answers = [
        row
        for row in answer_rows
        if isinstance(row.get("verification"), Mapping)
        and row["verification"].get("passed") is True
    ]
    quality_keys = (
        "citation_coverage",
        "citation_precision",
        "support_edge_coverage",
        "dual_support_coverage",
        "unresolved_rate",
        "contradiction_rate",
        "insufficient_rate",
    )
    quality: dict[str, float] = {}
    for key in quality_keys:
        values = [
            float(row["quality"][key])
            for row in results
            if isinstance(row.get("quality"), Mapping)
            and isinstance(row["quality"].get(key), (int, float))
        ]
        if values:
            quality[key] = _mean(values)
    usage_rows = [
        row.get("model_usage", {})
        for row in results
        if isinstance(row.get("model_usage"), Mapping)
    ]
    retrieval_rows = [
        row.get("retrieval_usage", {})
        for row in results
        if isinstance(row.get("retrieval_usage"), Mapping)
    ]
    reported = {
        "calls": sum(
            int(row.get("calls", 0))
            for row in usage_rows
            if isinstance(row.get("calls", 0), int)
        ),
        "input_tokens": sum(
            int(row.get("input_tokens", 0))
            for row in usage_rows
            if isinstance(row.get("input_tokens", 0), int)
        ),
        "output_tokens": sum(
            int(row.get("output_tokens", 0))
            for row in usage_rows
            if isinstance(row.get("output_tokens", 0), int)
        ),
        "total_tokens": sum(
            int(row.get("total_tokens", 0))
            for row in usage_rows
            if isinstance(row.get("total_tokens", 0), int)
        ),
        "reasoning_tokens": sum(
            int(row.get("reasoning_tokens", 0))
            for row in usage_rows
            if isinstance(row.get("reasoning_tokens", 0), int)
        ),
        "latency_ms": sum(
            int(row.get("latency_ms", 0))
            for row in usage_rows
            if isinstance(row.get("latency_ms", 0), int)
        ),
    }
    retrieval_usage: dict[str, dict[str, int]] = {}
    for raw_usage in retrieval_rows:
        for component, raw_values in raw_usage.items():
            if not isinstance(raw_values, Mapping):
                continue
            bucket = retrieval_usage.setdefault(str(component), {})
            for key, value in raw_values.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    bucket[str(key)] = bucket.get(str(key), 0) + value
    status_counts = Counter(str(row.get("status", "unknown")) for row in results)
    redacted_fields = {
        "evidence_ids",
        "evidence_keys",
        "claim_refs",
        "model_usage",
        "retrieval_usage",
    }
    return {
        "cases": len(results),
        "status_counts": dict(status_counts),
        "answer": {
            "scorable_cases": len(scorable_answers),
            "evaluated_cases": len(evaluated_answers),
            "correct_cases": len(correct_answers),
            "coverage": _mean(
                [1.0] * len(evaluated_answers)
                + [0.0] * (len(scorable_answers) - len(evaluated_answers))
            ),
            "accuracy": _mean([float(row.get("correct", False)) for row in evaluated_answers]),
            "overall_accuracy": _mean(
                [float(row.get("correct", False)) for row in scorable_answers]
            ),
            "verified_cases": len(verified_answers),
            "unverified_cases": sum(
                1
                for row in scorable_answers
                if not (
                    isinstance(row.get("verification"), Mapping)
                    and row["verification"].get("passed") is True
                )
            ),
            "unparseable_cases": sum(
                1
                for row in answer_rows
                if row.get("scorable") and not row.get("evaluated")
            ),
        },
        "quality": quality,
        "cost": {
            "observed_model_calls": observed_calls,
            "observed_calls_by_stage": dict(stage_counts),
            "provider_reported_usage": reported,
            "telemetry_complete": reported["calls"] >= observed_calls,
            "unreported_calls": max(0, observed_calls - reported["calls"]),
            "retrieval_usage": retrieval_usage,
            "wall_latency_ms": _mean([float(row.get("latency_ms", 0)) for row in results]),
        },
        "cases_detail": [
            {
                key: value
                for key, value in row.items()
                if key not in redacted_fields
            }
            for row in results
        ],
    }


def _evaluate_bundle(
    bundle: DatasetBundle,
    *,
    mode: str,
    model_source: str,
    model_profile: str | None,
    top_k: int,
    max_repair_rounds: int,
    retrieval_backend: str,
    data_root: str | Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset_id": bundle.dataset_id,
        "split": bundle.split,
        "status": "completed",
        "cases_loaded": len(bundle.cases),
        "retrieval_backend": retrieval_backend,
        "notes": list(bundle.notes),
    }
    kb_source = JsonKnowledgeBase(list(bundle.documents))
    kb = _build_evaluation_knowledge_base(
        kb_source,
        retrieval_backend=retrieval_backend,
        data_root=data_root,
        dataset_id=bundle.dataset_id,
        split=bundle.split,
    )
    retrieval_cases = [
        {"id": case.case_id, "query": case.query, "relevant_ids": list(case.relevant_ids)}
        for case in bundle.cases
        if case.query and case.relevant_ids
    ]
    if mode in {"retrieval", "both"}:
        if retrieval_cases and bundle.documents:
            retrieval_report = evaluate_retrieval_cases(
                retrieval_cases,
                lambda query: kb.search_many([query], limit=top_k, max_per_document=top_k),
                ks=(1, 3, top_k),
            )
            drain_embedding_usage = getattr(kb, "drain_embedding_usage", None)
            if callable(drain_embedding_usage):
                usage = drain_embedding_usage()
                if isinstance(usage, Mapping):
                    retrieval_report["embedding_usage"] = {
                        str(key): int(value)
                        for key, value in usage.items()
                        if isinstance(key, str)
                        and isinstance(value, int)
                        and value >= 0
                    }
            result["retrieval"] = retrieval_report
        else:
            result["retrieval"] = {"status": "skipped", "reason": "数据集没有可用的文档级 gold 证据。"}

    if mode not in {"agent", "both"}:
        return result
    agent, counter = _build_agent(
        kb,
        model_source=model_source,
        model_profile=model_profile,
        max_repair_rounds=max_repair_rounds,
    )
    case_results: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    observed_calls = 0
    for case in bundle.cases:
        row, calls, stages = _run_agent_case(agent, counter, case, model_profile=model_profile)
        case_results.append(row)
        observed_calls += calls
        stage_counts.update(stages)
    result["agent"] = _aggregate_agent_results(
        case_results,
        observed_calls=observed_calls,
        stage_counts=stage_counts,
    )
    if result["agent"]["status_counts"].get("error", 0):
        result["status"] = "completed_with_errors"

    counterfactual_cases: list[dict[str, Any]] = []
    counterfactual_skipped = 0
    counterfactual_results: list[dict[str, Any]] = []
    counterfactual_calls = 0
    counterfactual_stages: Counter[str] = Counter()
    for case in bundle.cases:
        if len(case.variants) < 2:
            continue
        snapshots: list[dict[str, Any]] = []
        for variant_id, variant_documents in case.variants[:2]:
            variant_source = JsonKnowledgeBase(list(variant_documents))
            variant_kb = _build_evaluation_knowledge_base(
                variant_source,
                retrieval_backend=retrieval_backend,
                data_root=data_root,
                dataset_id=bundle.dataset_id,
                split=bundle.split,
                suffix=f"{case.case_id}-{variant_id}",
            )
            variant_agent, variant_counter = _build_agent(
                variant_kb,
                model_source=model_source,
                model_profile=model_profile,
                max_repair_rounds=max_repair_rounds,
            )
            variant_result, variant_calls, variant_stages = _run_agent_case(
                variant_agent,
                variant_counter,
                case,
                model_profile=model_profile,
            )
            counterfactual_results.append(variant_result)
            counterfactual_calls += variant_calls
            counterfactual_stages.update(variant_stages)
            if variant_result.get("status") == "error":
                counterfactual_skipped += 1
                snapshots = []
                break
            answer = variant_result.get("answer", {})
            predicted = answer.get("predicted") if isinstance(answer, Mapping) else None
            decision = str(
                predicted
                or variant_result.get("decision")
                or variant_result.get("status", "")
            )
            snapshots.append(
                {
                    "variant": variant_id,
                    "decision": decision,
                    "evidence_ids": variant_result.get("evidence_ids", []),
                    "evidence_keys": variant_result.get("evidence_keys", []),
                    "claims": [{"refs": variant_result.get("claim_refs", [])}],
                }
            )
        if len(snapshots) == 2:
            counterfactual_cases.append(
                {
                    "id": case.case_id,
                    "baseline": snapshots[0],
                    "counterfactual": snapshots[1],
                    "expected": {
                        "decision_should_change": bool(case.expected_variant_change),
                        "evidence_should_change": True,
                    },
                }
            )
    if counterfactual_cases:
        result["counterfactual"] = evaluate_counterfactual_cases(counterfactual_cases)
    if counterfactual_results:
        counterfactual_summary = _aggregate_agent_results(
            counterfactual_results,
            observed_calls=counterfactual_calls,
            stage_counts=counterfactual_stages,
        )
        result["counterfactual_cost"] = {
            "cases": counterfactual_summary["cases"],
            "status_counts": counterfactual_summary["status_counts"],
            **counterfactual_summary["cost"],
        }
    if counterfactual_skipped:
        result["counterfactual_skipped_cases"] = counterfactual_skipped
        result["status"] = "completed_with_errors"
    return result


def run_dataset_evaluation(
    *,
    datasets: Sequence[str],
    data_root: str | Path = DEFAULT_DATA_ROOT,
    split: str = "test",
    max_cases: int = DEFAULT_MAX_CASES,
    mode: str = "both",
    model_source: str = "demo",
    model_profile: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_repair_rounds: int = 0,
    retrieval_backend: str = "lexical",
    case_offset: int = 0,
    case_limit: int | None = None,
) -> dict[str, Any]:
    """Run bounded evaluation and return a redacted JSON-safe report."""

    if mode not in {"retrieval", "agent", "both"}:
        raise DatasetEvaluationError("mode 必须是 retrieval、agent 或 both")
    if model_source not in {"demo", "environment"}:
        raise DatasetEvaluationError("model_source 必须是 demo 或 environment")
    normalized_backend = str(retrieval_backend).strip().lower()
    if normalized_backend not in SUPPORTED_RETRIEVAL_BACKENDS:
        raise DatasetEvaluationError(
            "retrieval_backend 必须是 lexical 或 faiss"
        )
    try:
        bounded_top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise DatasetEvaluationError("top_k 必须是整数") from exc
    if not 1 <= bounded_top_k <= 128:
        raise DatasetEvaluationError("top_k 必须在 1 到 128 之间")
    try:
        bounded_repair_rounds = max(0, int(max_repair_rounds))
    except (TypeError, ValueError) as exc:
        raise DatasetEvaluationError("max_repair_rounds 必须是整数") from exc
    try:
        bounded_max_cases = max(1, min(int(max_cases), MAX_CASES))
    except (TypeError, ValueError):
        bounded_max_cases = DEFAULT_MAX_CASES
    try:
        bounded_case_offset = max(0, int(case_offset))
    except (TypeError, ValueError) as exc:
        raise DatasetEvaluationError("case_offset 必须是非负整数") from exc
    if case_limit is None:
        bounded_case_limit = None
    else:
        try:
            bounded_case_limit = int(case_limit)
        except (TypeError, ValueError) as exc:
            raise DatasetEvaluationError("case_limit 必须是正整数") from exc
        if bounded_case_limit < 1:
            raise DatasetEvaluationError("case_limit 必须是正整数")
    dataset_values = [datasets] if isinstance(datasets, str) else datasets
    expanded_datasets: list[str] = []
    for dataset in dataset_values:
        normalized_dataset = str(dataset).strip().lower()
        if normalized_dataset == "all":
            expanded_datasets.extend(ALL_DATASETS)
        else:
            expanded_datasets.append(normalized_dataset)
    selected = list(
        dict.fromkeys(
            dataset for dataset in expanded_datasets if dataset
        )
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "config": {
            "datasets": selected,
            "split": split,
            "max_cases": bounded_max_cases,
            "case_offset": bounded_case_offset,
            "case_limit": bounded_case_limit,
            "mode": mode,
            "model_source": model_source,
            "model_profile": model_profile or "default",
            "top_k": bounded_top_k,
            "max_repair_rounds": bounded_repair_rounds,
            "retrieval_backend": normalized_backend,
            "data_root": str(Path(data_root).resolve()),
        },
        "results": [],
    }
    for dataset_id in selected:
        try:
            bundle = load_dataset_bundle(
                dataset_id,
                data_root=data_root,
                split=split,
                max_cases=bounded_max_cases,
            )
            if bounded_case_offset or bounded_case_limit is not None:
                case_end = (
                    None
                    if bounded_case_limit is None
                    else bounded_case_offset + bounded_case_limit
                )
                bundle = DatasetBundle(
                    dataset_id=bundle.dataset_id,
                    split=bundle.split,
                    cases=bundle.cases[bounded_case_offset:case_end],
                    documents=bundle.documents,
                    answer_supported=bundle.answer_supported,
                    notes=bundle.notes
                    + (
                        "分批执行范围: "
                        f"offset={bounded_case_offset}, "
                        f"limit={bounded_case_limit or 'remaining'}；检索语料保持完整。",
                    ),
                )
                if not bundle.cases:
                    raise DatasetEvaluationError("分批范围没有可评测案例")
            result = _evaluate_bundle(
                bundle,
                mode=mode,
                model_source=model_source,
                model_profile=model_profile,
                top_k=bounded_top_k,
                max_repair_rounds=bounded_repair_rounds,
                retrieval_backend=normalized_backend,
                data_root=data_root,
            )
        except DatasetUnavailable as exc:
            result = {
                "dataset_id": dataset_id,
                "split": split,
                "status": "skipped",
                "reason": str(exc)[:300],
            }
        except DatasetEvaluationError as exc:
            result = {
                "dataset_id": dataset_id,
                "split": split,
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc)[:300],
            }
        except Exception as exc:  # noqa: BLE001 - isolate one dataset/config failure
            result = {
                "dataset_id": dataset_id,
                "split": split,
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc)[:300] or "评测运行失败",
            }
        report["results"].append(result)
    counts = Counter(str(item.get("status", "unknown")) for item in report["results"])
    report["summary"] = {
        "datasets": len(report["results"]),
        "status_counts": dict(counts),
        "completed": counts.get("completed", 0),
        "completed_with_errors": counts.get("completed_with_errors", 0),
        "skipped": counts.get("skipped", 0),
        "errors": counts.get("error", 0) + counts.get("completed_with_errors", 0),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行已下载医学评测数据集的统一评测")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help=(
            f"数据集 ID，可重复；默认运行: {', '.join(SUPPORTED_DATASETS)}；"
            "传 all 可包含仅登记/需授权的数据集"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--mode", choices=("retrieval", "agent", "both"), default="both")
    parser.add_argument("--model-source", choices=("demo", "environment"), default="demo")
    parser.add_argument("--model-profile")
    parser.add_argument(
        "--retrieval-backend",
        choices=SUPPORTED_RETRIEVAL_BACKENDS,
        default="lexical",
        help="评测检索后端；faiss 会复用模型配置中的 embedding 和索引策略",
    )
    parser.add_argument("--max-repair-rounds", type=int, default=0)
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="跳过已加载案例的数量；与 --case-limit 配合用于可恢复分批执行",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="本批最多执行的案例数；检索语料仍由 --max-cases 决定",
    )
    parser.add_argument("--out", type=Path, help="输出 JSON 报告路径")
    parser.add_argument("--quiet", action="store_true", help="不向标准输出打印完整 JSON")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_datasets:
        print("\n".join(ALL_DATASETS))
        return 0
    if args.max_cases < 1 or args.max_cases > MAX_CASES:
        parser.error(f"--max-cases 必须在 1 到 {MAX_CASES} 之间")
    if args.top_k < 1 or args.top_k > 128:
        parser.error("--top-k 必须在 1 到 128 之间")
    if args.case_offset < 0:
        parser.error("--case-offset 必须是非负整数")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit 必须是正整数")
    datasets = args.datasets or list(SUPPORTED_DATASETS)
    report = run_dataset_evaluation(
        datasets=datasets,
        data_root=args.data_root,
        split=args.split,
        max_cases=args.max_cases,
        mode=args.mode,
        model_source=args.model_source,
        model_profile=args.model_profile,
        top_k=args.top_k,
        max_repair_rounds=max(0, args.max_repair_rounds),
        retrieval_backend=args.retrieval_backend,
        case_offset=args.case_offset,
        case_limit=args.case_limit,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered, end="")
    if args.fail_on_error and report["summary"]["errors"]:
        return 1
    return 0


__all__ = [
    "ALL_DATASETS",
    "DATASET_LOADERS",
    "DatasetBundle",
    "DatasetEvaluationError",
    "DatasetUnavailable",
    "EvaluationCase",
    "REFERENCE_ONLY_DATASET_REASONS",
    "SUPPORTED_RETRIEVAL_BACKENDS",
    "SUPPORTED_DATASETS",
    "load_dataset_bundle",
    "run_dataset_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
