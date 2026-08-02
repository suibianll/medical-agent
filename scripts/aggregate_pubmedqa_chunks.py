"""Aggregate resumable PubMedQA agent-evaluation chunk reports.

The chunk reports intentionally contain only redacted case-level telemetry.  This
script keeps the aggregation reproducible without loading prompts or provider
responses, and refuses to produce a full-scope report when a five-case window is
missing or duplicated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


CHUNK_NAME = re.compile(r"chunk-(\d+)-(\d+)\.json$")
QUALITY_KEYS = (
    "citation_coverage",
    "citation_precision",
    "support_edge_coverage",
    "dual_support_coverage",
    "unresolved_rate",
    "contradiction_rate",
    "insufficient_rate",
)
USAGE_KEYS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "latency_ms",
)
RETRIEVAL_SUM_KEYS = (
    "attempts",
    "external_calls",
    "cache_hits",
    "skipped_budget",
    "skipped_min_candidates",
    "failures",
    "candidate_count",
    "provider_calls",
    "requested_texts",
    "provider_texts",
    "batches",
    "latency_ms",
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return value
    return None


def _sum_number(values: list[Any]) -> int | float:
    valid = [value for value in values if _number(value) is not None]
    if not valid:
        return 0
    total = sum(valid)
    return int(total) if all(isinstance(value, int) for value in valid) else total


def _round(value: float | int | None, digits: int = 6) -> float | int | None:
    if value is None:
        return None
    result = round(float(value), digits)
    return int(result) if result.is_integer() else result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _aggregate_usage(chunks: list[Mapping[str, Any]]) -> dict[str, Any]:
    observed = 0
    stages: Counter[str] = Counter()
    reported: dict[str, Any] = {}
    telemetry_complete = 0
    unreported = 0
    chunk_wall: list[float] = []
    retrieval: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        agent = chunk.get("results", [{}])[0].get("agent", {})
        cost = agent.get("cost", {}) if isinstance(agent, Mapping) else {}
        observed += _as_int(cost.get("observed_model_calls"))
        raw_stages = cost.get("observed_calls_by_stage", {})
        if isinstance(raw_stages, Mapping):
            stages.update({str(key): _as_int(value) for key, value in raw_stages.items()})
        raw_reported = cost.get("provider_reported_usage", {})
        if isinstance(raw_reported, Mapping):
            for key in USAGE_KEYS:
                reported[key] = _sum_number([reported.get(key, 0), raw_reported.get(key, 0)])
        if cost.get("telemetry_complete") is True:
            telemetry_complete += 1
        unreported += _as_int(cost.get("unreported_calls"))
        wall = _number(cost.get("wall_latency_ms"))
        if wall is not None:
            chunk_wall.append(float(wall))
        raw_retrieval = cost.get("retrieval_usage", {})
        if not isinstance(raw_retrieval, Mapping):
            continue
        for component, values in raw_retrieval.items():
            if not isinstance(values, Mapping):
                continue
            target = retrieval.setdefault(str(component), {})
            for key in RETRIEVAL_SUM_KEYS:
                value = _number(values.get(key))
                if value is not None:
                    target[key] = _sum_number([target.get(key, 0), value])
            for key in ("cache_size", "max_calls_per_run"):
                value = _number(values.get(key))
                if value is not None:
                    target[key] = max(_as_int(target.get(key)), _as_int(value))
    reported["calls"] = _as_int(reported.get("calls"))
    reported["input_tokens"] = _as_int(reported.get("input_tokens"))
    reported["output_tokens"] = _as_int(reported.get("output_tokens"))
    reported["total_tokens"] = _as_int(reported.get("total_tokens"))
    reported["reasoning_tokens"] = _as_int(reported.get("reasoning_tokens"))
    reported["latency_ms"] = _as_int(reported.get("latency_ms"))
    return {
        "observed_model_calls": observed,
        "observed_calls_by_stage": dict(stages),
        "provider_reported_usage": reported,
        "telemetry_complete": telemetry_complete == len(chunks),
        "telemetry_complete_chunks": telemetry_complete,
        "telemetry_total_chunks": len(chunks),
        "unreported_calls": unreported,
        "retrieval_usage": retrieval,
        "chunk_wall_latency_ms_sum": _round(sum(chunk_wall)),
        "chunk_wall_latency_ms_max": _round(max(chunk_wall) if chunk_wall else None),
    }


def aggregate(chunk_root: Path, *, expected_cases: int = 500) -> dict[str, Any]:
    files: list[tuple[int, int, Path, dict[str, Any]]] = []
    for path in sorted(chunk_root.glob("chunk-*.json")):
        match = CHUNK_NAME.fullmatch(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        files.append((int(match.group(1)), int(match.group(2)), path, payload))
    files.sort(key=lambda item: item[0])
    if not files:
        raise RuntimeError(f"未找到 chunk JSON: {chunk_root}")

    expected_offsets = set(range(0, expected_cases, 5))
    offsets = [item[0] for item in files]
    missing = sorted(expected_offsets - set(offsets))
    duplicates = sorted(offset for offset, count in Counter(offsets).items() if count > 1)
    if missing or duplicates:
        raise RuntimeError(f"窗口不完整: missing={missing}, duplicates={duplicates}")

    chunks = [item[3] for item in files]
    cases: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    answer_stage_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    for _, _, _, payload in files:
        result = payload.get("results", [{}])[0]
        agent = result.get("agent", {}) if isinstance(result, Mapping) else {}
        details = agent.get("cases_detail", []) if isinstance(agent, Mapping) else []
        if not isinstance(details, list):
            continue
        for row in details:
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            cases.append(item)
            status = str(item.get("status", "unknown"))
            status_counts[status] += 1
            answer_stage = item.get("answer_stage")
            if answer_stage:
                answer_stage_counts[str(answer_stage)] += 1
            error_type = item.get("error_type")
            if error_type:
                error_type_counts[str(error_type)] += 1
    if len(cases) != expected_cases:
        raise RuntimeError(f"案例数量不完整: expected={expected_cases}, actual={len(cases)}")

    answer_rows = [row.get("answer", {}) for row in cases if isinstance(row.get("answer"), Mapping)]
    scorable = [row for row in answer_rows if row.get("scorable") is True]
    evaluated = [row for row in scorable if row.get("evaluated") is True]
    correct = [row for row in evaluated if row.get("correct") is True]
    answer_summary = {
        "scorable_cases": len(scorable),
        "evaluated_cases": len(evaluated),
        "correct_cases": len(correct),
        "coverage": _round(len(evaluated) / len(scorable) if scorable else None),
        "selective_accuracy": _round(len(correct) / len(evaluated) if evaluated else None),
        "overall_accuracy": _round(len(correct) / len(scorable) if scorable else None),
        "unparseable_cases": len(scorable) - len(evaluated),
    }

    confusion: Counter[str] = Counter()
    for row in evaluated:
        expected = str(row.get("expected", "")).lower()
        predicted = str(row.get("predicted", "")).lower()
        confusion[f"{expected}->{predicted}"] += 1

    by_expected: dict[str, dict[str, Any]] = {}
    expected_labels = sorted(
        {
            str(row.get("expected", "")).lower()
            for row in scorable
            if str(row.get("expected", "")).strip()
        }
    )
    for label in expected_labels:
        label_rows = [row for row in scorable if str(row.get("expected", "")).lower() == label]
        label_evaluated = [row for row in label_rows if row.get("evaluated") is True]
        label_correct = [row for row in label_evaluated if row.get("correct") is True]
        by_expected[label] = {
            "scorable_cases": len(label_rows),
            "evaluated_cases": len(label_evaluated),
            "correct_cases": len(label_correct),
            "coverage": _round(len(label_evaluated) / len(label_rows) if label_rows else None),
            "selective_accuracy": _round(
                len(label_correct) / len(label_evaluated) if label_evaluated else None
            ),
            "overall_accuracy": _round(len(label_correct) / len(label_rows) if label_rows else None),
        }

    quality_rows = [row.get("quality", {}) for row in cases if isinstance(row.get("quality"), Mapping)]
    quality_mean: dict[str, float | None] = {}
    for key in QUALITY_KEYS:
        values = [float(row[key]) for row in quality_rows if _number(row.get(key)) is not None]
        quality_mean[key] = _round(_mean(values))
    quality_totals = {
        "claims": sum(_as_int(row.get("claims")) for row in quality_rows),
        "cited_claims": sum(_as_int(row.get("cited_claims")) for row in quality_rows),
        "dual_support_claims": sum(_as_int(row.get("dual_support_claims")) for row in quality_rows),
        "unresolved_claims": sum(_as_int(row.get("unresolved_claims")) for row in quality_rows),
    }
    verdict_counts: Counter[str] = Counter()
    for row in quality_rows:
        verdicts = row.get("verdict_counts", {})
        if isinstance(verdicts, Mapping):
            verdict_counts.update({str(key): _as_int(value) for key, value in verdicts.items()})
    quality_totals["verdict_counts"] = dict(verdict_counts)
    quality_totals["claim_citation_coverage"] = _round(
        quality_totals["cited_claims"] / quality_totals["claims"]
        if quality_totals["claims"]
        else None
    )
    quality_totals["unresolved_rate"] = _round(
        quality_totals["unresolved_claims"] / quality_totals["claims"]
        if quality_totals["claims"]
        else None
    )

    latency_all = [float(row["latency_ms"]) for row in cases if _number(row.get("latency_ms")) is not None]
    latency_non_error = [
        float(row["latency_ms"])
        for row in cases
        if row.get("status") != "error" and _number(row.get("latency_ms")) is not None
    ]

    def latency_stats(values: list[float]) -> dict[str, Any]:
        return {
            "cases": len(values),
            "mean_ms": _round(_mean(values)),
            "p50_ms": _round(_percentile(values, 0.50)),
            "p90_ms": _round(_percentile(values, 0.90)),
            "p95_ms": _round(_percentile(values, 0.95)),
            "p99_ms": _round(_percentile(values, 0.99)),
            "max_ms": _round(max(values) if values else None),
        }

    generated = [str(payload.get("generated_at", "")) for payload in chunks]
    source_files = [path.name for _, _, path, _ in files]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "pubmedqa",
            "split": "test",
            "cases_attempted": expected_cases,
            "cases_detail": len(cases),
            "window_count": len(files),
            "window_size": 5,
            "coverage_complete": len(files) == expected_cases // 5,
            "source_files": source_files,
            "source_generated_at_min": min(generated) if generated else None,
            "source_generated_at_max": max(generated) if generated else None,
        },
        "status_counts": dict(status_counts),
        "answer": answer_summary,
        "confusion_matrix": dict(sorted(confusion.items())),
        "by_expected_label": by_expected,
        "answer_stage_counts": dict(answer_stage_counts),
        "error_type_counts": dict(error_type_counts),
        "quality": {
            "per_case_mean": quality_mean,
            "claim_totals": quality_totals,
            "cases_with_quality": len(quality_rows),
        },
        "latency": {
            "all_cases": latency_stats(latency_all),
            "non_error_cases": latency_stats(latency_non_error),
        },
        "cost": _aggregate_usage(chunks),
        "cases_detail": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总可恢复的 PubMedQA agent 评测分片")
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=500)
    args = parser.parse_args()
    payload = aggregate(args.chunk_root, expected_cases=args.expected_cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "cases": payload["scope"]["cases_detail"],
        "status_counts": payload["status_counts"],
        "answer": payload["answer"],
        "latency_non_error": payload["latency"]["non_error_cases"],
        "telemetry_complete": payload["cost"]["telemetry_complete"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
