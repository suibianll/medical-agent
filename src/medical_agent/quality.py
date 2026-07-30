"""Dependency-free quality metrics for retrieval and evidence chains.

The helpers in this module are intentionally offline and deterministic.  They
accept ranked IDs or already-validated run structures, never call a model, and
return aggregate numbers without echoing query, patient, evidence, or claim
text.  This makes them suitable for a small regression set in CI as well as a
safe per-run quality summary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


MAX_QUALITY_CASES = 2_048
MAX_QUALITY_IDS = 256
MAX_QUALITY_K = 128
SAFE_ABSTENTION_OUTCOMES = frozenset(
    {"ask_clarification", "defer", "emergency_escalation"}
)


def _normalise_ids(values: Any, *, limit: int = MAX_QUALITY_IDS) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("id", value.get("document_id", ""))
        if not isinstance(value, (str, int)):
            continue
        candidate = str(value).strip()
        if candidate and candidate not in result:
            result.append(candidate)
        if len(result) >= limit:
            break
    return result


def _normalise_ks(ks: Iterable[int]) -> list[int]:
    result: list[int] = []
    for value in ks:
        try:
            candidate = max(1, min(int(value), MAX_QUALITY_K))
        except (TypeError, ValueError):
            continue
        if candidate not in result:
            result.append(candidate)
    return sorted(result) or [1, 3, 5]


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Return binary-set recall for one ranked result list."""

    relevant = set(_normalise_ids(relevant_ids))
    if not relevant:
        return 0.0
    ranked = _normalise_ids(ranked_ids)
    try:
        bounded_k = max(1, min(int(k), MAX_QUALITY_K))
    except (TypeError, ValueError):
        bounded_k = 1
    return len(set(ranked[:bounded_k]) & relevant) / len(relevant)


def mean_reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""

    relevant = set(_normalise_ids(relevant_ids))
    if not relevant:
        return 0.0
    for index, identifier in enumerate(_normalise_ids(ranked_ids), start=1):
        if identifier in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Return binary-gain nDCG for one ranked result list."""

    relevant = set(_normalise_ids(relevant_ids))
    if not relevant:
        return 0.0
    try:
        bounded_k = max(1, min(int(k), MAX_QUALITY_K))
    except (TypeError, ValueError):
        bounded_k = 1
    ranked = _normalise_ids(ranked_ids)
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, identifier in enumerate(ranked[:bounded_k])
        if identifier in relevant
    )
    ideal_hits = min(len(relevant), bounded_k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval_cases(
    cases: Iterable[Mapping[str, Any]],
    search: Callable[[str], Sequence[Any]],
    *,
    ks: Iterable[int] = (1, 3, 5),
    max_cases: int = MAX_QUALITY_CASES,
) -> dict[str, Any]:
    """Evaluate a bounded gold set without making model calls.

    Each case must contain a non-empty ``query`` and ``relevant_ids``.  The
    search callback may return IDs or document dictionaries containing ``id``
    / ``document_id``.  Case identifiers are retained only when supplied, and
    raw query text is deliberately excluded from the report.
    """

    bounded_cases = max(1, min(int(max_cases), MAX_QUALITY_CASES))
    requested_ks = _normalise_ks(ks)
    totals: dict[str, list[float]] = {
        f"recall@{k}": [] for k in requested_ks
    }
    totals["mrr"] = []
    totals.update({f"ndcg@{k}": [] for k in requested_ks})
    per_case: list[dict[str, Any]] = []
    seen = evaluated = invalid = failed = 0
    for case in cases:
        if seen >= bounded_cases:
            break
        seen += 1
        if not isinstance(case, Mapping):
            invalid += 1
            continue
        query = " ".join(str(case.get("query", "")).split())
        relevant = _normalise_ids(case.get("relevant_ids", []))
        if not query or not relevant:
            invalid += 1
            continue
        try:
            ranked = _normalise_ids(search(query))
        except Exception:  # noqa: BLE001 - one broken case must not abort a set
            failed += 1
            continue
        values: dict[str, float] = {}
        for k in requested_ks:
            value = recall_at_k(ranked, relevant, k)
            values[f"recall@{k}"] = round(value, 6)
            totals[f"recall@{k}"].append(value)
        mrr = mean_reciprocal_rank(ranked, relevant)
        values["mrr"] = round(mrr, 6)
        totals["mrr"].append(mrr)
        for k in requested_ks:
            value = ndcg_at_k(ranked, relevant, k)
            values[f"ndcg@{k}"] = round(value, 6)
            totals[f"ndcg@{k}"].append(value)
        evaluated += 1
        case_id = case.get("id")
        item: dict[str, Any] = {"index": seen, "metrics": values}
        if isinstance(case_id, (str, int)) and str(case_id).strip():
            item["id"] = str(case_id).strip()[:120]
        per_case.append(item)

    metrics = {
        name: round(sum(values) / len(values), 6) if values else 0.0
        for name, values in totals.items()
    }
    return {
        "case_count": seen,
        "evaluated_cases": evaluated,
        "invalid_cases": invalid,
        "failed_cases": failed,
        "ks": requested_ks,
        "metrics": metrics,
        "per_case": per_case[:MAX_QUALITY_CASES],
    }


def _snapshot_evidence_ids(snapshot: Mapping[str, Any]) -> list[str]:
    identifiers = _normalise_ids(
        snapshot.get(
            "evidence_keys",
            snapshot.get("evidence_ids", snapshot.get("evidence", [])),
        )
    )
    claims = snapshot.get("claims", [])
    if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes)):
        for claim in claims[:MAX_QUALITY_IDS]:
            if isinstance(claim, Mapping):
                identifiers.extend(_normalise_ids(claim.get("refs", [])))
    return list(dict.fromkeys(identifiers))[:MAX_QUALITY_IDS]


def _snapshot_outcome(snapshot: Mapping[str, Any]) -> str:
    decision = snapshot.get("decision", snapshot)
    if isinstance(decision, Mapping):
        for field_name in ("outcome", "status", "decision"):
            value = decision.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()[:80]
        return ""
    return str(decision or "").strip().lower()[:80]


def _snapshot_citations_are_integral(snapshot: Mapping[str, Any]) -> bool:
    claims = snapshot.get("claims", [])
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        return True
    evidence_ids = set(
        _normalise_ids(snapshot.get("evidence_ids", snapshot.get("evidence", [])))
    )
    for claim in claims[:MAX_QUALITY_IDS]:
        if not isinstance(claim, Mapping):
            continue
        refs = _normalise_ids(claim.get("refs", []))
        if refs and not evidence_ids:
            return False
        if any(ref not in evidence_ids for ref in refs):
            return False
    return True


def evaluate_counterfactual_cases(
    cases: Iterable[Mapping[str, Any]],
    *,
    max_cases: int = MAX_QUALITY_CASES,
) -> dict[str, Any]:
    """Score paired baseline/counterfactual runs without model calls.

    A case contains ``baseline``, ``counterfactual`` and ``expected``
    mappings.  ``expected.decision_should_change`` is required; optional
    ``evidence_should_change``, ``expected_outcome`` and ``must_abstain``
    fields turn the same fixture into a clinical safety regression gate.  The
    report contains only booleans, counters and supplied case IDs—never query,
    patient, claim or evidence text.
    """

    try:
        bounded_cases = max(1, min(int(max_cases), MAX_QUALITY_CASES))
    except (TypeError, ValueError):
        bounded_cases = MAX_QUALITY_CASES
    per_case: list[dict[str, Any]] = []
    seen = evaluated = invalid = 0
    decision_scores: list[float] = []
    evidence_scores: list[float] = []
    citation_scores: list[float] = []
    abstention_scores: list[float] = []
    regressions = 0

    for case in cases:
        if seen >= bounded_cases:
            break
        seen += 1
        if not isinstance(case, Mapping):
            invalid += 1
            continue
        baseline = case.get("baseline")
        counterfactual = case.get("counterfactual")
        expected = case.get("expected")
        if not all(isinstance(value, Mapping) for value in (baseline, counterfactual, expected)):
            invalid += 1
            continue
        expected_change = expected.get("decision_should_change")
        if not isinstance(expected_change, bool):
            invalid += 1
            continue

        baseline_outcome = _snapshot_outcome(baseline)
        counterfactual_outcome = _snapshot_outcome(counterfactual)
        decision_changed = baseline_outcome != counterfactual_outcome
        decision_responsive = decision_changed == expected_change
        decision_scores.append(float(decision_responsive))

        evidence_expected = expected.get("evidence_should_change")
        evidence_responsive: bool | None = None
        if isinstance(evidence_expected, bool):
            baseline_ids = set(_snapshot_evidence_ids(baseline))
            counterfactual_ids = set(_snapshot_evidence_ids(counterfactual))
            evidence_changed = baseline_ids != counterfactual_ids
            evidence_responsive = evidence_changed == evidence_expected
            evidence_scores.append(float(evidence_responsive))

        citation_integrity = _snapshot_citations_are_integral(counterfactual)
        citation_scores.append(float(citation_integrity))

        abstention_expected = expected.get("must_abstain")
        safe_abstention: bool | None = None
        if isinstance(abstention_expected, bool):
            safe_outcomes = _normalise_ids(
                expected.get("safe_outcomes", SAFE_ABSTENTION_OUTCOMES)
            )
            safe_abstention = (
                not abstention_expected
                or counterfactual_outcome in set(safe_outcomes)
            )
            abstention_scores.append(float(safe_abstention))

        expected_outcome = expected.get("expected_outcome")
        outcome_matches: bool | None = None
        if isinstance(expected_outcome, str) and expected_outcome.strip():
            outcome_matches = counterfactual_outcome == expected_outcome.strip().lower()[:80]

        regression = not decision_responsive or not citation_integrity
        if evidence_responsive is False or safe_abstention is False or outcome_matches is False:
            regression = True
        regressions += int(regression)
        evaluated += 1
        item: dict[str, Any] = {
            "index": seen,
            "decision_changed": decision_changed,
            "decision_responsive": decision_responsive,
            "citation_integrity": citation_integrity,
            "regression": regression,
        }
        if evidence_responsive is not None:
            item["evidence_responsive"] = evidence_responsive
        if safe_abstention is not None:
            item["safe_abstention"] = safe_abstention
        if outcome_matches is not None:
            item["outcome_matches"] = outcome_matches
        case_id = case.get("id")
        if isinstance(case_id, (str, int)) and str(case_id).strip():
            item["id"] = str(case_id).strip()[:120]
        per_case.append(item)

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "case_count": seen,
        "evaluated_cases": evaluated,
        "invalid_cases": invalid,
        "metrics": {
            "decision_responsiveness": average(decision_scores),
            "evidence_responsiveness": average(evidence_scores),
            "citation_integrity": average(citation_scores),
            "safe_abstention_rate": average(abstention_scores),
            "regression_rate": round(regressions / evaluated, 6) if evaluated else 0.0,
        },
        "per_case": per_case[:MAX_QUALITY_CASES],
    }


def evaluate_counterfactual_pairs(
    cases: Iterable[Mapping[str, Any]],
    *,
    max_cases: int = MAX_QUALITY_CASES,
) -> dict[str, Any]:
    """Compatibility alias emphasizing that fixtures are paired runs."""

    return evaluate_counterfactual_cases(cases, max_cases=max_cases)


def evaluate_evidence_chain(
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise citation and support-edge quality for one completed run."""

    bounded_claims = [claim for claim in claims[:MAX_QUALITY_CASES] if isinstance(claim, Mapping)]
    if isinstance(evidence, Mapping):
        evidence_ids = {
            str(identifier)
            for identifier in list(evidence)[:MAX_QUALITY_IDS]
            if str(identifier).strip()
        }
    else:
        evidence_ids = {
            str(item.get("id", ""))
            for item in evidence[:MAX_QUALITY_IDS]
            if isinstance(item, Mapping) and str(item.get("id", "")).strip()
        }
    claim_count = len(bounded_claims)
    total_refs = valid_refs = cited_claims = 0
    dual_claims = dual_supported = 0
    issue_claim_ids: set[str] = set()
    raw_evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    raw_issues = raw_evaluation.get("issues", [])
    if isinstance(raw_issues, Sequence) and not isinstance(raw_issues, (str, bytes)):
        for issue in raw_issues[:MAX_QUALITY_IDS]:
            if isinstance(issue, Mapping) and str(issue.get("claim", "")).strip():
                issue_claim_ids.add(str(issue["claim"]))

    for claim in bounded_claims:
        claim_id = str(claim.get("id", ""))
        refs = _normalise_ids(claim.get("refs", []), limit=MAX_QUALITY_IDS)
        total_refs += len(refs)
        valid = [ref for ref in refs if ref in evidence_ids]
        valid_refs += len(valid)
        if valid:
            cited_claims += 1
        if bool(claim.get("requires_dual_support")):
            dual_claims += 1
            prefixes = {ref[:1] for ref in valid}
            if {"P", "K"}.issubset(prefixes) and claim_id not in issue_claim_ids:
                dual_supported += 1

    raw_edges = raw_evaluation.get("support_edges", [])
    edge_claim_ids = {
        str(edge.get("claim_id"))
        for edge in raw_edges[:MAX_QUALITY_IDS]
        if isinstance(edge, Mapping) and str(edge.get("claim_id", "")).strip()
    } if isinstance(raw_edges, Sequence) and not isinstance(raw_edges, (str, bytes)) else set()
    unresolved = len({claim_id for claim_id in issue_claim_ids if claim_id})
    denominator = claim_count or 1
    raw_verdict_counts = raw_evaluation.get("verdict_counts", {})
    if not isinstance(raw_verdict_counts, Mapping):
        raw_verdict_counts = {}
    verdict_counts = {
        str(key): int(value)
        for key, value in raw_verdict_counts.items()
        if isinstance(key, str)
        and isinstance(value, int)
        and value >= 0
    }
    judged_count = sum(verdict_counts.values())
    return {
        "claims": claim_count,
        "cited_claims": cited_claims,
        "citation_coverage": round(cited_claims / denominator, 6) if claim_count else 0.0,
        "citation_precision": round(valid_refs / total_refs, 6) if total_refs else 0.0,
        "support_edge_coverage": round(len(edge_claim_ids & {str(c.get("id", "")) for c in bounded_claims}) / denominator, 6)
        if claim_count
        else 0.0,
        "dual_support_claims": dual_claims,
        "dual_support_coverage": round(dual_supported / dual_claims, 6) if dual_claims else 1.0,
        "unresolved_claims": unresolved,
        "unresolved_rate": round(unresolved / denominator, 6) if claim_count else 0.0,
        "verdict_counts": verdict_counts,
        "contradiction_rate": round(
            verdict_counts.get("CONTRADICTED", 0) / judged_count, 6
        )
        if judged_count
        else 0.0,
        "insufficient_rate": round(
            (
                verdict_counts.get("INSUFFICIENT", 0)
                + verdict_counts.get("UNCERTAIN", 0)
            )
            / judged_count,
            6,
        )
        if judged_count
        else 0.0,
    }


__all__ = [
    "evaluate_counterfactual_cases",
    "evaluate_counterfactual_pairs",
    "evaluate_evidence_chain",
    "evaluate_retrieval_cases",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
]
