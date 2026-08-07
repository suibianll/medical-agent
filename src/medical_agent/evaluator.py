"""Code-first evidence-chain evaluation with a small semantic hook."""

from __future__ import annotations

import hashlib
from typing import Any


SEMANTIC_VERDICTS = frozenset(
    {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT",
        # Legacy provider values remain accepted during a rolling upgrade.
        "NOT_SUPPORTED",
        "UNCERTAIN",
    }
)
VERDICT_RELATIONS = {
    "SUPPORTED": ("supports", 1.0),
    "PARTIALLY_SUPPORTED": ("qualifies", 0.5),
    "CONTRADICTED": ("contradicts", 0.0),
    "NOT_SUPPORTED": ("contradicts", 0.0),
    "INSUFFICIENT": ("uncertain", 0.25),
    "UNCERTAIN": ("uncertain", 0.25),
}
VERDICT_ISSUES = {
    "PARTIALLY_SUPPORTED": "PARTIAL_SUPPORT",
    "CONTRADICTED": "CONTRADICTED",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
    "INSUFFICIENT": "INSUFFICIENT_EVIDENCE",
    "UNCERTAIN": "INSUFFICIENT_EVIDENCE",
}


def _evidence_map(evidence: Any) -> dict[str, dict[str, Any]]:
    if isinstance(evidence, dict):
        return evidence
    if isinstance(evidence, list):
        return {
            item["id"]: item
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    return {}


def _claim_id(claim: dict[str, Any], index: int) -> str:
    value = claim.get("id")
    return value if isinstance(value, str) and value else f"C{index}"


def _requires_dual_support(claim: dict[str, Any]) -> bool:
    return bool(claim.get("requires_dual_support", False))


def _evidence_signature(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    content_hash = item.get("content_hash")
    version = item.get("version")
    if isinstance(content_hash, str) and content_hash:
        return f"{content_hash}:{version or ''}"
    text = str(item.get("text", ""))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{digest}:{version or ''}"


def _normalise_verdict(value: Any) -> str:
    verdict = str(value or "UNCERTAIN").upper()
    return verdict if verdict in SEMANTIC_VERDICTS else "UNCERTAIN"


def _normalise_edge_verdicts(value: Any) -> dict[str, str]:
    if not isinstance(value, (list, tuple, dict)):
        return {}
    entries = (
        list(value.items())
        if isinstance(value, dict)
        else [(None, item) for item in value]
    )
    normalized: dict[str, str] = {}
    for raw_key, raw_item in entries:
        if isinstance(value, dict):
            evidence_id = str(raw_key)
            verdict_value = raw_item
        elif isinstance(raw_item, dict):
            evidence_id = str(
                raw_item.get("evidence_id")
                or raw_item.get("id")
                or raw_item.get("ref")
                or ""
            )
            verdict_value = raw_item.get("verdict")
        else:
            continue
        if evidence_id:
            normalized[evidence_id] = _normalise_verdict(verdict_value)
    return normalized


def _normalise_judge_result(value: Any) -> tuple[str, dict[str, str]]:
    if isinstance(value, dict):
        verdict = _normalise_verdict(value.get("verdict"))
        edge_values = value.get(
            "evidence_verdicts", value.get("edges", value.get("evidence"))
        )
        return verdict, _normalise_edge_verdicts(edge_values)
    return _normalise_verdict(value), {}


def _aggregate_edge_verdicts(verdicts: list[str]) -> str:
    """Collapse edge verdicts without hiding a contradiction or gap."""

    if not verdicts:
        return "UNCERTAIN"
    normalized = [_normalise_verdict(value) for value in verdicts]
    if "CONTRADICTED" in normalized or "NOT_SUPPORTED" in normalized:
        return "CONTRADICTED"
    if "INSUFFICIENT" in normalized or "UNCERTAIN" in normalized:
        return "PARTIALLY_SUPPORTED" if "SUPPORTED" in normalized else "INSUFFICIENT"
    if "PARTIALLY_SUPPORTED" in normalized:
        return "PARTIALLY_SUPPORTED"
    return "SUPPORTED"


def evaluate_claims(
    claims: list[dict[str, Any]],
    evidence: Any,
    model: Any | None = None,
    semantic_cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    """Check evidence existence, patient grounding and medical grounding.

    The optional model is used only for a deliberately simple semantic verdict.
    Deterministic gates always run first, making the MVP useful with a weak
    evaluator or no external model at all.
    """

    evidence_by_id = _evidence_map(evidence)
    issues: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []
    semantic_candidates: list[dict[str, Any]] = []
    support_edges: list[dict[str, Any]] = []

    for index, claim in enumerate(claims, start=1):
        claim_id = _claim_id(claim, index)
        issue_count_before = len(issues)
        refs = claim.get("refs", [])
        if not isinstance(refs, list) or not refs:
            issues.append({"claim": claim_id, "code": "NO_REF"})
            continue

        valid_refs = [ref for ref in refs if isinstance(ref, str) and ref in evidence_by_id]
        if len(valid_refs) != len(refs):
            issues.append({"claim": claim_id, "code": "BAD_REF"})
            continue

        for ref in valid_refs:
            evidence_item = evidence_by_id[ref]
            span = evidence_item.get("span", {}) if isinstance(evidence_item, dict) else {}
            support_edges.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": ref,
                    "span_id": span.get("id", f"{ref}:span:0")
                    if isinstance(span, dict)
                    else f"{ref}:span:0",
                    "relation": "supports",
                    "verifier": "citation_gate",
                    "verifier_score": 1.0,
                    "status": "pending",
                }
            )

        if _requires_dual_support(claim):
            if not any(ref.startswith("P") for ref in valid_refs):
                issues.append({"claim": claim_id, "code": "MISSING_PATIENT_REF"})
            if not any(ref.startswith("K") for ref in valid_refs):
                issues.append({"claim": claim_id, "code": "MISSING_KB_REF"})

        # A semantic model cannot repair missing/invalid citations.  Defer its
        # call until deterministic gates pass, avoiding expensive duplicate
        # findings that lead to the same targeted rerun.
        if model is not None and len(issues) == issue_count_before:
            semantic_candidates.append(
                {
                    "id": claim_id,
                    "claim": claim,
                    "evidence": [evidence_by_id[ref] for ref in valid_refs],
                    "evidence_by_id": evidence_by_id,
                }
            )

    deterministic_failed_claims = {
        str(issue.get("claim"))
        for issue in issues
        if isinstance(issue, dict) and issue.get("claim")
    }
    for edge in support_edges:
        if edge["claim_id"] in deterministic_failed_claims:
            edge["status"] = "deterministic_fail"
            edge["relation"] = "uncertain"
            edge["verifier_score"] = 0.5

    if model is not None and semantic_candidates:
        cached_verdicts: dict[str, Any] = {}
        uncached_candidates: list[dict[str, Any]] = []
        cache_keys: dict[str, tuple[Any, ...]] = {}
        for item in semantic_candidates:
            claim = item["claim"]
            refs = tuple(str(ref) for ref in claim.get("refs", []))
            evidence_signatures = tuple(
                _evidence_signature(item["evidence_by_id"].get(ref))
                for ref in refs
            )
            key = (
                str(claim.get("text", "")).strip(),
                refs,
                evidence_signatures,
            )
            cache_keys[item["id"]] = key
            cached = semantic_cache.get(key) if semantic_cache is not None else None
            if (
                isinstance(cached, str) and cached in SEMANTIC_VERDICTS
            ) or isinstance(cached, dict):
                cached_verdicts[item["id"]] = cached
            else:
                uncached_candidates.append(item)

        if not uncached_candidates:
            verdicts: Any = {}
        else:
            verdicts = model.judge_claims(uncached_candidates)
        if not isinstance(verdicts, dict):
            verdicts = {}
        verdicts = {**cached_verdicts, **verdicts}
        for item in semantic_candidates:
            claim_id = item["id"]
            verdict, edge_verdicts = _normalise_judge_result(
                verdicts.get(claim_id, "UNCERTAIN")
            )
            if semantic_cache is not None:
                semantic_cache[cache_keys[claim_id]] = verdicts.get(
                    claim_id, "UNCERTAIN"
                )
            effective_edge_verdicts: dict[str, str] = {}
            for edge in support_edges:
                if edge["claim_id"] != claim_id:
                    continue
                edge_verdict = edge_verdicts.get(edge["evidence_id"], verdict)
                effective_edge_verdicts[edge["evidence_id"]] = edge_verdict
                edge["verifier"] = "semantic_judge"
                relation, score = VERDICT_RELATIONS.get(
                    edge_verdict, ("uncertain", 0.25)
                )
                edge["status"] = edge_verdict.lower()
                edge["relation"] = relation
                edge["verifier_score"] = score
            if edge_verdicts:
                verdict = _aggregate_edge_verdicts(list(effective_edge_verdicts.values()))
            judgement = {"claim": claim_id, "verdict": verdict}
            if edge_verdicts:
                judgement["edge_verdicts"] = effective_edge_verdicts
            judgements.append(judgement)
            issue_code = VERDICT_ISSUES.get(verdict)
            if issue_code:
                issues.append({"claim": claim_id, "code": issue_code})

    for edge in support_edges:
        if edge["status"] == "pending":
            edge["status"] = "deterministic_pass"
    verdict_counts: dict[str, int] = {}
    for judgement in judgements:
        verdict = str(judgement.get("verdict", "UNCERTAIN"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    edge_verdict_counts: dict[str, int] = {}
    for edge in support_edges:
        status = str(edge.get("status", ""))
        if status in {verdict.lower() for verdict in SEMANTIC_VERDICTS}:
            edge_verdict_counts[status.upper()] = edge_verdict_counts.get(status.upper(), 0) + 1
    return {
        "pass": not issues,
        "issues": issues,
        "judgements": judgements,
        "support_edges": support_edges,
        "verdict_counts": verdict_counts,
        "edge_verdict_counts": edge_verdict_counts,
    }
