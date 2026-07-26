"""Code-first evidence-chain evaluation with a small semantic hook."""

from __future__ import annotations

from typing import Any


CLINICAL_KEYWORDS = (
    "建议",
    "应当",
    "应该",
    "需要",
    "调整",
    "剂量",
    "禁忌",
    "治疗",
    "诊断",
    "处置",
    "风险",
)


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
    if "requires_dual_support" in claim:
        return bool(claim["requires_dual_support"])
    text = str(claim.get("text", ""))
    return any(keyword in text for keyword in CLINICAL_KEYWORDS)


def evaluate_claims(
    claims: list[dict[str, Any]],
    evidence: Any,
    model: Any | None = None,
    semantic_cache: dict[tuple[str, tuple[str, ...]], str] | None = None,
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
        cached_verdicts: dict[str, str] = {}
        uncached_candidates: list[dict[str, Any]] = []
        cache_keys: dict[str, tuple[str, tuple[str, ...]]] = {}
        for item in semantic_candidates:
            claim = item["claim"]
            key = (
                str(claim.get("text", "")).strip(),
                tuple(str(ref) for ref in claim.get("refs", [])),
            )
            cache_keys[item["id"]] = key
            cached = semantic_cache.get(key) if semantic_cache is not None else None
            if cached in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
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
            verdict = str(verdicts.get(claim_id, "UNCERTAIN")).upper()
            if verdict not in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            if semantic_cache is not None:
                semantic_cache[cache_keys[claim_id]] = verdict
            judgements.append({"claim": claim_id, "verdict": verdict})
            for edge in support_edges:
                if edge["claim_id"] != claim_id:
                    continue
                edge["verifier"] = "semantic_judge"
                edge["status"] = verdict.lower()
                edge["relation"] = {
                    "SUPPORTED": "supports",
                    "NOT_SUPPORTED": "contradicts",
                    "UNCERTAIN": "uncertain",
                }[verdict]
                edge["verifier_score"] = {
                    "SUPPORTED": 1.0,
                    "NOT_SUPPORTED": 0.0,
                    "UNCERTAIN": 0.5,
                }[verdict]
            if verdict == "NOT_SUPPORTED":
                issues.append({"claim": claim_id, "code": "NOT_SUPPORTED"})

    for edge in support_edges:
        if edge["status"] == "pending":
            edge["status"] = "deterministic_pass"
    return {
        "pass": not issues,
        "issues": issues,
        "judgements": judgements,
        "support_edges": support_edges,
    }
