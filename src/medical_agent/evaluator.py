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
) -> dict[str, Any]:
    """Check evidence existence, patient grounding and medical grounding.

    The optional model is used only for a deliberately simple semantic verdict.
    Deterministic gates always run first, making the MVP useful with a weak
    evaluator or no external model at all.
    """

    evidence_by_id = _evidence_map(evidence)
    issues: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []

    for index, claim in enumerate(claims, start=1):
        claim_id = _claim_id(claim, index)
        refs = claim.get("refs", [])
        if not isinstance(refs, list) or not refs:
            issues.append({"claim": claim_id, "code": "NO_REF"})
            continue

        valid_refs = [ref for ref in refs if isinstance(ref, str) and ref in evidence_by_id]
        if len(valid_refs) != len(refs):
            issues.append({"claim": claim_id, "code": "BAD_REF"})
            continue

        if _requires_dual_support(claim):
            if not any(ref.startswith("P") for ref in valid_refs):
                issues.append({"claim": claim_id, "code": "MISSING_PATIENT_REF"})
            if not any(ref.startswith("K") for ref in valid_refs):
                issues.append({"claim": claim_id, "code": "MISSING_KB_REF"})

        # Citation-type completeness and semantic support are independent
        # gates.  Keep both findings when both fail so repair routing can
        # reopen retrieval roots and also narrow an unsupported conclusion.
        if model is not None:
            selected_evidence = [evidence_by_id[ref] for ref in valid_refs]
            verdict = model.judge_claim(claim=claim, evidence=selected_evidence)
            if verdict not in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            judgements.append({"claim": claim_id, "verdict": verdict})
            if verdict == "NOT_SUPPORTED":
                issues.append({"claim": claim_id, "code": "NOT_SUPPORTED"})

    return {"pass": not issues, "issues": issues, "judgements": judgements}
