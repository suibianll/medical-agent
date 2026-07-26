"""Deterministic answerability and harm-aware routing for medical runs."""

from __future__ import annotations

from typing import Any


EMERGENCY_MARKERS = (
    "胸痛",
    "呼吸困难",
    "意识丧失",
    "大出血",
    "严重过敏",
    "自杀",
    "急救",
    "emergency",
)
HIGH_RISK_MARKERS = (
    "剂量",
    "禁忌",
    "相互作用",
    "诊断",
    "处方",
    "停药",
    "急症",
    "风险",
    "dose",
    "contraindication",
)


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def route_decision(
    *,
    request: str,
    patient_record: str,
    claims: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Return a safe routing decision without an additional model call.

    This is intentionally conservative and explainable. Thresholds should be
    recalibrated on the clinical gold set before any production deployment.
    """

    request_text = str(request or "")
    # Use the user's request as the primary risk signal. Evidence and claim
    # text may quote generic emergency instructions and would otherwise create
    # false escalations for ordinary questions.
    emergency = _contains_marker(request_text, EMERGENCY_MARKERS)
    high_risk = emergency or _contains_marker(request_text, HIGH_RISK_MARKERS)
    issues = [
        str(issue.get("code"))
        for issue in evaluation.get("issues", [])
        if isinstance(issue, dict) and issue.get("code")
    ]
    passed = bool(evaluation.get("pass", False)) and not issues
    reasons: list[str] = []
    missing: list[str] = []

    if not patient_record.strip() and high_risk:
        missing.append("患者病历或关键临床上下文")
    if not claims:
        missing.append("可直接支持结论的证据")
    if issues:
        reasons.append("证据链核验未通过")
    if emergency:
        reasons.append("问题包含急症或紧急风险信号")
    elif high_risk:
        reasons.append("问题涉及高风险临床决策")

    if emergency and not passed:
        outcome = "emergency_escalation"
    elif not passed:
        outcome = "defer"
    elif not claims:
        outcome = "ask_clarification"
    else:
        outcome = "answer"

    return {
        "outcome": outcome,
        "risk_level": "high" if high_risk else "standard",
        "emergency_signal": emergency,
        "reasons": reasons[:4],
        "missing": missing[:4],
        "issue_codes": list(dict.fromkeys(issues))[:12],
    }
