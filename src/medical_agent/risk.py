"""Pluggable, configuration-driven decision routing for medical runs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ports import DecisionRouter


ROUTE_OUTCOMES = frozenset(
    {"answer", "ask_clarification", "defer", "emergency_escalation"}
)
ROUTE_RISK_LEVELS = frozenset({"standard", "high", "unknown"})
MAX_ROUTER_RESPONSE_BYTES = 512_000


def _issue_codes(evaluation: Mapping[str, Any]) -> list[str]:
    raw_issues = evaluation.get("issues", [])
    if not isinstance(raw_issues, list):
        return []
    return list(
        dict.fromkeys(
            str(issue.get("code"))
            for issue in raw_issues
            if isinstance(issue, dict) and issue.get("code")
        )
    )[:12]


class EvidenceDecisionRouter:
    """Safe local fallback based only on evidence-chain state.

    It deliberately does not inspect request text. A deployment that needs
    emergency classification must inject a configured rule or external API
    router; an unverified run always remains deferred here.
    """

    def decide(
        self,
        *,
        request: str,
        patient_record: str,
        claims: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        del request
        issues = _issue_codes(evaluation)
        passed = bool(evaluation.get("pass", False)) and not issues
        reasons: list[str] = []
        missing: list[str] = []
        if issues:
            reasons.append("evidence_chain_failed")
        if not patient_record.strip() and claims:
            reasons.append("patient_context_not_provided")
        if not claims:
            missing.append("directly_supporting_evidence")

        if not passed:
            outcome = "defer"
        elif not claims:
            outcome = "ask_clarification"
        else:
            outcome = "answer"
        return {
            "outcome": outcome,
            "risk_level": "unknown",
            "emergency_signal": False,
            "reasons": reasons[:4],
            "missing": missing[:4],
            "issue_codes": issues,
            "router": "evidence",
        }


class PatternDecisionRouter:
    """Apply only patterns supplied by configuration, never code constants."""

    def __init__(self, rules: Iterable[Mapping[str, Any]]) -> None:
        self._rules: list[dict[str, Any]] = []
        for raw_rule in list(rules)[:16]:
            if not isinstance(raw_rule, Mapping):
                continue
            outcome = str(raw_rule.get("outcome", "")).strip()
            if outcome not in ROUTE_OUTCOMES:
                continue
            patterns = raw_rule.get("patterns", [])
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list):
                continue
            compiled: list[re.Pattern[str]] = []
            for raw_pattern in patterns[:16]:
                try:
                    pattern = re.compile(str(raw_pattern)[:240], flags=re.IGNORECASE)
                except (re.error, TypeError):
                    continue
                compiled.append(pattern)
            if not compiled:
                continue
            self._rules.append(
                {
                    "patterns": compiled,
                    "outcome": outcome,
                    "risk_level": str(raw_rule.get("risk_level", "high")),
                    "emergency_signal": bool(raw_rule.get("emergency_signal", False)),
                    "reason": str(raw_rule.get("reason", "configured_rule"))[:120],
                }
            )

    def decide(
        self,
        *,
        request: str,
        patient_record: str,
        claims: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = EvidenceDecisionRouter().decide(
            request=request,
            patient_record=patient_record,
            claims=claims,
            evaluation=evaluation,
        )
        request_text = str(request or "")
        for rule in self._rules:
            if not any(pattern.search(request_text) for pattern in rule["patterns"]):
                continue
            risk_level = rule["risk_level"]
            if risk_level not in ROUTE_RISK_LEVELS:
                risk_level = "high"
            outcome = rule["outcome"]
            if outcome not in {"emergency_escalation", "defer"} and fallback[
                "outcome"
            ] != "answer":
                outcome = fallback["outcome"]
            return {
                **fallback,
                "outcome": outcome,
                "risk_level": risk_level,
                "emergency_signal": bool(rule["emergency_signal"]),
                "reasons": [rule["reason"], *fallback["reasons"]][:4],
                "router": "configured_rules",
            }
        return {**fallback, "router": "configured_rules"}


class ExternalApiDecisionRouter:
    """Call a configured risk classifier and enforce local evidence safety gates."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        provider: str = "generic",
        model: str = "",
        timeout_seconds: int = 30,
        send_patient_record: bool = False,
    ) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint.startswith(("https://", "http://")):
            raise ValueError("routing endpoint 必须以 http:// 或 https:// 开头。")
        if timeout_seconds <= 0:
            raise ValueError("routing 服务超时时间必须大于 0。")
        self.endpoint = endpoint
        self._api_key = api_key.strip()
        self.provider = provider.strip().lower() or "generic"
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.send_patient_record = bool(send_patient_record)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(
        self,
        *,
        request: str,
        patient_record: str,
        claims: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request": str(request or "")[:4000],
            "context": {
                "patient_record_present": bool(patient_record.strip()),
                "claim_count": len(claims),
                "evidence_pass": bool(evaluation.get("pass", False)),
                "issue_codes": _issue_codes(evaluation),
            },
        }
        if self.model:
            payload["model"] = self.model
        if self.send_patient_record:
            payload["patient_record"] = str(patient_record or "")[:8000]
        return payload

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise ValueError
        outcome = str(response.get("outcome", "")).strip()
        if outcome not in ROUTE_OUTCOMES:
            raise ValueError
        risk_level = str(response.get("risk_level", "unknown")).strip()
        if risk_level not in ROUTE_RISK_LEVELS:
            risk_level = "unknown"
        raw_reasons = response.get("reasons", [])
        raw_missing = response.get("missing", [])
        if isinstance(raw_reasons, str):
            raw_reasons = [raw_reasons]
        if isinstance(raw_missing, str):
            raw_missing = [raw_missing]
        if not isinstance(raw_reasons, list):
            raw_reasons = []
        if not isinstance(raw_missing, list):
            raw_missing = []
        reasons = [str(item)[:120] for item in raw_reasons if str(item).strip()]
        missing = [str(item)[:120] for item in raw_missing if str(item).strip()]
        return {
            "outcome": outcome,
            "risk_level": risk_level,
            "emergency_signal": bool(response.get("emergency_signal", False)),
            "reasons": reasons[:4],
            "missing": missing[:4],
            "issue_codes": [],
            "router": "external_api",
        }

    def decide(
        self,
        *,
        request: str,
        patient_record: str,
        claims: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = EvidenceDecisionRouter().decide(
            request=request,
            patient_record=patient_record,
            claims=claims,
            evaluation=evaluation,
        )
        request_payload = self._payload(
            request=request,
            patient_record=patient_record,
            claims=claims,
            evaluation=evaluation,
        )
        try:
            request_object = Request(
                self.endpoint,
                data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            with urlopen(request_object, timeout=self.timeout_seconds) as response:  # noqa: S310
                encoded = response.read(MAX_ROUTER_RESPONSE_BYTES + 1)
                if len(encoded) > MAX_ROUTER_RESPONSE_BYTES:
                    raise ValueError
                normalized = self._normalize_response(json.loads(encoded.decode("utf-8")))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return {
                **fallback,
                "reasons": ["router_unavailable", *fallback["reasons"]][:4],
                "router": "external_api_fallback",
            }

        # A remote classifier cannot turn a failed citation gate into a safe
        # answer. Emergency escalation remains available for explicit remote
        # signals, while all other outcomes inherit the local defer gate.
        if fallback["outcome"] != "answer":
            if normalized["outcome"] not in {"emergency_escalation", "defer"}:
                normalized["outcome"] = fallback["outcome"]
        normalized["issue_codes"] = _issue_codes(evaluation)
        return normalized


def route_decision(
    *,
    request: str,
    patient_record: str,
    claims: list[dict[str, Any]],
    evaluation: dict[str, Any],
    router: DecisionRouter | None = None,
) -> dict[str, Any]:
    """Compatibility entry point using the evidence-only policy by default."""

    selected = router or EvidenceDecisionRouter()
    return selected.decide(
        request=request,
        patient_record=patient_record,
        claims=claims,
        evaluation=evaluation,
    )
