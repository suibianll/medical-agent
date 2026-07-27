"""Configuration-driven source governance for clinical retrieval.

The policy is deliberately metadata-only: it never inspects medical text or
uses keyword routing.  Deployments can reject retracted/unsafe material,
require versioned sources, constrain source types, and set a minimum trust
priority before a document reaches lexical, dense, or reranker stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping


MAX_SOURCE_TYPES = 16
MAX_BLOCKED_STATUSES = 16
DEFAULT_BLOCKED_STATUSES = ("retracted", "revoked", "unsafe", "poisoned")


def _normalise_values(values: Any, *, limit: int) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).lower()
        if normalized and normalized not in result:
            result.append(normalized)
        if len(result) >= limit:
            break
    return tuple(result)


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _parse_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


@dataclass(frozen=True, slots=True)
class SourceGovernancePolicy:
    """Bounded, metadata-only allow/deny policy for source documents."""

    allowed_source_types: tuple[str, ...] = ()
    blocked_statuses: tuple[str, ...] = DEFAULT_BLOCKED_STATUSES
    min_priority: int = 0
    require_version: bool = False
    allow_synthetic: bool = True
    max_age_days: int = 0
    reject_unknown_date: bool = False
    as_of_date: date | None = None
    enabled: bool = False

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "SourceGovernancePolicy":
        if not isinstance(raw, Mapping) or not raw:
            return cls()
        raw_as_of = _parse_date(raw.get("as_of_date"))
        try:
            min_priority = max(0, min(int(raw.get("min_priority", 0)), 100))
        except (TypeError, ValueError):
            min_priority = 0
        try:
            max_age_days = max(0, min(int(raw.get("max_age_days", 0)), 36_500))
        except (TypeError, ValueError):
            max_age_days = 0
        blocked = (
            _normalise_values(raw["blocked_statuses"], limit=MAX_BLOCKED_STATUSES)
            if "blocked_statuses" in raw
            else DEFAULT_BLOCKED_STATUSES
        )
        return cls(
            allowed_source_types=_normalise_values(
                raw.get("allowed_source_types", []), limit=MAX_SOURCE_TYPES
            ),
            blocked_statuses=blocked,
            min_priority=min_priority,
            require_version=_parse_bool(raw.get("require_version"), False),
            allow_synthetic=_parse_bool(raw.get("allow_synthetic"), True),
            max_age_days=max_age_days,
            reject_unknown_date=_parse_bool(raw.get("reject_unknown_date"), False),
            as_of_date=raw_as_of,
            enabled=_parse_bool(raw.get("enabled"), True),
        )

    def _effective_date(self, document: Mapping[str, Any]) -> date | None:
        for field_name in (
            "effective_date",
            "updated_at",
            "published_at",
            "publication_date",
            "imported_at",
        ):
            parsed = _parse_date(document.get(field_name))
            if parsed is not None:
                return parsed
        return None

    def assess(self, document: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Return allow/deny plus safe metadata; no document text is copied."""

        if not self.enabled:
            return True, {"status": "not_configured"}
        source_type = str(document.get("source_type", "built_in")).strip().lower()
        status = ""
        for field_name in ("status", "publication_status", "safety_status"):
            candidate = str(document.get(field_name, "")).strip().lower()
            if candidate:
                status = candidate
                break
        try:
            priority = int(document.get("priority", 0) or 0)
        except (TypeError, ValueError):
            priority = 0
        reasons: list[str] = []
        if self.allowed_source_types and source_type not in self.allowed_source_types:
            reasons.append("source_type_not_allowed")
        if status and status in self.blocked_statuses:
            reasons.append("blocked_status")
        if priority < self.min_priority:
            reasons.append("priority_below_minimum")
        if not self.allow_synthetic and bool(document.get("synthetic", False)):
            reasons.append("synthetic_not_allowed")
        version = str(document.get("version", "")).strip()
        if self.require_version and not version:
            reasons.append("version_required")

        effective_date = self._effective_date(document)
        as_of = self.as_of_date or datetime.now(timezone.utc).date()
        age_days: int | None = None
        if self.max_age_days:
            if effective_date is None:
                if self.reject_unknown_date:
                    reasons.append("freshness_date_required")
            else:
                age_days = max(0, (as_of - effective_date).days)
                if age_days > self.max_age_days:
                    reasons.append("source_too_old")
        accepted = not reasons
        metadata: dict[str, Any] = {
            "status": "accepted" if accepted else "rejected",
            "source_type": source_type,
            "priority": priority,
            "versioned": bool(version),
            "reasons": reasons[:8],
        }
        if age_days is not None:
            metadata["age_days"] = age_days
        return accepted, metadata

    def filter_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled:
            return documents
        result: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, dict):
                continue
            accepted, metadata = self.assess(document)
            if not accepted:
                continue
            result.append({**document, "governance": metadata})
        return result

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allowed_source_types": list(self.allowed_source_types),
            "blocked_status_count": len(self.blocked_statuses),
            "min_priority": self.min_priority,
            "require_version": self.require_version,
            "allow_synthetic": self.allow_synthetic,
            "max_age_days": self.max_age_days,
            "reject_unknown_date": self.reject_unknown_date,
        }


__all__ = ["SourceGovernancePolicy"]
