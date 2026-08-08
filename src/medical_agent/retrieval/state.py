"""Bounded retrieval state used for observable, stoppable search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalState:
    """Small state machine; it deliberately does not contain source text."""

    max_rounds: int = 2
    max_candidates: int = 12
    rounds: int = 0
    queries: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    relevant_candidate_count: int = 0
    scope_coverage: float = 0.0
    best_score: float | None = None
    average_score: float | None = None
    score_coverage: float = 0.0
    score_gate_passed: bool = True
    stop_reason: str = ""

    def begin_round(self, queries: list[str]) -> None:
        if self.rounds >= self.max_rounds:
            self.stop_reason = "round_budget_exhausted"
            return
        self.rounds += 1
        for query in queries:
            if query and query not in self.queries:
                self.queries.append(query)

    def add_evidence(self, evidence_ids: list[str]) -> None:
        for evidence_id in evidence_ids:
            if evidence_id and evidence_id not in self.evidence_ids:
                self.evidence_ids.append(evidence_id)
        if len(self.evidence_ids) > self.max_candidates:
            self.evidence_ids = self.evidence_ids[: self.max_candidates]

    def finish(self) -> None:
        if self.stop_reason:
            return
        if not self.evidence_ids:
            self.stop_reason = "no_evidence"
        elif len(self.evidence_ids) >= self.max_candidates:
            self.stop_reason = "candidate_budget_exhausted"
        elif not self.score_gate_passed:
            self.stop_reason = "low_relevance"
        else:
            self.stop_reason = "evidence_found"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "max_rounds": self.max_rounds,
            "candidate_count": len(self.evidence_ids),
            "max_candidates": self.max_candidates,
            "query_count": len(self.queries),
            "relevant_candidate_count": self.relevant_candidate_count,
            "scope_coverage": round(self.scope_coverage, 4),
            "best_score": self.best_score,
            "average_score": self.average_score,
            "score_coverage": round(self.score_coverage, 4),
            "score_gate_passed": self.score_gate_passed,
            "missing_aspects": list(self.missing_aspects),
            "stop_reason": self.stop_reason or "in_progress",
        }
