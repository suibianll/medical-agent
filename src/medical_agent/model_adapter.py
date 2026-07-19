"""Model adapter contract.

An API-backed adapter may implement this protocol later.  It must only emit the
small structures documented here; server code owns identifiers and provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    @abstractmethod
    def plan(self, request: str, patient_record: str) -> dict[str, Any]:
        """Return only {"tasks": [{"id", "goal", "deps"}, ...]}."""

    @abstractmethod
    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> dict[str, Any]:
        """Return only {"queries": ["...", ...]}."""

    @abstractmethod
    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Return only {"facts": [{"text", "ref"}, ...]}."""

    @abstractmethod
    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
        evidence: list[dict[str, str]],
        upstream: dict[int, Any],
    ) -> dict[str, Any]:
        """Return only {"claims": [{"text", "refs"}], "unknowns": []}."""

    def judge_claim(
        self, *, claim: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> str:
        """Return SUPPORTED, NOT_SUPPORTED or UNCERTAIN.

        The default intentionally declines semantic judgement.  Deterministic
        validation remains available even without a powerful evaluator model.
        """

        return "UNCERTAIN"
