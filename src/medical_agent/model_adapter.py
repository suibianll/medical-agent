"""Model adapter contract.

An API-backed adapter may implement this protocol later.  It must only emit the
small structures documented here; server code owns identifiers and provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    def runtime_metadata(self) -> dict[str, str]:
        """Return safe runtime identity for the health endpoint.

        Implementations must not put endpoint URLs, API keys, request payloads
        or provider diagnostics in this value.
        """

        return {
            "mode": "demo",
            "provider": "local-demo",
            "name": "demo",
        }

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
    ) -> dict[str, Any]:
        """Return only {"claims": [{"text", "refs"}], "unknowns": []}."""

    @abstractmethod
    def judge_claims(self, items: list[dict[str, Any]]) -> dict[str, str]:
        """Return one semantic verdict per supplied claim ID."""
