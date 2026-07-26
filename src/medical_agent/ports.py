"""Application ports implemented by adapters and infrastructure."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import FactPayload, PlanPayload, QueryPayload, SynthesisPayload


class ModelAdapter(Protocol):
    def runtime_metadata(self) -> dict[str, str]: ...

    def plan(self, request: str, patient_record: str) -> PlanPayload: ...

    def make_queries(
        self,
        *,
        task: dict[str, Any],
        request: str,
        patient_record: str,
        upstream: dict[int, Any],
    ) -> QueryPayload: ...

    def extract_facts(
        self, *, task: dict[str, Any], evidence: list[dict[str, str]]
    ) -> FactPayload: ...

    def synthesize(
        self,
        *,
        task: dict[str, Any],
        request: str,
        facts: list[dict[str, str]],
    ) -> SynthesisPayload: ...

    def judge_claims(self, items: list[dict[str, Any]]) -> dict[str, str]: ...


class KnowledgeBasePort(Protocol):
    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        """Return ranked, provenance-preserving knowledge chunks."""

    def import_text(self, *, name: str, content: str) -> dict[str, Any]:
        """Import one authorized local knowledge source."""

    def list_documents(self) -> list[dict[str, Any]]:
        """Return document metadata without source text."""


class RerankerPort(Protocol):
    def rerank(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Return a bounded, provenance-preserving reranked candidate list."""

    def runtime_metadata(self) -> dict[str, str]:
        """Return safe provider identity without credentials or request text."""


class RunArchivePort(Protocol):
    def start(self, run_id: str, created_at: str | None = None) -> dict[str, Any]: ...

    def append_event(self, event: dict[str, Any]) -> None: ...

    def finalize(self, result: dict[str, Any]) -> dict[str, Any] | None: ...

    def mark_failed(self, run_id: str) -> None: ...

    def get_result(self, run_id: str) -> dict[str, Any] | None: ...

    def get_events(self, run_id: str) -> dict[str, Any] | None: ...


class AuditEventSink(Protocol):
    def record(self, event: dict[str, Any]) -> None:
        """Record one already-sanitized execution event."""
