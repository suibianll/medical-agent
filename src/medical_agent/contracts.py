"""Typed data contracts shared across the application boundary.

The runtime still exchanges ordinary dictionaries so the HTTP/JSON surface
stays lightweight.  TypedDict makes required shapes explicit to static tools
and keeps field names centralized instead of relying on ``dict[str, Any]``
throughout the workflow.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ModelMetadata(TypedDict):
    mode: str
    provider: str
    name: str


class ModelProfileMetadata(ModelMetadata):
    profile: str


class ModelCallMetrics(TypedDict, total=False):
    stage: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    success: bool


class EvidenceSpan(TypedDict, total=False):
    id: str
    evidence_id: str
    start: int
    end: int
    page: int
    bbox: list[float]
    table_row: int
    table_col: int
    confidence: float
    granularity: str


class SupportEdge(TypedDict, total=False):
    claim_id: str
    evidence_id: str
    span_id: str
    relation: str
    verifier: str
    verifier_score: float
    status: str


class PlanTask(TypedDict, total=False):
    id: int
    goal: str
    deps: list[int]
    evidence_scope: str
    analysis_mode: str


class PlanPayload(TypedDict):
    tasks: list[PlanTask]


class QueryPayload(TypedDict):
    queries: list[str]


class ExtractedFact(TypedDict):
    text: str
    ref: str


class FactPayload(TypedDict):
    facts: list[ExtractedFact]


class SynthesisPayload(TypedDict):
    claims: list["Claim"]
    unknowns: list[str]


class Claim(TypedDict, total=False):
    id: str
    task_id: int
    text: str
    refs: list[str]
    status: str
    issues: list[str]
    cited_text: str
    support_edges: list[SupportEdge]


class RunResult(TypedDict, total=False):
    status: str
    run: dict[str, Any]
    report: dict[str, Any]
    graph: dict[str, Any]
    evidence: list[dict[str, Any]]
    claims: list[Claim]
    errors: list[dict[str, Any]]
    answer: str
    mode: str
    archive: dict[str, Any]
    model_calls: list[ModelCallMetrics]
    model_usage: dict[str, Any]
    retrieval_usage: dict[str, Any]
    quality: dict[str, Any]
    decision: dict[str, Any]
