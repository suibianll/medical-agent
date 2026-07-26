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


class PlanTask(TypedDict):
    id: int
    goal: str
    deps: list[int]


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


class EvidenceItem(TypedDict, total=False):
    id: str
    source_type: str
    title: str
    text: str
    locator: str
    url: str
    version: str
    score: float


class Claim(TypedDict, total=False):
    id: str
    task_id: int
    text: str
    refs: list[str]
    status: str
    issues: list[str]
    cited_text: str


class EvaluationIssue(TypedDict, total=False):
    claim: str
    task_id: int
    code: str


EvaluationResult = TypedDict(
    "EvaluationResult",
    {
        "pass": bool,
        "issues": list[EvaluationIssue],
        "judgements": list[dict[str, Any]],
    },
)


class ProgressEvent(TypedDict, total=False):
    run_id: str
    sequence: int
    timestamp: str
    stage: str
    message: str
    status: str
    task_id: int
    task: PlanTask
    tasks: list[PlanTask]
    round: int
    counts: dict[str, int]
    evaluation: dict[str, Any]
    evidence_ids: list[str]
    facts: list[dict[str, Any]]
    claims: list[Claim]


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
