"""Composition root for concrete runtime dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .adapters.openai_compatible import OpenAICompatibleModelAdapter
from .application.agent import MedicalAgent
from .audit_log import SafeAuditLogger
from .demo_model import DemoModelAdapter
from .infrastructure.model_config import (
    EmbeddingConfig,
    RetrievalConfig,
    load_model_configuration,
)
from .ports import AuditEventSink, KnowledgeBasePort, ModelAdapter, RunArchivePort
from .retrieval.knowledge import JsonKnowledgeBase
from .retrieval.vector import (
    FaissKnowledgeBase,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from .run_archive import InMemoryRunArchive


def default_knowledge_storage_path() -> Path:
    """Return a writable runtime-data path, never a package installation path."""

    configured = os.environ.get("MEDICAL_AGENT_DATA_DIR", "").strip()
    data_dir = Path(configured).expanduser() if configured else Path.cwd() / "data"
    return data_dir / "imported_knowledge.json"


def create_agent(
    *,
    model_profiles: Mapping[str, ModelAdapter] | None = None,
    model_profile_labels: Mapping[str, str] | None = None,
    default_model_profile: str | None = None,
    knowledge_base: KnowledgeBasePort | None = None,
    run_archive: RunArchivePort | None = None,
    audit_logger: AuditEventSink | None = None,
    verifier_model: ModelAdapter | None = None,
    max_repair_rounds: int = 2,
    max_workers: int = 3,
    retrieval_limit: int = 8,
    retrieval_candidate_budget: int = 12,
    retrieval_max_per_document: int = 2,
) -> MedicalAgent:
    """Build an explicitly configured agent for tests or embedding."""

    profiles = dict(model_profiles or {"demo": DemoModelAdapter()})
    return MedicalAgent(
        model_profiles=profiles,
        model_profile_labels=model_profile_labels,
        default_model_profile=default_model_profile,
        knowledge_base=knowledge_base
        or JsonKnowledgeBase.demo(storage_path=default_knowledge_storage_path()),
        run_archive=run_archive or InMemoryRunArchive(),
        audit_logger=audit_logger or SafeAuditLogger(),
        verifier_model=verifier_model,
        max_repair_rounds=max_repair_rounds,
        max_workers=max_workers,
        retrieval_limit=retrieval_limit,
        retrieval_candidate_budget=retrieval_candidate_budget,
        retrieval_max_per_document=retrieval_max_per_document,
    )


def _build_embedding_provider(config: EmbeddingConfig) -> Any:
    if config.provider == "openai-compatible":
        return OpenAICompatibleEmbeddingProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            dimensions=config.dimensions,
            timeout_seconds=config.timeout_seconds,
        )
    return HashEmbeddingProvider(dimensions=config.dimensions)


def _build_knowledge_base(config: RetrievalConfig) -> KnowledgeBasePort:
    storage_path = default_knowledge_storage_path()
    lexical = JsonKnowledgeBase.demo(storage_path=storage_path)
    if config.backend != "faiss":
        return lexical
    return FaissKnowledgeBase(
        lexical,
        embedding_provider=_build_embedding_provider(config.embedding),
        index_path=Path(config.index_path).expanduser(),
    )


def create_agent_from_environment() -> MedicalAgent:
    """Build the local runtime from environment/file configuration."""

    configuration = load_model_configuration()
    profiles: dict[str, ModelAdapter] = {}
    labels: dict[str, str] = {}
    for spec in configuration.profiles:
        profiles[spec.profile_id] = OpenAICompatibleModelAdapter(
            api_key=spec.api_key,
            base_url=spec.base_url,
            model=spec.model,
            provider=spec.provider,
        )
        labels[spec.profile_id] = spec.label
    profiles["demo"] = DemoModelAdapter()
    labels["demo"] = "本地演示模型"
    default_profile = (
        configuration.default_profile
        if configuration.default_profile in profiles
        else next(iter(profiles))
    )
    return create_agent(
        model_profiles=profiles,
        model_profile_labels=labels,
        default_model_profile=default_profile,
        knowledge_base=_build_knowledge_base(configuration.retrieval),
        max_workers=2 if default_profile != "demo" else 3,
        retrieval_limit=configuration.retrieval.top_k,
        retrieval_candidate_budget=configuration.retrieval.candidate_budget,
        retrieval_max_per_document=configuration.retrieval.max_per_document,
    )
