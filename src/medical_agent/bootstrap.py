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
    RerankerConfig,
    RetrievalConfig,
    RoutingConfig,
    load_model_configuration,
    validate_model_configuration,
    ModelConfigurationError,
)
from .ports import AuditEventSink, KnowledgeBasePort, ModelAdapter, RunArchivePort
from .retrieval.knowledge import JsonKnowledgeBase
from .retrieval.governance import SourceGovernancePolicy
from .retrieval.hybrid import HybridKnowledgeBase
from .retrieval.vector import (
    FaissKnowledgeBase,
    CachedEmbeddingProvider,
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from .retrieval.reranker import ExternalApiReranker
from .run_archive import InMemoryRunArchive
from .risk import EvidenceDecisionRouter, ExternalApiDecisionRouter, PatternDecisionRouter


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
    retrieval_max_rounds: int = 2,
    retrieval_refine_on_empty: bool = True,
    retrieval_refine_min_candidates: int = 1,
    retrieval_relevance_threshold: float = 0.0,
    reranker: Any | None = None,
    reranker_max_calls_per_run: int = 8,
    reranker_min_candidates: int = 2,
    reranker_cache_size: int = 128,
    reranker_cache_ttl_seconds: int = 300,
    decision_router: Any | None = None,
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
        retrieval_max_rounds=retrieval_max_rounds,
        retrieval_refine_on_empty=retrieval_refine_on_empty,
        retrieval_refine_min_candidates=retrieval_refine_min_candidates,
        retrieval_relevance_threshold=retrieval_relevance_threshold,
        reranker=reranker,
        reranker_max_calls_per_run=reranker_max_calls_per_run,
        reranker_min_candidates=reranker_min_candidates,
        reranker_cache_size=reranker_cache_size,
        reranker_cache_ttl_seconds=reranker_cache_ttl_seconds,
        decision_router=decision_router,
    )


def _build_embedding_provider(config: EmbeddingConfig) -> Any:
    if config.provider == "openai-compatible":
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            dimensions=config.dimensions,
            timeout_seconds=config.timeout_seconds,
        )
    else:
        provider = HashEmbeddingProvider(dimensions=config.dimensions)
    return CachedEmbeddingProvider(
        provider,
        cache_size=config.cache_size,
        cache_ttl_seconds=config.cache_ttl_seconds,
    )


def _build_knowledge_base(config: RetrievalConfig) -> KnowledgeBasePort:
    storage_path = default_knowledge_storage_path()
    governance_policy = SourceGovernancePolicy.from_config(config.source_policy)
    lexical = JsonKnowledgeBase.demo(
        storage_path=storage_path,
        governance_policy=governance_policy,
    )
    if config.backend == "lexical":
        return lexical
    dense = FaissKnowledgeBase(
        lexical,
        embedding_provider=_build_embedding_provider(config.embedding),
        index_path=Path(config.index_path).expanduser(),
    )
    if config.backend == "faiss":
        return dense
    return HybridKnowledgeBase(
        lexical,
        dense,
        sparse_weight=config.hybrid_sparse_weight,
        dense_weight=config.hybrid_dense_weight,
        rrf_k=config.hybrid_rrf_k,
    )


def _build_reranker(config: RerankerConfig) -> ExternalApiReranker | None:
    if not config.enabled:
        return None
    return ExternalApiReranker(
        endpoint=config.endpoint,
        api_key=config.api_key,
        model=config.model,
        provider=config.provider,
        timeout_seconds=config.timeout_seconds,
        top_n=config.top_n,
        auth_header=config.auth_header,
    )


def _build_decision_router(config: RoutingConfig) -> Any:
    if config.mode == "api":
        return ExternalApiDecisionRouter(
            endpoint=config.endpoint,
            api_key=config.api_key,
            provider=config.provider,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            send_patient_record=config.send_patient_record,
        )
    if config.mode == "rules":
        return PatternDecisionRouter(config.rules)
    return EvidenceDecisionRouter()


def create_agent_from_environment(
    *,
    knowledge_base: KnowledgeBasePort | None = None,
    max_repair_rounds: int = 2,
) -> MedicalAgent:
    """Build the local runtime from environment/file configuration."""

    configuration = load_model_configuration()
    diagnostics = validate_model_configuration(configuration)
    if not diagnostics["valid"]:
        raise ModelConfigurationError(diagnostics)
    profiles: dict[str, ModelAdapter] = {}
    labels: dict[str, str] = {}
    for spec in configuration.profiles:
        profiles[spec.profile_id] = OpenAICompatibleModelAdapter(
            api_key=spec.api_key,
            base_url=spec.base_url,
            model=spec.model,
            provider=spec.provider,
            timeout_seconds=spec.timeout_seconds,
            max_output_tokens=spec.max_output_tokens,
            enable_thinking=spec.enable_thinking,
            thinking_budget=spec.thinking_budget,
            stream=spec.stream,
            thinking_stages=spec.thinking_stages,
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
        knowledge_base=knowledge_base or _build_knowledge_base(configuration.retrieval),
        max_repair_rounds=max_repair_rounds,
        max_workers=2 if default_profile != "demo" else 3,
        retrieval_limit=configuration.retrieval.top_k,
        retrieval_candidate_budget=configuration.retrieval.candidate_budget,
        retrieval_max_per_document=configuration.retrieval.max_per_document,
        retrieval_max_rounds=configuration.retrieval.max_rounds,
        retrieval_refine_on_empty=configuration.retrieval.refine_on_empty,
        retrieval_refine_min_candidates=configuration.retrieval.refine_min_candidates,
        retrieval_relevance_threshold=configuration.retrieval.relevance_threshold,
        reranker=_build_reranker(configuration.reranker),
        reranker_max_calls_per_run=configuration.reranker.max_calls_per_run,
        reranker_min_candidates=configuration.reranker.min_candidates,
        reranker_cache_size=configuration.reranker.cache_size,
        reranker_cache_ttl_seconds=configuration.reranker.cache_ttl_seconds,
        decision_router=_build_decision_router(configuration.routing),
    )
