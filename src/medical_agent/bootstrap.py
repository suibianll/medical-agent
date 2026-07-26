"""Composition root for concrete runtime dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .adapters.model_factory import create_model_runtime
from .audit_log import AuditEventSink, SafeAuditLogger
from .demo_model import DemoModelAdapter
from .infrastructure.model_config import load_model_configuration
from .model_adapter import ModelAdapter
from .ports import KnowledgeBasePort, RunArchivePort
from .retrieval import JsonKnowledgeBase
from .run_archive import InMemoryRunArchive
from .service import MedicalAgentService


def default_knowledge_storage_path() -> Path:
    """Return a writable runtime-data path, never a package installation path."""

    configured = os.environ.get("MEDICAL_AGENT_DATA_DIR", "").strip()
    data_dir = Path(configured).expanduser() if configured else Path.cwd() / "data"
    return data_dir / "imported_knowledge.json"


def create_service(
    *,
    model_profiles: Mapping[str, ModelAdapter] | None = None,
    model_profile_labels: Mapping[str, str] | None = None,
    default_model_profile: str | None = None,
    knowledge_base: KnowledgeBasePort | None = None,
    run_archive: RunArchivePort | None = None,
    audit_logger: AuditEventSink | None = None,
    max_repair_rounds: int = 2,
    max_workers: int = 3,
) -> MedicalAgentService:
    """Build an explicitly configured service for tests or embedding."""

    profiles = dict(model_profiles or {"demo": DemoModelAdapter()})
    return MedicalAgentService(
        model_profiles=profiles,
        model_profile_labels=model_profile_labels,
        default_model_profile=default_model_profile,
        knowledge_base=knowledge_base
        or JsonKnowledgeBase.demo(storage_path=default_knowledge_storage_path()),
        run_archive=run_archive or InMemoryRunArchive(),
        audit_logger=audit_logger or SafeAuditLogger(),
        max_repair_rounds=max_repair_rounds,
        max_workers=max_workers,
    )


def create_service_from_environment() -> MedicalAgentService:
    """Build the local runtime from environment/file configuration."""

    runtime = create_model_runtime(load_model_configuration())
    return create_service(
        model_profiles=runtime.profiles,
        model_profile_labels=runtime.labels,
        default_model_profile=runtime.default_profile,
        max_workers=2 if runtime.default_profile != "demo" else 3,
    )
