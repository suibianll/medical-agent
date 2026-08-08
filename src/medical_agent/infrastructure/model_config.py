"""Load and normalize local/environment model profile configuration.

This module parses configuration only.  It never creates clients, logs secret
values, or exposes profile credentials through public metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelProfileConfig:
    profile_id: str
    label: str
    api_key: str
    base_url: str
    model: str
    provider: str
    timeout_seconds: int = 90
    max_output_tokens: int | None = None
    enable_thinking: bool | None = None
    thinking_budget: int | None = None
    stream: bool = False
    thinking_stages: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider: str = "hash"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    dimensions: int = 256
    timeout_seconds: int = 60
    cache_size: int = 256
    cache_ttl_seconds: int = 600


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    backend: str = "lexical"
    top_k: int = 8
    candidate_budget: int = 12
    max_per_document: int = 2
    index_path: str = "data/knowledge.faiss"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    max_rounds: int = 2
    refine_on_empty: bool = True
    refine_min_candidates: int = 1
    # Kept as a normalized, read-only-by-convention mapping so the retrieval
    # layer can own policy semantics without making infrastructure depend on
    # a concrete backend implementation.
    source_policy: Mapping[str, Any] = field(default_factory=dict)
    relevance_threshold: float = 0.0
    hybrid_sparse_weight: float = 0.45
    hybrid_dense_weight: float = 0.55
    hybrid_rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    enabled: bool = False
    provider: str = "generic"
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: int = 30
    top_n: int = 8
    auth_header: str = ""
    max_calls_per_run: int = 8
    min_candidates: int = 2
    cache_size: int = 128
    cache_ttl_seconds: int = 300


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    mode: str = "evidence"
    provider: str = "generic"
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: int = 30
    send_patient_record: bool = False
    rules: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    profiles: tuple[ModelProfileConfig, ...]
    default_profile: str
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


class ModelConfigurationError(ValueError):
    """Raised when enabled runtime components cannot be constructed safely."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        errors = diagnostics.get("errors", []) if isinstance(diagnostics, dict) else []
        message = "模型运行配置校验失败。"
        if errors and isinstance(errors[0], dict):
            message = f"{message} {errors[0].get('message', '请检查配置项。')}"
        super().__init__(message)


def profile_id(value: Any, fallback: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return (candidate.strip("-._") or fallback)[:64]


def _read_local_config(environment: Mapping[str, str]) -> dict[str, Any]:
    configured_path = environment.get("MEDICAL_AGENT_CONFIG", "").strip()
    path = (
        Path(configured_path).expanduser()
        if configured_path
        else Path.cwd() / "config" / "model.local.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_secret(
    spec: Mapping[str, Any], environment: Mapping[str, str], field_name: str
) -> str:
    env_name = str(spec.get(f"{field_name}_env", "")).strip()
    if env_name:
        configured = environment.get(env_name, "").strip()
        if configured:
            return configured
    return str(spec.get(field_name, "")).strip()


def _positive_int(value: Any, fallback: int, *, minimum: int, maximum: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(candidate, maximum))


def _bounded_float(value: Any, fallback: float, *, minimum: float, maximum: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(candidate):
        return fallback
    return max(minimum, min(candidate, maximum))


def _parse_retrieval_config(
    raw: Any, environment: Mapping[str, str]
) -> RetrievalConfig:
    if not isinstance(raw, dict):
        return RetrievalConfig()
    backend = str(raw.get("backend", "lexical")).strip().lower()
    if backend not in {"lexical", "faiss", "hybrid"}:
        backend = "lexical"
    faiss_raw = raw.get("faiss")
    if not isinstance(faiss_raw, dict):
        faiss_raw = {}
    embedding_raw = raw.get("embedding")
    if not isinstance(embedding_raw, dict):
        embedding_raw = faiss_raw.get("embedding", {})
    if not isinstance(embedding_raw, dict):
        embedding_raw = {}
    source_policy_raw = raw.get("source_policy", raw.get("governance", {}))
    if not isinstance(source_policy_raw, dict):
        source_policy_raw = {}
    source_policy: dict[str, Any] = {}
    if source_policy_raw:
        def _normalise_policy_values(value: Any) -> list[str]:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                return []
            result: list[str] = []
            for item in value[:16]:
                normalized = " ".join(str(item).split()).lower()
                if normalized and normalized not in result:
                    result.append(normalized)
            return result

        source_policy["enabled"] = _parse_bool(source_policy_raw.get("enabled"), True)
        source_policy["allowed_source_types"] = _normalise_policy_values(
            source_policy_raw.get("allowed_source_types", [])
        )
        blocked_statuses = source_policy_raw.get(
            "blocked_statuses",
            ["retracted", "revoked", "unsafe", "poisoned"],
        )
        source_policy["blocked_statuses"] = _normalise_policy_values(blocked_statuses)
        source_policy["min_priority"] = _positive_int(
            source_policy_raw.get("min_priority", 0), 0, minimum=0, maximum=100
        )
        source_policy["require_version"] = _parse_bool(
            source_policy_raw.get("require_version"), False
        )
        source_policy["allow_synthetic"] = _parse_bool(
            source_policy_raw.get("allow_synthetic"), True
        )
        source_policy["max_age_days"] = _positive_int(
            source_policy_raw.get("max_age_days", 0), 0, minimum=0, maximum=36_500
        )
        source_policy["reject_unknown_date"] = _parse_bool(
            source_policy_raw.get("reject_unknown_date"), False
        )
        as_of_date = str(source_policy_raw.get("as_of_date", "")).strip()
        if as_of_date:
            source_policy["as_of_date"] = as_of_date[:40]
    embedding_provider = str(embedding_raw.get("provider", "hash")).strip().lower()
    if embedding_provider in {"openai", "openai_compatible", "openai-compatible", "http"}:
        embedding_provider = "openai-compatible"
    elif embedding_provider != "hash":
        embedding_provider = "hash"
    dimensions = _positive_int(
        embedding_raw.get("dimensions", 256), 256, minimum=8, maximum=8192
    )
    return RetrievalConfig(
        backend=backend,
        top_k=_positive_int(raw.get("top_k", 8), 8, minimum=1, maximum=64),
        candidate_budget=_positive_int(
            raw.get("candidate_budget", 12), 12, minimum=1, maximum=128
        ),
        max_per_document=_positive_int(
            raw.get("max_per_document", 2), 2, minimum=1, maximum=16
        ),
        max_rounds=_positive_int(raw.get("max_rounds", 2), 2, minimum=1, maximum=4),
        refine_on_empty=_parse_bool(raw.get("refine_on_empty"), True),
        refine_min_candidates=_positive_int(
            raw.get("refine_min_candidates", 1), 1, minimum=0, maximum=128
        ),
        relevance_threshold=_bounded_float(
            raw.get("relevance_threshold", 0.0), 0.0, minimum=0.0, maximum=1_000_000.0
        ),
        hybrid_sparse_weight=_bounded_float(
            raw.get("hybrid_sparse_weight", 0.45), 0.45, minimum=0.0, maximum=1.0
        ),
        hybrid_dense_weight=_bounded_float(
            raw.get("hybrid_dense_weight", 0.55), 0.55, minimum=0.0, maximum=1.0
        ),
        hybrid_rrf_k=_positive_int(
            raw.get("hybrid_rrf_k", 60), 60, minimum=1, maximum=1_000
        ),
        index_path=str(
            raw.get("index_path", faiss_raw.get("index_path", "data/knowledge.faiss"))
        ).strip()
        or "data/knowledge.faiss",
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            api_key=_resolve_secret(embedding_raw, environment, "api_key"),
            base_url=str(embedding_raw.get("base_url", "")).strip(),
            model=str(embedding_raw.get("model", "")).strip(),
            dimensions=dimensions,
            timeout_seconds=_positive_int(
                embedding_raw.get("timeout_seconds", 60), 60, minimum=1, maximum=300
            ),
            cache_size=_positive_int(
                embedding_raw.get("cache_size", 256), 256, minimum=0, maximum=2048
            ),
            cache_ttl_seconds=_positive_int(
                embedding_raw.get("cache_ttl_seconds", 600), 600, minimum=0, maximum=86_400
            ),
        ),
        source_policy=source_policy,
    )


def _parse_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _optional_stages(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return None
    normalized: list[str] = []
    for item in values:
        stage = str(item).strip().lower()
        if stage and stage not in normalized:
            normalized.append(stage[:40])
    return tuple(normalized)


def _parse_reranker_config(
    raw: Any, environment: Mapping[str, str]
) -> RerankerConfig:
    if not isinstance(raw, dict):
        return RerankerConfig()
    endpoint = str(raw.get("endpoint", "")).strip()
    return RerankerConfig(
        enabled=_parse_bool(raw.get("enabled"), bool(endpoint)),
        provider=str(raw.get("provider", "generic")).strip().lower() or "generic",
        endpoint=endpoint,
        api_key=_resolve_secret(raw, environment, "api_key"),
        model=str(raw.get("model", "")).strip(),
        timeout_seconds=_positive_int(
            raw.get("timeout_seconds", 30), 30, minimum=1, maximum=300
        ),
        top_n=_positive_int(raw.get("top_n", 8), 8, minimum=1, maximum=64),
        auth_header=str(raw.get("auth_header", "")).strip(),
        max_calls_per_run=_positive_int(
            raw.get("max_calls_per_run", 8), 8, minimum=0, maximum=64
        ),
        min_candidates=_positive_int(
            raw.get("min_candidates", 2), 2, minimum=1, maximum=64
        ),
        cache_size=_positive_int(
            raw.get("cache_size", 128), 128, minimum=0, maximum=1024
        ),
        cache_ttl_seconds=_positive_int(
            raw.get("cache_ttl_seconds", 300), 300, minimum=0, maximum=86_400
        ),
    )


def _parse_routing_config(
    raw: Any, environment: Mapping[str, str]
) -> RoutingConfig:
    if not isinstance(raw, dict):
        return RoutingConfig()
    mode = str(raw.get("mode", "evidence")).strip().lower()
    if mode not in {"evidence", "rules", "api"}:
        mode = "evidence"
    raw_rules = raw.get("rules", [])
    rules: list[dict[str, Any]] = []
    if isinstance(raw_rules, list):
        for raw_rule in raw_rules[:16]:
            if not isinstance(raw_rule, dict):
                continue
            outcome = str(raw_rule.get("outcome", "")).strip()
            patterns = raw_rule.get("patterns", [])
            if outcome not in {
                "answer",
                "ask_clarification",
                "defer",
                "emergency_escalation",
            }:
                continue
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list):
                continue
            normalized_patterns = [
                str(pattern)[:240]
                for pattern in patterns[:16]
                if str(pattern).strip()
            ]
            if normalized_patterns:
                rules.append(
                    {
                        "outcome": outcome,
                        "patterns": normalized_patterns,
                        "risk_level": str(raw_rule.get("risk_level", "high")),
                        "emergency_signal": bool(raw_rule.get("emergency_signal", False)),
                        "reason": str(raw_rule.get("reason", "configured_rule"))[:120],
                    }
                )
    return RoutingConfig(
        mode=mode,
        provider=str(raw.get("provider", "generic")).strip().lower() or "generic",
        endpoint=str(raw.get("endpoint", "")).strip(),
        api_key=_resolve_secret(raw, environment, "api_key"),
        model=str(raw.get("model", "")).strip(),
        timeout_seconds=_positive_int(
            raw.get("timeout_seconds", 30), 30, minimum=1, maximum=300
        ),
        send_patient_record=_parse_bool(raw.get("send_patient_record"), False),
        rules=tuple(rules),
    )


def load_model_configuration(
    environment: Mapping[str, str] | None = None,
) -> ModelConfiguration:
    env = environment if environment is not None else os.environ
    local_config = _read_local_config(env)
    profiles: dict[str, ModelProfileConfig] = {}
    configured_default = ""

    def add_profile(raw_id: Any, spec: Any, fallback: str) -> str | None:
        if not isinstance(spec, dict):
            return None
        api_key = _resolve_secret(spec, env, "api_key")
        base_url = str(spec.get("base_url", "")).strip()
        model_name = str(spec.get("model", "")).strip()
        provider = str(spec.get("provider", "")).strip().lower()
        if not (api_key and base_url and model_name and provider):
            return None
        normalized_id = profile_id(raw_id, fallback)
        profiles[normalized_id] = ModelProfileConfig(
            profile_id=normalized_id,
            label=str(spec.get("label", model_name)).strip()[:80] or model_name,
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            provider=provider[:80],
            timeout_seconds=_positive_int(
                spec.get("timeout_seconds", 90), 90, minimum=1, maximum=600
            ),
            max_output_tokens=(
                _positive_int(
                    spec.get("max_output_tokens"),
                    4096,
                    minimum=512,
                    maximum=32_768,
                )
                if spec.get("max_output_tokens") is not None
                else None
            ),
            enable_thinking=_optional_bool(spec.get("enable_thinking")),
            thinking_budget=(
                _positive_int(
                    spec.get("thinking_budget"),
                    4096,
                    minimum=128,
                    maximum=32_768,
                )
                if spec.get("thinking_budget") is not None
                else None
            ),
            stream=_parse_bool(spec.get("stream"), False),
            thinking_stages=_optional_stages(spec.get("thinking_stages")),
        )
        return normalized_id

    raw_profiles = local_config.get("profiles")
    if isinstance(raw_profiles, list):
        for index, spec in enumerate(raw_profiles, start=1):
            raw_id = spec.get("id") if isinstance(spec, dict) else ""
            add_profile(raw_id, spec, f"profile-{index}")

    raw_default = local_config.get("default_profile", "")
    if isinstance(raw_default, str) and raw_default in profiles:
        configured_default = raw_default

    env_api_key = env.get("MEDICAL_AGENT_API_KEY", "").strip()
    env_base_url = env.get("MEDICAL_AGENT_BASE_URL", "").strip()
    env_model = env.get("MEDICAL_AGENT_MODEL", "").strip()
    env_provider = env.get("MEDICAL_AGENT_PROVIDER", "").strip().lower()
    if env_api_key and env_base_url and env_model and env_provider:
        env_id = profile_id(env.get("MEDICAL_AGENT_PROFILE", "environment"), "environment")
        profiles[env_id] = ModelProfileConfig(
            profile_id=env_id,
            label=env_model[:80],
            api_key=env_api_key,
            base_url=env_base_url,
            model=env_model,
            provider=env_provider[:80],
            timeout_seconds=_positive_int(
                env.get("MEDICAL_AGENT_MODEL_TIMEOUT_SECONDS", 90),
                90,
                minimum=1,
                maximum=600,
            ),
            max_output_tokens=(
                _positive_int(
                    env.get("MEDICAL_AGENT_MAX_OUTPUT_TOKENS"),
                    4096,
                    minimum=512,
                    maximum=32_768,
                )
                if env.get("MEDICAL_AGENT_MAX_OUTPUT_TOKENS") is not None
                else None
            ),
            enable_thinking=_optional_bool(env.get("MEDICAL_AGENT_ENABLE_THINKING")),
            thinking_budget=(
                _positive_int(
                    env.get("MEDICAL_AGENT_THINKING_BUDGET"),
                    4096,
                    minimum=128,
                    maximum=32_768,
                )
                if env.get("MEDICAL_AGENT_THINKING_BUDGET") is not None
                else None
            ),
            stream=_parse_bool(env.get("MEDICAL_AGENT_STREAM"), False),
            thinking_stages=_optional_stages(env.get("MEDICAL_AGENT_THINKING_STAGES")),
        )
        configured_default = env_id

    if not configured_default and profiles:
        configured_default = next(iter(profiles))
    return ModelConfiguration(
        tuple(profiles.values()),
        configured_default,
        retrieval=_parse_retrieval_config(local_config.get("retrieval"), env),
        reranker=_parse_reranker_config(local_config.get("reranker"), env),
        routing=_parse_routing_config(local_config.get("routing"), env),
    )


def validate_model_configuration(configuration: ModelConfiguration) -> dict[str, Any]:
    """Validate enabled integrations before the composition root builds clients.

    The result is safe to expose in diagnostics: it contains field paths and
    remediation hints only, never endpoint values, API keys or local records.
    """

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def error(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    def warning(code: str, message: str) -> None:
        warnings.append({"code": code, "message": message})

    def validate_endpoint(field: str, value: str) -> None:
        if value and not value.startswith(("http://", "https://")):
            error(
                "INVALID_ENDPOINT",
                f"{field} 必须以 http:// 或 https:// 开头。",
            )

    retrieval = configuration.retrieval
    if retrieval.backend in {"faiss", "hybrid"}:
        embedding = retrieval.embedding
        if embedding.provider == "openai-compatible":
            if not embedding.api_key:
                error(
                    "EMBEDDING_API_KEY_REQUIRED",
                    "retrieval.embedding.api_key（或 api_key_env）不能为空。",
                )
            if not embedding.base_url:
                error(
                    "EMBEDDING_BASE_URL_REQUIRED",
                    "retrieval.embedding.base_url 不能为空。",
                )
            else:
                validate_endpoint("retrieval.embedding.base_url", embedding.base_url)
            if not embedding.model:
                error(
                    "EMBEDDING_MODEL_REQUIRED",
                    "retrieval.embedding.model 不能为空。",
                )
        elif embedding.provider != "hash":
            error(
                "EMBEDDING_PROVIDER_UNSUPPORTED",
                "retrieval.embedding.provider 必须是 hash 或 openai-compatible。",
            )

    reranker = configuration.reranker
    if reranker.enabled:
        if not reranker.endpoint:
            error("RERANKER_ENDPOINT_REQUIRED", "启用 reranker 时必须配置 reranker.endpoint。")
        else:
            validate_endpoint("reranker.endpoint", reranker.endpoint)
        if not reranker.api_key:
            warning(
                "RERANKER_API_KEY_EMPTY",
                "reranker 未配置 api_key；仅适用于无需认证的服务。",
            )
    elif reranker.endpoint:
        warning(
            "RERANKER_DISABLED",
            "已配置 reranker.endpoint 但 enabled=false，运行时将忽略外部重排。",
        )

    routing = configuration.routing
    if routing.mode == "api":
        if not routing.endpoint:
            error("ROUTER_ENDPOINT_REQUIRED", "routing.mode=api 时必须配置 routing.endpoint。")
        else:
            validate_endpoint("routing.endpoint", routing.endpoint)
        if not routing.api_key:
            warning(
                "ROUTER_API_KEY_EMPTY",
                "外部路由未配置 api_key；请确认服务是否允许匿名访问。",
            )

    if not configuration.profiles:
        warning(
            "DEMO_MODEL_FALLBACK",
            "未配置真实模型 profile，组合根将仅提供本地演示模型。",
        )
    return {
        "valid": not errors,
        "errors": errors[:32],
        "warnings": warnings[:32],
    }
