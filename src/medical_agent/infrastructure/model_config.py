"""Load and normalize local/environment model profile configuration.

This module parses configuration only.  It never creates clients, logs secret
values, or exposes profile credentials through public metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    profiles: tuple[ModelProfileConfig, ...]
    default_profile: str


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
        api_key = str(spec.get("api_key", "")).strip()
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
        )
        configured_default = env_id

    if not configured_default and profiles:
        configured_default = next(iter(profiles))
    return ModelConfiguration(tuple(profiles.values()), configured_default)
