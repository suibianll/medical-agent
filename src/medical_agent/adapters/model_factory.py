"""Create model adapters from normalized configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .openai_compatible import (
    AliyunCompatibleModelAdapter,
    OpenAICompatibleModelAdapter,
)
from ..demo_model import DemoModelAdapter
from ..infrastructure.model_config import ModelConfiguration
from ..model_adapter import ModelAdapter


@dataclass(slots=True)
class ModelRuntime:
    profiles: dict[str, ModelAdapter]
    labels: dict[str, str]
    default_profile: str


def create_model_runtime(configuration: ModelConfiguration) -> ModelRuntime:
    profiles: dict[str, ModelAdapter] = {}
    labels: dict[str, str] = {}
    for spec in configuration.profiles:
        adapter_type = (
            AliyunCompatibleModelAdapter
            if spec.provider == "aliyun-model-studio"
            else OpenAICompatibleModelAdapter
        )
        profiles[spec.profile_id] = adapter_type(
            api_key=spec.api_key,
            base_url=spec.base_url,
            model=spec.model,
            provider=spec.provider,
        )
        labels[spec.profile_id] = spec.label

    if not profiles:
        return ModelRuntime({"demo": DemoModelAdapter()}, {"demo": "本地演示模型"}, "demo")

    profiles["demo"] = DemoModelAdapter()
    labels["demo"] = "本地演示模型"
    default_profile = (
        configuration.default_profile
        if configuration.default_profile in profiles
        else next(iter(profiles))
    )
    return ModelRuntime(profiles, labels, default_profile)
