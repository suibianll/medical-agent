"""Backward-compatible imports for the OpenAI-compatible model adapter.

New code should import from :mod:`medical_agent.adapters.openai_compatible` and
:mod:`medical_agent.infrastructure.openai_client`.  This facade keeps existing
deployments and third-party imports working after the architecture split.
"""

from __future__ import annotations

from typing import Any

from .adapters.openai_compatible import (
    AliyunCompatibleModelAdapter,
    OpenAICompatibleModelAdapter,
)
from .infrastructure.openai_client import ModelProviderError, normalize_base_url
from .utils.json_tools import extract_json_object


def _extract_json(content: str) -> dict[str, Any]:
    """Compatibility alias for the provider-neutral JSON response parser."""

    return extract_json_object(content, error_type=ModelProviderError)


__all__ = [
    "AliyunCompatibleModelAdapter",
    "OpenAICompatibleModelAdapter",
    "ModelProviderError",
    "normalize_base_url",
    "_extract_json",
]
