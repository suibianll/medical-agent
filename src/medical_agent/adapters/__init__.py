"""Adapters that connect model protocols to external providers."""

from .openai_compatible import OpenAICompatibleModelAdapter

__all__ = ["OpenAICompatibleModelAdapter"]
