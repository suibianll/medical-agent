"""Retrieval adapters exposed through a stable package-level API."""

from .knowledge import (
    CHUNK_CHARS,
    MAX_IMPORTED_DOCUMENT_CHARS,
    MAX_IMPORTED_DOCUMENTS_PER_REQUEST,
    JsonKnowledgeBase,
    KnowledgeImportError,
)
from .patient import PatientRecordRetriever

__all__ = [
    "CHUNK_CHARS",
    "MAX_IMPORTED_DOCUMENT_CHARS",
    "MAX_IMPORTED_DOCUMENTS_PER_REQUEST",
    "JsonKnowledgeBase",
    "KnowledgeImportError",
    "PatientRecordRetriever",
]
