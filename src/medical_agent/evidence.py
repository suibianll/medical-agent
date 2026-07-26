"""Server-owned evidence registry.

Models only receive a projection of an evidence item: id, text and source.
The registry retains provenance fields needed for citations, audit and graphing.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from threading import RLock
from typing import Any, Iterable


class EvidenceRegistry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._dedupe: dict[tuple[str, str, str, str], str] = {}
        self._counters = {"patient": 0, "knowledge": 0}
        # A single registry is shared by all workers in one DAG run.  Protect
        # dedupe lookup, ID allocation and insertion as one atomic operation.
        self._lock = RLock()

    def _add(
        self,
        *,
        kind: str,
        text: str,
        source: str,
        locator: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if kind not in self._counters:
                raise ValueError(f"未知证据类型：{kind}")
            key = (kind, text.strip(), source.strip(), locator.strip())
            existing_id = self._dedupe.get(key)
            if existing_id:
                return deepcopy(self._items[existing_id])

            self._counters[kind] += 1
            prefix = "P" if kind == "patient" else "K"
            evidence_id = f"{prefix}{self._counters[kind]}"
            normalized_text = text.strip()
            content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            item = {
                "id": evidence_id,
                "kind": kind,
                "text": normalized_text,
                "source": source.strip(),
                "locator": locator.strip(),
                "document_id": document_id,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
                "span": {
                    "id": f"{evidence_id}:span:0",
                    "evidence_id": evidence_id,
                    "start": 0,
                    "end": len(normalized_text),
                    "granularity": "chunk",
                },
                "metadata": metadata or {},
            }
            self._items[evidence_id] = item
            self._dedupe[key] = evidence_id
            return deepcopy(item)

    def add_patient(
        self,
        text: str,
        *,
        source: str = "患者病历",
        locator: str = "病历原文",
        document_id: str = "patient-input",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._add(
            kind="patient",
            text=text,
            source=source,
            locator=locator,
            document_id=document_id,
            metadata=metadata,
        )

    def add_knowledge(
        self,
        text: str,
        *,
        source: str,
        locator: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._add(
            kind="knowledge",
            text=text,
            source=source,
            locator=locator,
            document_id=document_id,
            metadata=metadata,
        )

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(evidence_id)
            return deepcopy(item) if item else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in self._items.values()]

    def as_map(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: deepcopy(value) for key, value in self._items.items()}

    def model_view(self, evidence_ids: Iterable[str] | None = None) -> list[dict[str, str]]:
        """Return the compact projection that may be inserted into a model prompt."""

        with self._lock:
            ids = list(evidence_ids) if evidence_ids is not None else list(self._items)
            result: list[dict[str, str]] = []
            for evidence_id in ids:
                item = self._items.get(evidence_id)
                if item:
                    result.append(
                        {
                            "id": item["id"],
                            "text": item["text"],
                            "source": item["source"],
                        }
                    )
            return result
