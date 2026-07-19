"""Server-owned evidence registry.

Models only receive a projection of an evidence item: id, text and source.
The registry retains provenance fields needed for citations, audit and graphing.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable


class EvidenceRegistry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._dedupe: dict[tuple[str, str, str, str], str] = {}
        self._counters = {"patient": 0, "knowledge": 0}

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
        if kind not in self._counters:
            raise ValueError(f"未知证据类型：{kind}")
        key = (kind, text.strip(), source.strip(), locator.strip())
        existing_id = self._dedupe.get(key)
        if existing_id:
            return deepcopy(self._items[existing_id])

        self._counters[kind] += 1
        prefix = "P" if kind == "patient" else "K"
        evidence_id = f"{prefix}{self._counters[kind]}"
        item = {
            "id": evidence_id,
            "kind": kind,
            "text": text.strip(),
            "source": source.strip(),
            "locator": locator.strip(),
            "document_id": document_id,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
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
        item = self._items.get(evidence_id)
        return deepcopy(item) if item else None

    def all(self) -> list[dict[str, Any]]:
        return [deepcopy(item) for item in self._items.values()]

    def as_map(self) -> dict[str, dict[str, Any]]:
        return {key: deepcopy(value) for key, value in self._items.items()}

    def model_view(self, evidence_ids: Iterable[str] | None = None) -> list[dict[str, str]]:
        """Return the compact projection that may be inserted into a model prompt."""

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
