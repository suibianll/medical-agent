"""Retrieval adapters used by the local demo.

The interfaces are deliberately small so an actual vector store, FHIR source,
or hospital knowledge base can replace these demo implementations later.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def _normalise(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def _tokens(value: str) -> set[str]:
    compact = _normalise(value)
    latin = set(re.findall(r"[a-z0-9._-]{2,}", compact))
    chinese_bigrams = {compact[index : index + 2] for index in range(len(compact) - 1)}
    return latin | chinese_bigrams


def _score(query: str, text: str, keywords: list[str] | None = None) -> int:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    score = len(query_tokens & text_tokens)
    query_compact = _normalise(query)
    for keyword in keywords or []:
        if _normalise(keyword) and _normalise(keyword) in query_compact:
            score += 4
    return score


class PatientRecordRetriever:
    def __init__(self, patient_record: str) -> None:
        self.patient_record = patient_record.strip()
        raw_segments = [
            segment.strip()
            for segment in re.split(r"[\r\n。；;]+", self.patient_record)
            if segment.strip()
        ]
        self.segments = raw_segments or ([self.patient_record] if self.patient_record else [])

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        ranked = [
            {
                "text": segment,
                "locator": f"病历片段 {index}",
                "score": _score(query, segment),
            }
            for index, segment in enumerate(self.segments, start=1)
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = ranked[:limit]
        # Preserve a usable patient fact even when lexical overlap is weak.
        return [item for item in selected if item["text"]]


class JsonKnowledgeBase:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    @classmethod
    def demo(cls) -> "JsonKnowledgeBase":
        data_path = Path(__file__).resolve().parents[2] / "data" / "knowledge.json"
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload["documents"])

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for document in self.documents:
            text = f"{document.get('title', '')} {document.get('text', '')}"
            score = _score(query, text, document.get("keywords", []))
            ranked.append({**document, "score": score})

        ranked.sort(key=lambda item: (item["score"], item.get("priority", 0)), reverse=True)
        # Demo corpus is intentionally small; return fallback evidence so the
        # pipeline can demonstrate its structured no-evidence path only when
        # there truly is no corpus at all.
        return ranked[:limit]
