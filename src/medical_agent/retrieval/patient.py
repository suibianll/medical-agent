"""Patient-record retrieval for the local text-input workflow."""

from __future__ import annotations

import re
from typing import Any

from .scoring import normalized_score, score


class PatientRecordRetriever:
    def __init__(self, patient_record: str) -> None:
        self.patient_record = patient_record.strip()
        raw_segments = [
            segment.strip()
            for segment in re.split(r"[\r\n。；;]+", self.patient_record)
            if segment.strip()
        ]
        self.segments = raw_segments or (
            [self.patient_record] if self.patient_record else []
        )

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        ranked = [
            {
                "text": segment,
                "locator": f"病历片段 {index}",
                "score": score(query, segment),
                "relevance_score": normalized_score(query, segment),
            }
            for index, segment in enumerate(self.segments, start=1)
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = ranked[:limit]
        # Preserve a usable patient fact even when lexical overlap is weak.
        return [item for item in selected if item["text"]]
