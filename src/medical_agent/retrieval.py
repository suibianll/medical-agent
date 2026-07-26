"""Retrieval adapters used by the local demo.

The interfaces are deliberately small so an actual vector store, FHIR source,
or hospital knowledge base can replace these demo implementations later.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


MAX_IMPORTED_DOCUMENT_CHARS = 1_000_000
MAX_IMPORTED_DOCUMENTS_PER_REQUEST = 32
CHUNK_CHARS = 900


class KnowledgeImportError(ValueError):
    """Raised when a local knowledge import cannot be safely indexed."""


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
    """Thread-safe local knowledge base with persistent, user-imported documents.

    Built-in demo documents remain read-only. Imported text is chunked locally
    and saved separately so it can be used after a server restart without ever
    being mixed into the source-controlled demo corpus.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        imported_documents: list[dict[str, Any]] | None = None,
        storage_path: Path | None = None,
    ) -> None:
        self._base_documents = deepcopy(documents)
        self._imported_documents = deepcopy(imported_documents or [])
        self._storage_path = storage_path
        self._lock = RLock()

    @property
    def documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._base_documents + self._imported_documents)

    @classmethod
    def demo(cls) -> "JsonKnowledgeBase":
        data_dir = Path(__file__).resolve().parents[2] / "data"
        data_path = data_dir / "knowledge.json"
        storage_path = data_dir / "imported_knowledge.json"
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        imported_documents: list[dict[str, Any]] = []
        if storage_path.is_file():
            try:
                with storage_path.open("r", encoding="utf-8") as handle:
                    imported_payload = json.load(handle)
                candidate = imported_payload.get("documents", [])
                if isinstance(candidate, list):
                    imported_documents = [
                        item for item in candidate if isinstance(item, dict) and item.get("text")
                    ]
            except (OSError, json.JSONDecodeError):
                # A malformed local import must not prevent the service from
                # starting. The user can re-import the source document.
                imported_documents = []
        return cls(
            payload["documents"],
            imported_documents=imported_documents,
            storage_path=storage_path,
        )

    @staticmethod
    def _safe_name(name: str) -> str:
        candidate = Path(str(name or "")).name.strip()
        candidate = re.sub(r"[\x00-\x1f]", "", candidate)
        return candidate[:120] or "导入资料"

    @staticmethod
    def _split_text(content: str) -> list[str]:
        text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        chunks: list[str] = []
        while text:
            if len(text) <= CHUNK_CHARS:
                chunks.append(text)
                break
            breakpoint = max(
                text.rfind("\n", 0, CHUNK_CHARS),
                text.rfind("。", 0, CHUNK_CHARS),
                text.rfind(".", 0, CHUNK_CHARS),
                text.rfind("；", 0, CHUNK_CHARS),
            )
            if breakpoint < CHUNK_CHARS // 3:
                breakpoint = CHUNK_CHARS
            else:
                breakpoint += 1
            chunks.append(text[:breakpoint].strip())
            text = text[breakpoint:].strip()
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _entries_from_content(name: str, content: str) -> list[tuple[str, str]]:
        source_name = JsonKnowledgeBase._safe_name(name)
        stripped = content.strip()
        if not stripped:
            raise KnowledgeImportError("知识库内容不能为空。")

        parse_as_json = source_name.lower().endswith(".json") or stripped.startswith(("{", "["))
        if parse_as_json:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                if source_name.lower().endswith(".json"):
                    raise KnowledgeImportError("JSON 知识文件格式无效。") from None
            else:
                documents = payload.get("documents") if isinstance(payload, dict) else payload
                if isinstance(documents, list):
                    entries: list[tuple[str, str]] = []
                    for index, item in enumerate(documents[:MAX_IMPORTED_DOCUMENTS_PER_REQUEST], start=1):
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text", "")
                        title = item.get("title", f"{source_name} {index}")
                        if isinstance(text, str) and text.strip():
                            entries.append((JsonKnowledgeBase._safe_name(str(title)), text.strip()))
                    if entries:
                        return entries
                if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                    return [(JsonKnowledgeBase._safe_name(str(payload.get("title", source_name))), payload["text"].strip())]

        return [(source_name, stripped)]

    def _persist_imports(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._storage_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"documents": self._imported_documents}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self._storage_path)

    def import_text(self, *, name: str, content: str) -> dict[str, Any]:
        """Import local text/Markdown/JSON as retrievable, provenance-preserving chunks."""

        if not isinstance(content, str):
            raise KnowledgeImportError("知识库内容必须是文本。")
        if len(content) > MAX_IMPORTED_DOCUMENT_CHARS:
            raise KnowledgeImportError(
                f"单次导入不能超过 {MAX_IMPORTED_DOCUMENT_CHARS:,} 个字符。"
            )

        entries = self._entries_from_content(name, content)
        created_at = datetime.now(timezone.utc).isoformat()
        imported: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for title, text in entries:
            parent_id = f"import-{uuid4().hex[:12]}"
            chunks = self._split_text(text)
            if not chunks:
                continue
            summaries.append(
                {
                    "id": parent_id,
                    "name": title,
                    "kind": "imported",
                    "chunks": len(chunks),
                    "characters": len(text),
                    "imported_at": created_at,
                }
            )
            for index, chunk in enumerate(chunks, start=1):
                imported.append(
                    {
                        "id": f"{parent_id}-{index}",
                        "document_id": parent_id,
                        "title": title,
                        "text": chunk,
                        "locator": f"导入文档片段 {index}",
                        "url": "",
                        "version": f"local-import-{created_at[:10]}",
                        "keywords": [],
                        "priority": 50,
                        "synthetic": False,
                        "imported_at": created_at,
                        "source_type": "imported",
                    }
                )

        if not imported:
            raise KnowledgeImportError("未能从导入内容中提取可检索文本。")

        with self._lock:
            self._imported_documents.extend(imported)
            self._persist_imports()
        return {"documents": summaries, "chunks_added": len(imported)}

    def list_documents(self) -> list[dict[str, Any]]:
        """Return compact metadata; avoid returning imported document text by default."""

        with self._lock:
            built_in = [
                {
                    "id": document.get("id"),
                    "name": document.get("title", document.get("id", "知识库文档")),
                    "kind": "built_in",
                    "chunks": 1,
                    "characters": len(str(document.get("text", ""))),
                    "version": document.get("version", "未标注"),
                }
                for document in self._base_documents
            ]
            imported: dict[str, dict[str, Any]] = {}
            for document in self._imported_documents:
                parent_id = str(document.get("document_id", document.get("id")))
                summary = imported.setdefault(
                    parent_id,
                    {
                        "id": parent_id,
                        "name": document.get("title", "导入资料"),
                        "kind": "imported",
                        "chunks": 0,
                        "characters": 0,
                        "imported_at": document.get("imported_at"),
                        "version": document.get("version", "local-import"),
                    },
                )
                summary["chunks"] += 1
                summary["characters"] += len(str(document.get("text", "")))
        return built_in + list(imported.values())

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        with self._lock:
            documents = deepcopy(self._base_documents + self._imported_documents)
        ranked: list[dict[str, Any]] = []
        for document in documents:
            text = f"{document.get('title', '')} {document.get('text', '')}"
            score = _score(query, text, document.get("keywords", []))
            ranked.append({**document, "score": score})

        ranked.sort(key=lambda item: (item["score"], item.get("priority", 0)), reverse=True)
        # Demo corpus is intentionally small; return fallback evidence so the
        # pipeline can demonstrate its structured no-evidence path only when
        # there truly is no corpus at all.
        return ranked[:limit]
