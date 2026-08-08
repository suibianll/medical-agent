"""Thread-safe local JSON knowledge base and import persistence."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any
from uuid import uuid4

from .fusion import fuse_ranked_results, normalize_queries
from .governance import SourceGovernancePolicy
from .scoring import normalized_score, score


MAX_IMPORTED_DOCUMENT_CHARS = 1_000_000
MAX_IMPORTED_DOCUMENTS_PER_REQUEST = 32
CHUNK_CHARS = 900


class KnowledgeImportError(ValueError):
    """Raised when a local knowledge import cannot be safely indexed."""


class JsonKnowledgeBase:
    """Thread-safe local knowledge base with persistent imported documents.

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
        governance_policy: SourceGovernancePolicy | None = None,
    ) -> None:
        self._base_documents = deepcopy(documents)
        self._imported_documents = deepcopy(imported_documents or [])
        self._storage_path = storage_path
        self._governance_policy = governance_policy or SourceGovernancePolicy()
        self._lock = RLock()

    @property
    def documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._base_documents + self._imported_documents)

    @classmethod
    def demo(
        cls,
        *,
        storage_path: Path | None = None,
        governance_policy: SourceGovernancePolicy | None = None,
    ) -> "JsonKnowledgeBase":
        data_path = Path(__file__).resolve().parents[1] / "resources" / "knowledge.json"
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        imported_documents: list[dict[str, Any]] = []
        if storage_path is not None and storage_path.is_file():
            try:
                with storage_path.open("r", encoding="utf-8") as handle:
                    imported_payload = json.load(handle)
                candidate = imported_payload.get("documents", [])
                if isinstance(candidate, list):
                    imported_documents = [
                        item
                        for item in candidate
                        if isinstance(item, dict) and item.get("text")
                    ]
            except (OSError, json.JSONDecodeError):
                # A malformed local import must not prevent the application from
                # starting. The user can re-import the source document.
                imported_documents = []
        return cls(
            payload["documents"],
            imported_documents=imported_documents,
            storage_path=storage_path,
            governance_policy=governance_policy,
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

        parse_as_json = source_name.lower().endswith(".json") or stripped.startswith(
            ("{", "[")
        )
        if parse_as_json:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                if source_name.lower().endswith(".json"):
                    raise KnowledgeImportError("JSON 知识文件格式无效。") from None
            else:
                documents = payload.get("documents") if isinstance(payload, dict) else payload
                if isinstance(documents, list):
                    if len(documents) > MAX_IMPORTED_DOCUMENTS_PER_REQUEST:
                        raise KnowledgeImportError(
                            f"单次最多导入 {MAX_IMPORTED_DOCUMENTS_PER_REQUEST} 份资料。"
                        )
                    entries: list[tuple[str, str]] = []
                    for index, item in enumerate(
                        documents, start=1
                    ):
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text", "")
                        title = item.get("title", f"{source_name} {index}")
                        if isinstance(text, str) and text.strip():
                            entries.append(
                                (
                                    JsonKnowledgeBase._safe_name(str(title)),
                                    text.strip(),
                                )
                            )
                    if entries:
                        return entries
                if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                    return [
                        (
                            JsonKnowledgeBase._safe_name(
                                str(payload.get("title", source_name))
                            ),
                            payload["text"].strip(),
                        )
                    ]

        return [(source_name, stripped)]

    def _persist_imports(self, documents: list[dict[str, Any]]) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._storage_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"documents": documents},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._storage_path)

    def import_text(self, *, name: str, content: str) -> dict[str, Any]:
        """Import text/Markdown/JSON as retrievable provenance-preserving chunks."""

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
            candidate_documents = [*self._imported_documents, *imported]
            # Publish the in-memory state only after the atomic file replacement
            # succeeds, so a failed request cannot leave process-only chunks.
            self._persist_imports(candidate_documents)
            self._imported_documents = candidate_documents
        return {"documents": summaries, "chunks_added": len(imported)}

    def list_documents(self) -> list[dict[str, Any]]:
        """Return compact metadata without imported document text."""

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

    def eligible_documents(self) -> list[dict[str, Any]]:
        """Return documents that pass metadata-only source governance.

        The public ``documents`` property intentionally remains an inventory
        view for import/document management. Retrieval alone uses this filtered
        view so rejected sources never reach lexical ranking, embeddings or a
        reranker.
        """

        with self._lock:
            documents = deepcopy(self._base_documents + self._imported_documents)
        return self._governance_policy.filter_documents(documents)

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        documents = self.eligible_documents()
        ranked: list[dict[str, Any]] = []
        for document in documents:
            text = f"{document.get('title', '')} {document.get('text', '')}"
            ranked.append(
                {
                    **document,
                    "score": score(query, text, document.get("keywords", [])),
                    "relevance_score": normalized_score(
                        query, text, document.get("keywords", [])
                    ),
                }
            )

        ranked.sort(
            key=lambda item: (item["score"], item.get("priority", 0)),
            reverse=True,
        )
        # The demo corpus intentionally returns lexical fallbacks when present.
        return ranked[:limit]

    def search_many(
        self,
        queries: list[str],
        *,
        limit: int = 8,
        source_types: set[str] | None = None,
        max_per_document: int = 2,
    ) -> list[dict[str, Any]]:
        """Fuse several sparse query views with reciprocal-rank fusion.

        ``HybridKnowledgeBase`` composes this bounded sparse result stream with
        a dense backend without changing the application workflow contract.
        """

        normalized_queries = normalize_queries(queries)
        if not normalized_queries:
            return []

        per_query_limit = max(limit * 2, 8)
        query_results: list[tuple[str, list[dict[str, Any]]]] = []
        for query in normalized_queries[:3]:
            documents = [
                document
                for document in self.search(query, per_query_limit)
                if not source_types
                or str(document.get("source_type", "built_in")) in source_types
            ]
            query_results.append((query, documents))
        return fuse_ranked_results(
            query_results,
            limit=limit,
            max_per_document=max_per_document,
            method="rrf_lexical",
        )

    def retrieval_metadata(self) -> dict[str, Any]:
        return {
            "backend": "lexical",
            "governance": self._governance_policy.runtime_metadata(),
        }
