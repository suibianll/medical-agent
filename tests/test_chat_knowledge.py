"""Contracts for conversational answers and local knowledge imports."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


from medical_agent.demo_model import DemoModelAdapter
from medical_agent.bootstrap import create_agent
from medical_agent.retrieval import JsonKnowledgeBase


class JsonKnowledgeBaseImportTests(unittest.TestCase):
    def test_import_persists_chunk_metadata_and_retrieves_imported_content(self) -> None:
        with TemporaryDirectory() as directory:
            storage_path = Path(directory) / "imported_knowledge.json"
            knowledge_base = JsonKnowledgeBase([], storage_path=storage_path)
            marker = "KEPLER-ORBIT-42"
            content = (
                "This imported guide provides routine clinical background.\n" * 30
                + f"The unique retrieval marker is {marker}.\n"
                + "Additional imported context follows for retrieval testing.\n" * 30
            )

            imported = knowledge_base.import_text(
                name="unit-guide.txt",
                content=content,
            )

            self.assertGreater(imported["chunks_added"], 1)
            self.assertEqual(len(imported["documents"]), 1)
            self.assertEqual(imported["documents"][0]["name"], "unit-guide.txt")
            self.assertEqual(imported["documents"][0]["chunks"], imported["chunks_added"])

            listed = knowledge_base.list_documents()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["kind"], "imported")
            self.assertEqual(listed[0]["chunks"], imported["chunks_added"])

            self.assertTrue(storage_path.is_file())
            persisted = json.loads(storage_path.read_text(encoding="utf-8"))
            self.assertEqual(len(persisted["documents"]), imported["chunks_added"])
            self.assertTrue(
                all(item["source_type"] == "imported" for item in persisted["documents"])
            )
            self.assertEqual(len(knowledge_base.documents), imported["chunks_added"])

            retrieved = knowledge_base.search(marker)
            self.assertTrue(any(marker in item["text"] for item in retrieved))
            self.assertTrue(all(item["title"] == "unit-guide.txt" for item in retrieved))


class MedicalAgentChatAndKnowledgeTests(unittest.TestCase):
    def test_failed_persistence_does_not_commit_imported_chunks_in_memory(self) -> None:
        knowledge_base = JsonKnowledgeBase([])

        with patch.object(
            knowledge_base,
            "_persist_imports",
            side_effect=OSError("simulated write failure"),
        ):
            with self.assertRaises(OSError):
                knowledge_base.import_text(name="guide.txt", content="transactional evidence")

        self.assertEqual(knowledge_base.documents, [])

    def test_agent_import_and_list_use_the_configured_knowledge_base(self) -> None:
        with TemporaryDirectory() as directory:
            knowledge_base = JsonKnowledgeBase(
                [], storage_path=Path(directory) / "imports.json"
            )
            service = create_agent(
                model_profiles={"test": DemoModelAdapter()},
                default_model_profile="test",
                knowledge_base=knowledge_base,
                max_workers=1,
            )

            imported = service.import_knowledge(
                name="service-guide.md",
                content="Service-specific evidence is available for direct retrieval.",
            )
            documents = service.list_knowledge()

            self.assertEqual(imported["chunks_added"], 1)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["name"], "service-guide.md")
            self.assertEqual(documents[0]["kind"], "imported")

    def test_general_chat_passes_with_knowledge_citations_only(self) -> None:
        with TemporaryDirectory() as directory:
            knowledge_base = JsonKnowledgeBase(
                [], storage_path=Path(directory) / "imports.json"
            )
            service = create_agent(
                model_profiles={"test": DemoModelAdapter()},
                default_model_profile="test",
                knowledge_base=knowledge_base,
                max_workers=1,
            )
            service.import_knowledge(
                name="general-guide.txt",
                content=(
                    "A general medical knowledge guide says that ORBIT-42 questions "
                    "require evidence-backed, non-personalized information."
                ),
            )

            result = service.chat(
                message="What does the ORBIT-42 guide say?",
                patient_record="",
            )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["mode"], "general")
            self.assertTrue(result["claims"])
            self.assertEqual(result["run"]["evaluation"]["issues"], [])
            self.assertIn("[K", result["answer"])
            for claim in result["claims"]:
                self.assertFalse(claim["requires_dual_support"])
                self.assertTrue(claim["refs"])
                self.assertTrue(all(ref.startswith("K") for ref in claim["refs"]))
                self.assertFalse(any(ref.startswith("P") for ref in claim["refs"]))
            self.assertTrue(all(item["kind"] == "knowledge" for item in result["evidence"]))


if __name__ == "__main__":
    unittest.main()
