"""Service-level contracts for safe execution progress and model metadata."""

from __future__ import annotations

import unittest


from medical_agent.adapters.openai_compatible import OpenAICompatibleModelAdapter
from medical_agent.bootstrap import create_service
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.retrieval import JsonKnowledgeBase
from medical_agent.service import MedicalAgentService


class RuntimeMetadataTests(unittest.TestCase):
    def test_demo_and_real_models_expose_only_safe_runtime_identity(self) -> None:
        demo = create_service(
            model_profiles={"demo": DemoModelAdapter()},
            default_model_profile="demo",
            knowledge_base=JsonKnowledgeBase([]),
        )
        real_adapter = OpenAICompatibleModelAdapter(
            api_key="test-key-not-a-real-secret",
            base_url="https://example.invalid",
            model="unit-model",
            provider="aliyun-model-studio",
        )
        real = create_service(
            model_profiles={"real": real_adapter},
            default_model_profile="real",
            knowledge_base=JsonKnowledgeBase([]),
        )

        self.assertEqual(
            demo.model_metadata(),
            {"mode": "demo", "provider": "local-demo", "name": "demo"},
        )
        self.assertEqual(
            real.model_metadata(),
            {
                "mode": "real",
                "provider": "aliyun-model-studio",
                "name": "unit-model",
            },
        )
        self.assertNotIn("test-key-not-a-real-secret", str(real.model_metadata()))
        self.assertNotIn("example.invalid", str(real.model_metadata()))


class ServiceProgressTests(unittest.TestCase):
    def test_demo_chat_emits_sequenced_safe_progress_events(self) -> None:
        knowledge_base = JsonKnowledgeBase([])
        knowledge_base.import_text(
            name="progress-guide.txt",
            content=(
                "PROGRESS-ORBIT-42 requires an evidence-backed general medical "
                "information answer with a traceable source citation."
            ),
        )
        service = create_service(
            model_profiles={"demo": DemoModelAdapter()},
            default_model_profile="demo",
            knowledge_base=knowledge_base,
            max_workers=1,
        )
        events: list[dict] = []

        result = service.chat(
            message="What does PROGRESS-ORBIT-42 require?",
            on_progress=events.append,
        )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(events)
        self.assertEqual(events[-1]["stage"], "completed")
        self.assertEqual(events[-1]["status"], result["status"])
        self.assertEqual(
            [event["sequence"] for event in events], list(range(1, len(events) + 1))
        )
        self.assertEqual({event["run_id"] for event in events}, {result["run"]["id"]})

        allowed_fields = {
            "stage",
            "message",
            "run_id",
            "sequence",
            "timestamp",
            "task_id",
            "task",
            "tasks",
            "status",
            "queries",
            "evidence_ids",
            "facts",
            "claims",
            "evaluation",
            "round",
            "counts",
        }
        self.assertTrue(all(set(event).issubset(allowed_fields) for event in events))
        self.assertTrue(all(isinstance(event["message"], str) for event in events))
        self.assertIn("planning", {event["stage"] for event in events})
        self.assertIn("retrieve", {event["stage"] for event in events})
        self.assertIn("evaluate", {event["stage"] for event in events})
        self.assertIn("completed", {event["stage"] for event in events})


if __name__ == "__main__":
    unittest.main()
