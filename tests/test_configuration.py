from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from medical_agent.bootstrap import create_agent, create_agent_from_environment
from medical_agent.infrastructure.model_config import (
    ModelConfigurationError,
    load_model_configuration,
    validate_model_configuration,
)


class ConfigurationValidationTests(unittest.TestCase):
    def test_retrieval_source_policy_is_normalized_without_secrets(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.local.json"
            path.write_text(
                json.dumps(
                    {
                        "retrieval": {
                            "source_policy": {
                                "enabled": True,
                                "allowed_source_types": ["Guideline", "guideline"],
                                "blocked_statuses": ["RETRACTED"],
                                "min_priority": 7,
                                "require_version": "true",
                                "allow_synthetic": False,
                                "max_age_days": 90,
                                "reject_unknown_date": True,
                                "as_of_date": "2026-07-27",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {"MEDICAL_AGENT_CONFIG": str(path)}
            )

        policy = configuration.retrieval.source_policy
        self.assertEqual(policy["allowed_source_types"], ["guideline"])
        self.assertEqual(policy["blocked_statuses"], ["retracted"])
        self.assertEqual(policy["min_priority"], 7)
        self.assertTrue(policy["require_version"])
        self.assertFalse(policy["allow_synthetic"])
        self.assertNotIn("api_key", str(policy))

    def test_enabled_integrations_report_actionable_safe_errors(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.local.json"
            path.write_text(
                json.dumps(
                    {
                        "retrieval": {
                            "backend": "faiss",
                            "embedding": {"provider": "openai-compatible"},
                        },
                        "reranker": {"enabled": True},
                        "routing": {"mode": "api"},
                    }
                ),
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {"MEDICAL_AGENT_CONFIG": str(path)}
            )

        diagnostics = validate_model_configuration(configuration)
        self.assertFalse(diagnostics["valid"])
        codes = {item["code"] for item in diagnostics["errors"]}
        self.assertIn("EMBEDDING_API_KEY_REQUIRED", codes)
        self.assertIn("EMBEDDING_BASE_URL_REQUIRED", codes)
        self.assertIn("EMBEDDING_MODEL_REQUIRED", codes)
        self.assertIn("RERANKER_ENDPOINT_REQUIRED", codes)
        self.assertIn("ROUTER_ENDPOINT_REQUIRED", codes)
        self.assertNotIn("secret", str(diagnostics))

    def test_environment_factory_fails_before_building_invalid_external_clients(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.local.json"
            path.write_text('{"reranker":{"enabled":true}}', encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "MEDICAL_AGENT_CONFIG": str(path),
                    "MEDICAL_AGENT_API_KEY": "",
                    "MEDICAL_AGENT_BASE_URL": "",
                    "MEDICAL_AGENT_MODEL": "",
                    "MEDICAL_AGENT_PROVIDER": "",
                },
                clear=False,
            ):
                with self.assertRaises(ModelConfigurationError) as raised:
                    create_agent_from_environment()

        self.assertIn("RERANKER_ENDPOINT_REQUIRED", str(raised.exception.diagnostics))
        self.assertNotIn("api_key", str(raised.exception))

    def test_runtime_catalog_contains_only_safe_backend_identities(self) -> None:
        class _Knowledge:
            def search(self, query: str, limit: int = 4):
                return []

            def import_text(self, *, name: str, content: str):
                return {"name": name, "chunks": 0}

            def list_documents(self):
                return []

            def retrieval_metadata(self):
                return {
                    "backend": "faiss",
                    "index_path": "private/index.faiss",
                    "embedding": {
                        "provider": "openai-compatible",
                        "name": "embed-model",
                        "dimensions": "1536",
                    },
                }

        class _Reranker:
            def rerank(self, *, query, documents, limit=8):
                return documents[:limit]

            def runtime_metadata(self):
                return {
                    "provider": "cohere",
                    "name": "rerank-v3",
                    "endpoint": "https://private.example.invalid",
                }

        agent = create_agent(knowledge_base=_Knowledge(), reranker=_Reranker())
        catalog = agent.model_catalog()
        runtime = catalog["runtime"]
        self.assertEqual(runtime["retrieval"]["backend"], "faiss")
        self.assertEqual(runtime["retrieval"]["embedding"]["provider"], "openai-compatible")
        self.assertEqual(runtime["reranker"]["provider"], "cohere")
        self.assertNotIn("private/index.faiss", str(catalog))
        self.assertNotIn("private.example.invalid", str(catalog))


if __name__ == "__main__":
    unittest.main()
