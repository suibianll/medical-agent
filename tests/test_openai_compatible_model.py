import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from medical_agent.adapters.openai_compatible import OpenAICompatibleModelAdapter
from medical_agent.bootstrap import create_agent_from_environment
from medical_agent.infrastructure.openai_client import normalize_base_url
from medical_agent.utils.json_tools import extract_json_object


class OpenAICompatibleConfigurationTests(unittest.TestCase):
    def test_normalizes_aliyun_workspace_url(self) -> None:
        self.assertEqual(
            normalize_base_url(
                "https://workspace.cn-beijing.maas.aliyuncs.com",
                "aliyun-model-studio",
            ),
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(
            normalize_base_url(
                "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/",
                "aliyun-model-studio",
            ),
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )

    def test_parses_fenced_json_response(self) -> None:
        payload = extract_json_object('```json\n{"queries":["abc"]}\n```')
        self.assertEqual(payload, {"queries": ["abc"]})

    def test_environment_factory_selects_real_adapter_when_complete(self) -> None:
        values = {
            "MEDICAL_AGENT_API_KEY": "test-key-not-a-real-secret",
            "MEDICAL_AGENT_BASE_URL": "https://workspace.cn-beijing.maas.aliyuncs.com",
            "MEDICAL_AGENT_MODEL": "qwen3.7-plus",
            "MEDICAL_AGENT_PROVIDER": "aliyun-model-studio",
        }
        with patch.dict(os.environ, values, clear=False):
            service = create_agent_from_environment()

        adapter = service.model_profiles[service.default_model_profile]
        self.assertIsInstance(adapter, OpenAICompatibleModelAdapter)
        self.assertEqual(
            adapter.base_url.split("/compatible-mode")[0],
            values["MEDICAL_AGENT_BASE_URL"],
        )

    def test_environment_factory_loads_profile_config(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "model.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "configured",
                        "profiles": [
                            {
                                "id": "configured",
                                "provider": "aliyun-model-studio",
                                "api_key": "test-config-key",
                                "base_url": "https://config.example.invalid",
                                "model": "config-model",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "MEDICAL_AGENT_CONFIG": str(config_path),
                    "MEDICAL_AGENT_API_KEY": "",
                    "MEDICAL_AGENT_BASE_URL": "",
                    "MEDICAL_AGENT_MODEL": "",
                },
                clear=False,
            ):
                service = create_agent_from_environment()

        adapter = service.model_profiles[service.default_model_profile]
        self.assertIsInstance(adapter, OpenAICompatibleModelAdapter)
        self.assertEqual(adapter.model, "config-model")
        self.assertEqual(
            adapter.base_url,
            "https://config.example.invalid/compatible-mode/v1",
        )
        self.assertNotIn("test-config-key", str(service.model_metadata()))

    def test_adapter_exposes_safe_runtime_metadata_without_network_call(self) -> None:
        adapter = OpenAICompatibleModelAdapter(
            api_key="test-key-not-a-real-secret",
            base_url="https://example.invalid",
            model="unit-model",
            provider="aliyun-model-studio",
        )

        metadata = adapter.runtime_metadata()

        self.assertEqual(
            metadata,
            {
                "mode": "real",
                "provider": "aliyun-model-studio",
                "name": "unit-model",
            },
        )
        self.assertNotIn("test-key-not-a-real-secret", str(metadata))
        self.assertNotIn("example.invalid", str(metadata))


if __name__ == "__main__":
    unittest.main()
