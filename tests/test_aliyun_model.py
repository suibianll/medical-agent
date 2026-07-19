import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from medical_agent.aliyun_model import (
    AliyunCompatibleModelAdapter,
    _extract_json,
    normalize_base_url,
)
from medical_agent.service import MedicalAgentService


class AliyunAdapterConfigurationTests(unittest.TestCase):
    def test_normalizes_workspace_url_without_persisting_credentials(self):
        self.assertEqual(
            normalize_base_url("https://workspace.cn-beijing.maas.aliyuncs.com"),
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(
            normalize_base_url("https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/"),
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )

    def test_parses_json_even_when_model_wraps_it_in_a_code_fence(self):
        payload = _extract_json('```json\n{"queries":["abc"]}\n```')
        self.assertEqual(payload, {"queries": ["abc"]})

    def test_environment_factory_selects_real_adapter_only_when_all_values_exist(self):
        values = {
            "MEDICAL_AGENT_API_KEY": "test-key-not-a-real-secret",
            "MEDICAL_AGENT_BASE_URL": "https://workspace.cn-beijing.maas.aliyuncs.com",
            "MEDICAL_AGENT_MODEL": "qwen3.7-plus",
        }
        with patch.dict(os.environ, values, clear=False):
            service = MedicalAgentService.from_environment()
        self.assertEqual(service.model.__class__.__name__, "AliyunCompatibleModelAdapter")
        self.assertEqual(service.model.base_url.split("/compatible-mode")[0], values["MEDICAL_AGENT_BASE_URL"])

    def test_environment_factory_loads_a_gitignored_local_config(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "model.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "api_key": "test-config-key",
                        "base_url": "https://config.example.invalid",
                        "model": "config-model",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "MEDICAL_AGENT_CONFIG": str(config_path),
                    "MEDICAL_AGENT_API_KEY": "",
                    "DASHSCOPE_API_KEY": "",
                    "MEDICAL_AGENT_BASE_URL": "",
                    "MEDICAL_AGENT_MODEL": "",
                },
                clear=False,
            ):
                service = MedicalAgentService.from_environment()

        self.assertEqual(service.model.__class__.__name__, "AliyunCompatibleModelAdapter")
        self.assertEqual(service.model.model, "config-model")
        self.assertEqual(
            service.model.base_url,
            "https://config.example.invalid/compatible-mode/v1",
        )
        self.assertNotIn("test-config-key", str(service.model_metadata()))

    def test_real_adapter_exposes_safe_runtime_metadata_without_network_call(self):
        adapter = AliyunCompatibleModelAdapter(
            api_key="test-key-not-a-real-secret",
            base_url="https://example.invalid",
            model="unit-model",
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
        # Runtime identity must be safe to expose in a health response.
        self.assertNotIn("test-key-not-a-real-secret", str(metadata))
        self.assertNotIn("example.invalid", str(metadata))
