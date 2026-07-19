import os
import unittest
from unittest.mock import patch

from medical_agent.aliyun_model import _extract_json, normalize_base_url
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
