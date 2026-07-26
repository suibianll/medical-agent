"""Architecture contracts for prompt, adapter and infrastructure separation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_agent.prompts.common import JSON_SYSTEM_PROMPT, render_json_prompt
from medical_agent.prompts.conversation import build_contextual_request
from medical_agent.prompts.planning import build_plan_prompt
from medical_agent.infrastructure.model_config import load_model_configuration


class PromptArchitectureTests(unittest.TestCase):
    def test_plan_prompt_keeps_dependency_schema_and_small_payload(self) -> None:
        prompt = build_plan_prompt("review", "patient")

        self.assertIn('"deps":[]', prompt.task)
        self.assertEqual(prompt.payload, {"request": "review", "patient_record": "patient"})
        self.assertLessEqual(prompt.max_tokens, 900)

    def test_json_wrapper_is_owned_by_prompt_package(self) -> None:
        rendered = render_json_prompt(build_plan_prompt("review", ""))

        self.assertEqual(rendered.system, JSON_SYSTEM_PROMPT)
        self.assertIn("任务：", rendered.user)
        self.assertIn('"request": "review"', rendered.user)

    def test_conversation_context_cannot_be_presented_as_evidence(self) -> None:
        request = build_contextual_request(
            "当前请求", [{"role": "user", "content": "先前内容"}]
        )

        self.assertIn("不能作为患者事实或外部医学证据", request)
        self.assertIn("当前请求", request)

    def test_provider_adapter_contains_no_embedded_medical_prompt_text(self) -> None:
        adapter_path = SRC / "medical_agent" / "adapters" / "openai_compatible.py"
        source = adapter_path.read_text(encoding="utf-8")

        for marker in ("你是医疗", "返回模板", "判断给定证据", "不得编造"):
            self.assertNotIn(marker, source)

    def test_explicit_empty_environment_does_not_read_process_variables(self) -> None:
        configuration = load_model_configuration({"MEDICAL_AGENT_CONFIG": "missing.json"})

        self.assertEqual(configuration.profiles, ())
        self.assertEqual(configuration.default_profile, "")


if __name__ == "__main__":
    unittest.main()
