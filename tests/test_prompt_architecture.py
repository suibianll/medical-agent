"""Architecture contracts for prompt, adapter and infrastructure separation."""

from __future__ import annotations

from pathlib import Path
import unittest

import medical_agent
from medical_agent.infrastructure.model_config import load_model_configuration
from medical_agent.prompting import (
    JSON_SYSTEM_PROMPT,
    build_claim_batch_judge_prompt,
    build_contextual_request,
    build_plan_prompt,
    build_synthesis_prompt,
    render_json_prompt,
    task_prompt_view,
)


class PromptArchitectureTests(unittest.TestCase):
    def test_plan_prompt_keeps_dependency_schema_and_small_payload(self) -> None:
        prompt = build_plan_prompt("review", "patient")

        self.assertIn('"deps":[]', prompt.task)
        self.assertIn("evidence_scope", prompt.task)
        self.assertIn("analysis_mode", prompt.task)
        self.assertEqual(prompt.payload, {"request": "review", "patient_record": "patient"})
        self.assertLessEqual(prompt.max_tokens, 900)

    def test_json_wrapper_is_owned_by_prompt_package(self) -> None:
        rendered = render_json_prompt(build_plan_prompt("review", ""))

        self.assertEqual(rendered.system, JSON_SYSTEM_PROMPT)
        self.assertIn("INSTRUCTION", rendered.user)
        self.assertIn("DATA_JSON", rendered.user)
        self.assertIn('"request": "review"', rendered.user)
        self.assertIn("不可信数据", rendered.system)

    def test_conversation_context_cannot_be_presented_as_evidence(self) -> None:
        request = build_contextual_request(
            "当前请求", [{"role": "user", "content": "先前内容"}]
        )

        self.assertIn("不是患者事实或医学证据", request)
        self.assertIn("当前请求", request)

    def test_provider_adapter_contains_no_embedded_medical_prompt_text(self) -> None:
        package_dir = Path(medical_agent.__file__).resolve().parent
        adapter_path = package_dir / "adapters" / "openai_compatible.py"
        source = adapter_path.read_text(encoding="utf-8")

        for marker in ("你是医疗", "返回模板", "判断给定证据", "不得编造"):
            self.assertNotIn(marker, source)

    def test_explicit_empty_environment_does_not_read_process_variables(self) -> None:
        configuration = load_model_configuration({"MEDICAL_AGENT_CONFIG": "missing.json"})

        self.assertEqual(configuration.profiles, ())
        self.assertEqual(configuration.default_profile, "")

    def test_synthesis_receives_validated_facts_without_duplicate_evidence(self) -> None:
        prompt = build_synthesis_prompt(
            task={"id": 1, "goal": "分析", "deps": []},
            request="review",
            facts=[{"text": "fact", "ref": "K1"}],
        )

        self.assertEqual(prompt.payload["facts"], [{"text": "fact", "ref": "K1"}])
        self.assertNotIn("evidence", prompt.payload)
        self.assertLessEqual(prompt.max_tokens, 800)

    def test_batch_evaluation_keeps_a_flat_bounded_protocol(self) -> None:
        prompt = build_claim_batch_judge_prompt(
            [
                {
                    "id": "C1",
                    "claim": {"text": "claim"},
                    "evidence": [{"id": "K1", "text": "support"}],
                }
            ]
        )

        self.assertEqual(prompt.payload["items"][0]["id"], "C1")
        self.assertIn('"verdicts"', prompt.task)
        self.assertLessEqual(prompt.max_tokens, 900)

    def test_task_prompt_view_drops_runtime_only_fields(self) -> None:
        view = task_prompt_view(
            {"id": 1, "goal": "goal", "deps": [], "runtime_secret": "hidden"}
        )

        self.assertEqual(view, {"id": 1, "goal": "goal", "deps": []})


if __name__ == "__main__":
    unittest.main()
