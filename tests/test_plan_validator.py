"""Contract tests for the small model-facing task-plan schema."""

from __future__ import annotations

import unittest


from medical_agent.plan_validator import build_fallback_plan, validate_plan


class ValidatePlanTests(unittest.TestCase):
    def test_accepts_a_small_dependency_dag_and_returns_normalized_tasks(self) -> None:
        plan = {
            "tasks": [
                {"id": 1, "goal": "Extract relevant patient facts", "deps": []},
                {"id": 2, "goal": "Find supporting medical evidence", "deps": [1]},
                {"id": 3, "goal": "Summarize the evidence-backed answer", "deps": [1, 2]},
            ]
        }

        result = validate_plan(plan)

        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["tasks"], plan["tasks"])

    def test_rejects_non_object_plan_input(self) -> None:
        result = validate_plan(
            '{"tasks": [{"id": 1, "goal": "Extract facts", "deps": []}]}'
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["errors"][0]["code"], "PLAN_NOT_OBJECT")

    def test_rejects_non_continuous_ids_and_forward_dependencies(self) -> None:
        result = validate_plan(
            {
                "tasks": [
                    {"id": 1, "goal": "Extract facts", "deps": []},
                    {"id": 3, "goal": "Analyze facts", "deps": [3]},
                ]
            }
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["tasks"], [])
        codes = {issue["code"] for issue in result["errors"]}
        self.assertIn("TASK_ID_NOT_CONTINUOUS", codes)
        self.assertIn("INVALID_DEPENDENCY", codes)

    def test_rejects_duplicate_dependencies(self) -> None:
        result = validate_plan(
            {
                "tasks": [
                    {"id": 1, "goal": "Extract facts", "deps": []},
                    {"id": 2, "goal": "Analyze facts", "deps": [1, 1]},
                ]
            }
        )

        self.assertFalse(result["valid"])
        self.assertIn(
            "DEPENDENCY_DUPLICATE", {issue["code"] for issue in result["errors"]}
        )

    def test_accepts_structured_scope_and_analysis_mode_without_text_routing(self) -> None:
        result = validate_plan(
            {
                "tasks": [
                    {
                        "id": 1,
                        "goal": "任意自然语言目标",
                        "deps": [],
                        "evidence_scope": "knowledge",
                        "analysis_mode": "retrieval",
                    }
                ]
            }
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["tasks"][0]["evidence_scope"], "knowledge")
        self.assertEqual(result["tasks"][0]["analysis_mode"], "retrieval")

    def test_fallback_plan_is_valid_and_conservative(self) -> None:
        for has_patient_record, expected_count in ((False, 2), (True, 3)):
            result = validate_plan(build_fallback_plan(has_patient_record=has_patient_record))

            self.assertTrue(result["valid"])
            self.assertEqual(len(result["tasks"]), expected_count)
            self.assertTrue(all("goal" in task for task in result["tasks"]))


if __name__ == "__main__":
    unittest.main()
