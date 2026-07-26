"""Tests for dependency-aware task scheduling and failure containment."""

from __future__ import annotations

import unittest


from medical_agent.dag_scheduler import execute_dag


class ExecuteDagTests(unittest.TestCase):
    def test_runs_ready_tasks_in_waves_and_passes_upstream_results(self) -> None:
        tasks = [
            {"id": 1, "goal": "patient facts", "deps": []},
            {"id": 2, "goal": "knowledge retrieval", "deps": []},
            {"id": 3, "goal": "patient analysis", "deps": [1]},
            {"id": 4, "goal": "combined summary", "deps": [1, 2]},
        ]

        def worker(task: dict, upstream: dict) -> dict:
            return {"task_id": task["id"], "upstream": dict(upstream)}

        result = execute_dag(tasks, worker, max_workers=2)

        self.assertEqual(result["waves"], [[1, 2], [3, 4]])
        self.assertTrue(
            all(state["status"] == "completed" for state in result["tasks"].values())
        )
        self.assertEqual(
            result["tasks"][3]["result"]["upstream"],
            {1: {"task_id": 1, "upstream": {}}},
        )
        self.assertEqual(
            set(result["tasks"][4]["result"]["upstream"]), {1, 2}
        )

    def test_failed_task_blocks_only_its_descendants(self) -> None:
        tasks = [
            {"id": 1, "goal": "fails", "deps": []},
            {"id": 2, "goal": "independent root", "deps": []},
            {"id": 3, "goal": "blocked direct child", "deps": [1]},
            {"id": 4, "goal": "blocked descendant", "deps": [3]},
            {"id": 5, "goal": "independent completed branch", "deps": [2]},
        ]
        executed: list[int] = []

        def worker(task: dict, upstream: dict) -> dict:
            executed.append(task["id"])
            if task["id"] == 1:
                raise RuntimeError("upstream unavailable")
            return {"task_id": task["id"]}

        result = execute_dag(tasks, worker, max_workers=2)

        states = result["tasks"]
        self.assertEqual(states[1]["status"], "failed")
        self.assertEqual(states[3]["status"], "blocked")
        self.assertEqual(states[4]["status"], "blocked")
        self.assertEqual(states[2]["status"], "completed")
        self.assertEqual(states[5]["status"], "completed")
        self.assertEqual(set(executed), {1, 2, 5})
        self.assertEqual(result["waves"], [[1, 2], [5]])


if __name__ == "__main__":
    unittest.main()
