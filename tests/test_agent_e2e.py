"""Offline end-to-end contract test for the local demo agent."""

from __future__ import annotations

import unittest

from medical_agent.bootstrap import create_agent


class MedicalAgentE2ETests(unittest.TestCase):
    def test_default_local_run_creates_cited_report_and_evidence_graph(self) -> None:
        agent = create_agent(max_workers=2)

        result = agent.run(
            request="评估患者当前用药风险，并说明还需要补充哪些信息。",
            patient_record=(
                "患者58岁，正在服用降压药。近一周出现头晕，"
                "肾功能检查结果待复查，无已知药物过敏史。"
            ),
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["run"]["evaluation"]["issues"], [])
        self.assertTrue(result["claims"])
        self.assertTrue(result["evidence"])
        self.assertTrue(result["report"]["markdown"])

        evidence_ids = {item["id"] for item in result["evidence"]}
        graph_edge_ids = {edge["id"] for edge in result["graph"]["edges"]}
        markdown = result["report"]["markdown"]

        for claim in result["claims"]:
            self.assertTrue(claim["refs"], msg=f"{claim['id']} must cite evidence")
            for ref in claim["refs"]:
                self.assertIn(ref, evidence_ids)
                self.assertIn(f"[{ref}]", markdown)
                self.assertIn(f"{ref}-{claim['id']}", graph_edge_ids)
            self.assertIn(f"{claim['id']}-REPORT", graph_edge_ids)

        self.assertTrue(
            all(task["status"] == "completed" for task in result["run"]["tasks"])
        )
        self.assertEqual(result["run"]["waves"], [[1], [2], [3, 4], [5]])
        for task in result["run"]["tasks"]:
            for dependency_id in task["deps"]:
                self.assertIn(
                    f"T{dependency_id}-T{task['id']}-depends_on",
                    graph_edge_ids,
                )


if __name__ == "__main__":
    unittest.main()
