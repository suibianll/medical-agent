from __future__ import annotations

import unittest

from medical_agent.retrieval.governance import SourceGovernancePolicy
from medical_agent.retrieval.knowledge import JsonKnowledgeBase


class SourceGovernancePolicyTests(unittest.TestCase):
    def test_empty_configuration_keeps_governance_disabled(self) -> None:
        self.assertFalse(SourceGovernancePolicy.from_config({}).enabled)

    def test_policy_accepts_trusted_fresh_versioned_metadata_only(self) -> None:
        policy = SourceGovernancePolicy.from_config(
            {
                "enabled": True,
                "allowed_source_types": ["guideline"],
                "blocked_statuses": ["retracted"],
                "min_priority": 5,
                "require_version": True,
                "allow_synthetic": False,
                "max_age_days": 365,
                "reject_unknown_date": True,
                "as_of_date": "2026-07-27",
            }
        )
        accepted, metadata = policy.assess(
            {
                "id": "trusted-1",
                "title": "not copied into governance metadata",
                "text": "clinical source text must never be inspected by policy",
                "source_type": "guideline",
                "priority": 8,
                "version": "2026.2",
                "synthetic": False,
                "updated_at": "2026-06-01",
            }
        )

        self.assertTrue(accepted)
        self.assertEqual(metadata["status"], "accepted")
        self.assertEqual(metadata["age_days"], 56)
        self.assertNotIn("text", metadata)
        self.assertNotIn("title", metadata)

    def test_policy_rejects_each_configured_risk_and_allows_explicit_empty_blocklist(self) -> None:
        policy = SourceGovernancePolicy.from_config(
            {
                "enabled": True,
                "allowed_source_types": ["guideline"],
                "blocked_statuses": ["retracted"],
                "min_priority": 5,
                "require_version": True,
                "allow_synthetic": False,
                "max_age_days": 30,
                "reject_unknown_date": True,
                "as_of_date": "2026-07-27",
            }
        )
        accepted, metadata = policy.assess(
            {
                "source_type": "preprint",
                "status": "retracted",
                "priority": 1,
                "synthetic": True,
                "updated_at": "2025-01-01",
            }
        )
        self.assertFalse(accepted)
        self.assertEqual(
            set(metadata["reasons"]),
            {
                "source_type_not_allowed",
                "blocked_status",
                "priority_below_minimum",
                "synthetic_not_allowed",
                "version_required",
                "source_too_old",
            },
        )

        no_blocklist = SourceGovernancePolicy.from_config(
            {"enabled": True, "blocked_statuses": []}
        )
        accepted, _ = no_blocklist.assess({"status": "retracted"})
        self.assertTrue(accepted)

    def test_knowledge_base_filters_before_ranking_but_preserves_inventory(self) -> None:
        policy = SourceGovernancePolicy.from_config(
            {
                "enabled": True,
                "allowed_source_types": ["guideline"],
                "min_priority": 5,
            }
        )
        knowledge_base = JsonKnowledgeBase(
            [
                {
                    "id": "accepted",
                    "title": "guideline",
                    "text": "shared clinical marker",
                    "source_type": "guideline",
                    "priority": 9,
                },
                {
                    "id": "rejected",
                    "title": "untrusted",
                    "text": "shared clinical marker",
                    "source_type": "blog",
                    "priority": 99,
                },
            ],
            governance_policy=policy,
        )

        result = knowledge_base.search("shared clinical marker", limit=8)
        self.assertEqual([item["id"] for item in result], ["accepted"])
        self.assertEqual(result[0]["governance"]["status"], "accepted")
        self.assertEqual({item["id"] for item in knowledge_base.documents}, {"accepted", "rejected"})
        runtime = knowledge_base.retrieval_metadata()
        self.assertTrue(runtime["governance"]["enabled"])
        self.assertNotIn("shared clinical marker", str(runtime))


if __name__ == "__main__":
    unittest.main()
