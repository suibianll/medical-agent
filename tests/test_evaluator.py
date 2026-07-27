"""Tests for deterministic evidence-citation gates."""

from __future__ import annotations

import unittest


from medical_agent.evaluator import evaluate_claims


EVIDENCE = {
    "P1": {"id": "P1", "kind": "patient", "text": "Patient fact"},
    "K1": {"id": "K1", "kind": "knowledge", "text": "Guideline fact"},
}


class EvaluateClaimsTests(unittest.TestCase):
    def test_rejects_missing_and_unknown_citations(self) -> None:
        result = evaluate_claims(
            [
                {"id": "C1", "text": "A claim without a citation", "refs": []},
                {"id": "C2", "text": "A claim with a bad citation", "refs": ["K404"]},
            ],
            EVIDENCE,
        )

        self.assertFalse(result["pass"])
        issues = {(issue["claim"], issue["code"]) for issue in result["issues"]}
        self.assertIn(("C1", "NO_REF"), issues)
        self.assertIn(("C2", "BAD_REF"), issues)

    def test_clinical_claim_requires_patient_and_knowledge_evidence(self) -> None:
        result = evaluate_claims(
            [
                {
                    "id": "C1",
                    "text": "Dose assessment",
                    "refs": ["P1"],
                    "requires_dual_support": True,
                },
                {
                    "id": "C2",
                    "text": "Treatment assessment",
                    "refs": ["K1"],
                    "requires_dual_support": True,
                },
                {
                    "id": "C3",
                    "text": "Evidence-backed assessment",
                    "refs": ["P1", "K1"],
                    "requires_dual_support": True,
                },
            ],
            EVIDENCE,
        )

        self.assertFalse(result["pass"])
        issues = {(issue["claim"], issue["code"]) for issue in result["issues"]}
        self.assertIn(("C1", "MISSING_KB_REF"), issues)
        self.assertIn(("C2", "MISSING_PATIENT_REF"), issues)
        self.assertNotIn(("C3", "MISSING_KB_REF"), issues)
        self.assertNotIn(("C3", "MISSING_PATIENT_REF"), issues)

    def test_optional_semantic_judge_can_reject_otherwise_valid_claim(self) -> None:
        class RejectingJudge:
            def judge_claims(self, items: list[dict]) -> dict[str, str]:
                return {item["id"]: "NOT_SUPPORTED" for item in items}

        result = evaluate_claims(
            [{"id": "C1", "text": "A grounded statement", "refs": ["P1", "K1"]}],
            EVIDENCE,
            RejectingJudge(),
        )

        self.assertFalse(result["pass"])
        self.assertIn(
            {"claim": "C1", "code": "NOT_SUPPORTED"}, result["issues"]
        )
        self.assertEqual(result["judgements"], [{"claim": "C1", "verdict": "NOT_SUPPORTED"}])

    def test_claim_verifier_distinguishes_partial_conflict_and_insufficient(self) -> None:
        class TaxonomyJudge:
            def judge_claims(self, items: list[dict]) -> dict[str, str]:
                return {
                    item["id"]: verdict
                    for item, verdict in zip(
                        items,
                        ["PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
                        strict=False,
                    )
                }

        result = evaluate_claims(
            [
                {"id": "C1", "text": "partial", "refs": ["K1"]},
                {"id": "C2", "text": "conflict", "refs": ["K1"]},
                {"id": "C3", "text": "unknown", "refs": ["K1"]},
            ],
            EVIDENCE,
            TaxonomyJudge(),
        )

        self.assertFalse(result["pass"])
        self.assertEqual(
            result["verdict_counts"],
            {
                "PARTIALLY_SUPPORTED": 1,
                "CONTRADICTED": 1,
                "INSUFFICIENT": 1,
            },
        )
        issue_codes = {(issue["claim"], issue["code"]) for issue in result["issues"]}
        self.assertIn(("C1", "PARTIAL_SUPPORT"), issue_codes)
        self.assertIn(("C2", "CONTRADICTED"), issue_codes)
        self.assertIn(("C3", "INSUFFICIENT_EVIDENCE"), issue_codes)
        relations = {edge["claim_id"]: edge["relation"] for edge in result["support_edges"]}
        self.assertEqual(relations, {"C1": "qualifies", "C2": "contradicts", "C3": "uncertain"})

    def test_dual_support_is_not_inferred_from_claim_words(self) -> None:
        result = evaluate_claims(
            [{"id": "C1", "text": "诊断建议", "refs": ["P1"]}],
            EVIDENCE,
        )

        self.assertTrue(result["pass"])


if __name__ == "__main__":
    unittest.main()
