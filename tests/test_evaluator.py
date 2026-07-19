"""Tests for deterministic evidence-citation gates."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
            def judge_claim(self, *, claim: dict, evidence: list[dict]) -> str:
                return "NOT_SUPPORTED"

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


if __name__ == "__main__":
    unittest.main()
