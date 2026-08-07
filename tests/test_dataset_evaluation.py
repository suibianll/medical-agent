"""Tests for the normalized benchmark adapters and redacted runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from medical_agent.dataset_evaluation import (
    EvaluationCase,
    _aggregate_agent_results,
    _answer_score,
    _benchmark_request,
    _verify_benchmark_answer_provenance,
    load_dataset_bundle,
    run_dataset_evaluation,
)


class DatasetEvaluationTests(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_pubmedqa_adapter_and_retrieval_report_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_json(
                root / "pubmedqa" / "ori_pqal.json",
                {
                    "q1": {
                        "QUESTION": "private benchmark question marker",
                        "CONTEXTS": ["The intervention improved outcomes."],
                        "final_decision": "yes",
                    }
                },
            )
            self._write_json(root / "pubmedqa" / "test_ground_truth.json", {"q1": "yes"})

            bundle = load_dataset_bundle("pubmedqa", data_root=root, max_cases=1)
            self.assertEqual(len(bundle.cases), 1)
            self.assertEqual(bundle.cases[0].gold_answer, "yes")
            self.assertEqual(len(bundle.documents), 1)

            report = run_dataset_evaluation(
                datasets=["pubmedqa"],
                data_root=root,
                mode="retrieval",
                max_cases=1,
            )
            result = report["results"][0]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["retrieval"]["evaluated_cases"], 1)
            self.assertNotIn("private benchmark question marker", json.dumps(report))

    def test_benchmark_request_adds_gold_free_label_contract(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Does the intervention help?",
            gold_answer="yes",
            answer_type="label",
        )

        request = _benchmark_request(case)

        self.assertIn("Final answer: <label>", request)
        self.assertIn("yes, no, maybe", request)

    def test_label_parser_does_not_use_gold_as_fallback(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Does the intervention help?",
            gold_answer="yes",
            answer_type="label",
        )

        score = _answer_score(case, "The evidence contains the word yes incidentally.")

        self.assertFalse(score["evaluated"])

    def test_benchmark_request_includes_choice_options(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Which option?",
            gold_answer="B",
            answer_type="choice",
            options={"A": "alpha", "B": "beta"},
        )

        request = _benchmark_request(case)

        self.assertIn("A. alpha", request)
        self.assertIn("B. beta", request)
        self.assertNotIn("correct", request.lower())

    def test_text_answer_contract_is_machine_parseable(self) -> None:
        case = EvaluationCase(
            case_id="case-1",
            query="Which entity?",
            gold_answer="Alpha entity",
            answer_type="text",
        )

        score = _answer_score(case, "Final answer: Alpha entity")

        self.assertTrue(score["scorable"])
        self.assertTrue(score["evaluated"])
        self.assertTrue(score["correct"])

    def test_final_answer_provenance_requires_supported_workflow_claim(self) -> None:
        result = {
            "evidence": [{"id": "K1", "text": "The intervention improved outcomes."}],
            "claims": [{"id": "C1", "status": "supported", "refs": ["K1"]}],
        }

        verification = _verify_benchmark_answer_provenance(
            {"claims": [{"text": "Final answer: yes", "refs": ["K1"]}]},
            result,
        )

        self.assertTrue(verification["passed"])

    def test_final_answer_provenance_rejects_unverified_source(self) -> None:
        result = {
            "evidence": [{"id": "K1", "text": "The intervention is unclear."}],
            "claims": [{"id": "C1", "status": "needs_repair", "refs": ["K1"]}],
        }

        verification = _verify_benchmark_answer_provenance(
            {"claims": [{"text": "Final answer: yes", "refs": ["K1"]}]},
            result,
        )

        self.assertFalse(verification["passed"])
        self.assertIn("FINAL_ANSWER_UNSUPPORTED_SOURCE", verification["issues"])

    def test_unscored_case_is_not_unparseable(self) -> None:
        case = EvaluationCase(case_id="case-1", query="Find evidence")

        score = _answer_score(case, "supporting claim")

        self.assertFalse(score["scorable"])
        self.assertFalse(score["evaluated"])

    def test_agent_cost_flags_missing_provider_telemetry(self) -> None:
        summary = _aggregate_agent_results(
            [
                {
                    "status": "error",
                    "latency_ms": 12.0,
                    "answer": {"scorable": False, "evaluated": False},
                    "model_usage": {"calls": 0},
                }
            ],
            observed_calls=1,
            stage_counts={"plan": 1},
        )

        self.assertFalse(summary["cost"]["telemetry_complete"])
        self.assertEqual(summary["cost"]["unreported_calls"], 1)

    def test_agent_cost_accepts_multiple_provider_calls_per_stage(self) -> None:
        summary = _aggregate_agent_results(
            [
                {
                    "status": "passed",
                    "latency_ms": 12.0,
                    "answer": {"scorable": False, "evaluated": False},
                    "model_usage": {"calls": 2},
                }
            ],
            observed_calls=1,
            stage_counts={"judge_claims": 1},
        )

        self.assertTrue(summary["cost"]["telemetry_complete"])
        self.assertEqual(summary["cost"]["unreported_calls"], 0)

    def test_agent_answer_reports_coverage_and_overall_accuracy(self) -> None:
        summary = _aggregate_agent_results(
            [
                {
                    "status": "passed",
                    "latency_ms": 10.0,
                    "answer": {
                        "scorable": True,
                        "evaluated": True,
                        "correct": True,
                    },
                },
                {
                    "status": "needs_human_review",
                    "latency_ms": 20.0,
                    "answer": {
                        "scorable": True,
                        "evaluated": False,
                        "correct": False,
                    },
                },
            ],
            observed_calls=0,
            stage_counts={},
        )

        self.assertEqual(summary["answer"]["coverage"], 0.5)
        self.assertEqual(summary["answer"]["accuracy"], 1.0)
        self.assertEqual(summary["answer"]["overall_accuracy"], 0.5)

    def test_evidencebench_adapter_maps_gold_sentence_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_json(
                root / "evidencebench" / "evidencebench_test_set.json",
                {
                    "case-1": {
                        "hypothesis": "The treatment is effective.",
                        "paper_as_candidate_pool": ["irrelevant", "supporting sentence"],
                        "aspect2sentence_indices": {"aspect-1": [1]},
                        "results_aspect_list_ids": ["aspect-1"],
                    }
                },
            )
            bundle = load_dataset_bundle(
                "evidencebench", data_root=root, split="test", max_cases=1
            )
            self.assertEqual(bundle.cases[0].relevant_ids, ("case-1-sent-1",))
            self.assertEqual(len(bundle.documents), 2)

    def test_medqa_adapter_normalizes_choice_answers_without_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "medqa-usmle-hf-qa-mirror" / "phrases_no_exclude_test.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "medqa-1",
                        "question": "Which option is correct?",
                        "options": {"A": "first", "B": "second"},
                        "answer_idx": "B",
                        "meta_info": "test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bundle = load_dataset_bundle("medqa-usmle", data_root=root, max_cases=1)
            self.assertEqual(bundle.cases[0].gold_answer, "B")
            self.assertEqual(bundle.cases[0].options["B"], "second")
            self.assertFalse(bundle.documents)

    def test_faithfulness_adapter_preserves_isolated_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "faithfulness-qa-2026" / "data" / "faithfulness_qa_squad_test.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "faith-1",
                        "question": "What entity is mentioned?",
                        "original_context": "The entity is alpha.",
                        "modified_context": "The entity is beta.",
                        "original_answer": "alpha",
                        "faithful_answer": "beta",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bundle = load_dataset_bundle(
                "faithfulness-qa-2026", data_root=root, max_cases=1
            )
            self.assertTrue(bundle.cases[0].expected_variant_change)
            self.assertEqual(
                [name for name, _ in bundle.cases[0].variants], ["original", "modified"]
            )
            self.assertEqual(len(bundle.documents), 1)

    def test_reference_only_dataset_is_reported_as_skipped(self) -> None:
        report = run_dataset_evaluation(
            datasets=["bioasq"],
            mode="retrieval",
            max_cases=1,
        )
        self.assertEqual(report["results"][0]["status"], "skipped")
        self.assertEqual(report["summary"]["skipped"], 1)

    def test_case_window_keeps_full_retrieval_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_json(
                root / "pubmedqa" / "ori_pqal.json",
                {
                    f"q{index}": {
                        "QUESTION": f"Question {index}",
                        "CONTEXTS": [f"Context {index}"],
                        "final_decision": "yes",
                    }
                    for index in range(4)
                },
            )
            self._write_json(
                root / "pubmedqa" / "test_ground_truth.json",
                {f"q{index}": "yes" for index in range(4)},
            )

            report = run_dataset_evaluation(
                datasets=["pubmedqa"],
                data_root=root,
                mode="retrieval",
                max_cases=4,
                case_offset=1,
                case_limit=2,
            )

        result = report["results"][0]
        self.assertEqual(result["cases_loaded"], 2)
        self.assertEqual(result["retrieval"]["evaluated_cases"], 2)
        self.assertEqual(report["config"]["case_offset"], 1)
        self.assertEqual(report["config"]["case_limit"], 2)

    def test_faiss_backend_is_selected_for_dataset_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_json(
                root / "pubmedqa" / "ori_pqal.json",
                {
                    "q1": {
                        "QUESTION": "Which intervention helped?",
                        "CONTEXTS": ["The intervention improved outcomes."],
                        "final_decision": "yes",
                    }
                },
            )
            self._write_json(root / "pubmedqa" / "test_ground_truth.json", {"q1": "yes"})
            fake_configuration = SimpleNamespace(
                retrieval=SimpleNamespace(backend="faiss", embedding=object())
            )
            with (
                patch(
                    "medical_agent.dataset_evaluation.load_model_configuration",
                    return_value=fake_configuration,
                ),
                patch(
                    "medical_agent.dataset_evaluation._build_embedding_provider",
                    return_value=object(),
                ),
                patch(
                    "medical_agent.dataset_evaluation.FaissKnowledgeBase",
                    side_effect=lambda source, **_kwargs: source,
                ) as faiss_builder,
            ):
                report = run_dataset_evaluation(
                    datasets=["pubmedqa"],
                    data_root=root,
                    mode="retrieval",
                    max_cases=1,
                    retrieval_backend="faiss",
                )

        self.assertEqual(report["config"]["retrieval_backend"], "faiss")
        self.assertEqual(report["results"][0]["retrieval_backend"], "faiss")
        faiss_builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
