"""Tests for the dataset registry, synthetic fixture and repository audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from medical_agent.evaluation import (
    EvaluationAssetError,
    load_dataset_manifest,
    run_local_smoke_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    module_path = ROOT / "scripts" / "audit_repository.py"
    spec = importlib.util.spec_from_file_location("medical_agent_repo_audit", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("无法加载仓库审计模块")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationAssetsTests(unittest.TestCase):
    def test_manifest_has_reproducible_metadata_and_access_boundaries(self) -> None:
        manifest = load_dataset_manifest(ROOT / "evaluation" / "datasets.json")
        datasets = manifest["datasets"]
        self.assertGreaterEqual(len(datasets), 10)
        self.assertEqual(len({item["id"] for item in datasets}), len(datasets))
        self.assertIn("credentialed", {item["access"] for item in datasets})
        self.assertIn("human_only", {item["access"] for item in datasets})
        self.assertTrue(all(item["official_url"].startswith("http") for item in datasets))

    def test_manifest_rejects_unknown_access_level(self) -> None:
        path = ROOT / "evaluation" / "datasets.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["datasets"][0]["access"] = "automatic_download_everything"
        with self.assertRaises(EvaluationAssetError):
            # The validator is intentionally strict even for local metadata.
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "manifest.json"
                candidate.write_text(json.dumps(payload), encoding="utf-8")
                load_dataset_manifest(candidate)

    def test_synthetic_smoke_evaluation_uses_existing_quality_contracts(self) -> None:
        result = run_local_smoke_evaluation(
            ROOT / "evaluation" / "datasets.json",
            ROOT / "evaluation" / "smoke_cases.json",
        )
        self.assertTrue(result["fixture"]["synthetic"])
        self.assertEqual(result["retrieval"]["evaluated_cases"], 3)
        self.assertGreater(result["retrieval"]["metrics"]["recall@5"], 0.0)
        self.assertEqual(result["evidence_chain"]["citation_coverage"], 1.0)
        self.assertEqual(result["counterfactual"]["metrics"]["regression_rate"], 0.0)
        self.assertNotIn("用药安全", json.dumps(result, ensure_ascii=False))

    def test_repository_audit_is_read_only_and_reports_inventory(self) -> None:
        audit = _load_audit_module()
        result = audit.run_repository_audit(ROOT)
        self.assertIn("repo", result)
        self.assertIn("inventory", result)
        self.assertTrue(result["static_checks"]["evaluation_manifest_present"])
        self.assertGreaterEqual(result["inventory"]["test_files"], 20)
        self.assertIsInstance(result["static_checks"]["secret_findings"], list)


if __name__ == "__main__":
    unittest.main()
