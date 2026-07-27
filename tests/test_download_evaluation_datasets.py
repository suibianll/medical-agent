"""Tests for the data-driven evaluation dataset downloader."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_downloader():
    module_path = ROOT / "scripts" / "download_evaluation_datasets.py"
    spec = importlib.util.spec_from_file_location("medical_agent_dataset_downloader", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("无法加载评测数据下载器")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationDatasetDownloaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.downloader = _load_downloader()

    def test_source_manifest_contains_official_and_mirror_provenance(self) -> None:
        records = self.downloader._load_manifest(ROOT / "evaluation" / "download_sources.json")
        by_id = {record["dataset_id"]: record for record in records}
        self.assertIn("pubmedqa", by_id)
        self.assertEqual(by_id["pubmedqa"]["method"], "http_files")
        self.assertIn("pubmedqa-hf-mirror", by_id)
        self.assertIn("pubmedqa-hf-a-mirror", by_id)
        self.assertIn("medmcqa-hf-mirror", by_id)
        self.assertTrue(by_id["medmcqa-hf-mirror"]["mirror_url"].startswith("https://"))

    def test_manual_and_credentialed_sources_do_not_download(self) -> None:
        records = self.downloader._load_manifest(ROOT / "evaluation" / "download_sources.json")
        selected = {
            record["dataset_id"]: record
            for record in records
            if record["dataset_id"] in {"bioasq", "mednli"}
        }
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            manual = self.downloader._download_record(
                selected["bioasq"], output_root, allow_terms_check=True, max_bytes=1024
            )
            credentialed = self.downloader._download_record(
                selected["mednli"], output_root, allow_terms_check=True, max_bytes=1024
            )
        self.assertEqual(manual["status"], "manual_required")
        self.assertEqual(credentialed["status"], "blocked_access")

    def test_terms_check_requires_explicit_confirmation(self) -> None:
        records = self.downloader._load_manifest(ROOT / "evaluation" / "download_sources.json")
        faithfulness = next(
            record for record in records if record["dataset_id"] == "faithfulness-qa-2026"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = self.downloader._download_record(
                faithfulness, Path(directory), allow_terms_check=False, max_bytes=1024
            )
        self.assertEqual(result["status"], "terms_confirmation_required")

    def test_http_file_path_cannot_escape_dataset_directory(self) -> None:
        record = {
            "dataset_id": "path-check",
            "access": "downloadable",
            "method": "http_files",
            "files": [{"url": "https://example.invalid/file", "path": "../escape"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = self.downloader._download_record(
                record, Path(directory), allow_terms_check=True, max_bytes=1024
            )
        self.assertEqual(result["status"], "failed")
        self.assertIn("下载路径", result["message"])

    def test_source_manifest_is_valid_json(self) -> None:
        payload = json.loads(
            (ROOT / "evaluation" / "download_sources.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertGreaterEqual(len(payload["downloads"]), 10)


if __name__ == "__main__":
    unittest.main()
