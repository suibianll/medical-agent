"""HTTP contracts used by the local knowledge-base chat workbench."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.request import Request, urlopen


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_agent.demo_model import DemoModelAdapter
from medical_agent.retrieval import JsonKnowledgeBase
from medical_agent.server import MedicalAgentRequestHandler, ThreadingHTTPServer
from medical_agent.service import MedicalAgentService


class MedicalAgentHttpApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        service = MedicalAgentService(
            model_profiles={"demo": DemoModelAdapter()},
            default_model_profile="demo",
            knowledge_base=JsonKnowledgeBase(
                [], storage_path=Path(cls.temporary_directory.name) / "imports.json"
            ),
            max_workers=1,
        )

        class TestHandler(MedicalAgentRequestHandler):
            # Isolate this server from environment-provided model credentials
            # and from the repository's persistent imported demo documents.
            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                pass

        TestHandler.service = service
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temporary_directory.cleanup()

    @classmethod
    def _json_request(cls, path: str, payload: dict | None = None) -> tuple[int, dict]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
        request = Request(
            f"{cls.base_url}{path}",
            data=body,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json; charset=utf-8"} if body else {},
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310 - loopback test server
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_root_serves_the_chat_workbench(self) -> None:
        with urlopen(f"{self.base_url}/", timeout=5) as response:  # noqa: S310
            page = response.read().decode("utf-8")
        self.assertIn('id="knowledge-form"', page)
        self.assertIn('id="chat-form"', page)
        self.assertIn('id="model-profile"', page)
        self.assertIn('id="evidence-graph"', page)
        self.assertIn('src="/app.js"', page)
        with urlopen(f"{self.base_url}/app.js", timeout=5) as response:  # noqa: S310
            script = response.read().decode("utf-8")
        self.assertIn('const CHAT_URL = "/api/chat"', script)
        self.assertIn('const KNOWLEDGE_IMPORT_URL = "/api/knowledge/import"', script)

        status, health = self._json_request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["models"]["default"], "demo")
        self.assertEqual(health["models"]["profiles"][0]["id"], "demo")
        self.assertNotIn("api_key", json.dumps(health))
        self.assertNotIn("base_url", json.dumps(health))

    def test_evidence_page_exposes_sentence_proofs_and_task_dependencies(self) -> None:
        with urlopen(f"{self.base_url}/evidence.html", timeout=5) as response:  # noqa: S310
            page = response.read().decode("utf-8")
        self.assertIn('id="sentence-proof-list"', page)
        self.assertIn('id="sentence-coverage"', page)
        self.assertIn('id="task-dependency-map"', page)
        self.assertIn('id="run-dag"', page)

        with urlopen(f"{self.base_url}/evidence.js", timeout=5) as response:  # noqa: S310
            script = response.read().decode("utf-8")
        self.assertIn("function renderSentenceProof", script)
        self.assertIn("function renderTaskDependencyMap", script)

    def test_import_then_chat_returns_citations_and_graph(self) -> None:
        status, before = self._json_request("/api/knowledge")
        self.assertEqual(status, 200)
        self.assertEqual(before["count"], 0)

        status, imported = self._json_request(
            "/api/knowledge/import",
            {
                "name": "api-guide.txt",
                "content": "API-ORBIT-42 指南要求一般医学信息保留可追溯知识库引用。",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(imported["imported"]["chunks_added"], 1)
        self.assertEqual(imported["count"], 1)

        status, chat = self._json_request(
            "/api/chat",
            {
                "message": "API-ORBIT-42 指南要求什么？",
                "history": [{"role": "user", "content": "请基于资料回答。"}],
                "modelProfile": "demo",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(chat["status"], "passed")
        self.assertIn("[K", chat["answer"])
        self.assertTrue(chat["claims"])
        self.assertTrue(chat["evidence"])
        self.assertTrue(chat["graph"]["nodes"])
        self.assertEqual(chat["run"]["model"]["profile"], "demo")
        self.assertTrue(
            all(ref.startswith("K") for claim in chat["claims"] for ref in claim["refs"])
        )


if __name__ == "__main__":
    unittest.main()
