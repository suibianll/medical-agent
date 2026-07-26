"""HTTP contracts for runtime metadata and the chat progress SSE stream."""

from __future__ import annotations

import json
from threading import Thread
import unittest
from urllib.request import Request, urlopen


from medical_agent.demo_model import DemoModelAdapter
from medical_agent.application.agent import MedicalAgent
from medical_agent.bootstrap import create_agent
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.server import MedicalAgentHTTPServer, MedicalAgentRequestHandler


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse the small, JSON-only SSE dialect exposed by the local server."""

    events: list[tuple[str, dict]] = []
    event_name = "message"
    data_lines: list[str] = []

    def finish_event() -> None:
        nonlocal event_name, data_lines
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
        event_name = "message"
        data_lines = []

    for line in body.splitlines():
        if not line:
            finish_event()
        elif line.startswith(":"):
            # SSE heartbeat/comment; it is intentionally not a UI event.
            continue
        elif line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data_lines.append(line.partition(":")[2].lstrip())
    finish_event()
    return events


class _FailingDemoModel(DemoModelAdapter):
    def plan(self, request: str, patient_record: str) -> dict:
        raise RuntimeError("provider diagnostic: test-key-must-not-be-exposed")


class MedicalAgentSseApiTests(unittest.TestCase):
    @staticmethod
    def _start_server(service: MedicalAgent) -> tuple[MedicalAgentHTTPServer, Thread, str]:
        class TestHandler(MedicalAgentRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                pass

        server = MedicalAgentHTTPServer(
            ("127.0.0.1", 0), service, handler_class=TestHandler
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, f"http://127.0.0.1:{server.server_port}"

    @staticmethod
    def _stop_server(server: MedicalAgentHTTPServer, thread: Thread) -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    @staticmethod
    def _sse_request(base_url: str, payload: dict) -> tuple[str, list[tuple[str, dict]]]:
        request = Request(
            f"{base_url}/api/chat/stream",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "text/event-stream",
            },
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - loopback test server
            return response.headers["Content-Type"], _parse_sse(
                response.read().decode("utf-8")
            )

    def test_health_reports_demo_model_mode(self) -> None:
        service = create_agent(
            model_profiles={"test": DemoModelAdapter()},
            default_model_profile="test",
            knowledge_base=JsonKnowledgeBase([]),
        )
        server, thread, base_url = self._start_server(service)
        try:
            with urlopen(f"{base_url}/api/health", timeout=5) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            self._stop_server(server, thread)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "medical-agent-mvp")
        self.assertEqual(payload["model"]["mode"], "demo")
        self.assertEqual(payload["model"]["provider"], "local-demo")
        self.assertEqual(payload["model"]["name"], "demo")

    def test_chat_stream_has_safe_progress_events_then_full_result(self) -> None:
        knowledge_base = JsonKnowledgeBase([])
        knowledge_base.import_text(
            name="sse-guide.txt",
            content=(
                "SSE-ORBIT-42 requires a traceable, evidence-backed general "
                "medical information response."
            ),
        )
        service = create_agent(
            model_profiles={"test": DemoModelAdapter()},
            default_model_profile="test",
            knowledge_base=knowledge_base,
            max_workers=1,
        )
        server, thread, base_url = self._start_server(service)
        try:
            content_type, events = self._sse_request(
                base_url,
                {"message": "What does SSE-ORBIT-42 require?"},
            )
        finally:
            self._stop_server(server, thread)

        self.assertTrue(content_type.startswith("text/event-stream"))
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[-1][0], "result")
        self.assertNotIn("error", [event_name for event_name, _ in events])

        progress = [payload for event_name, payload in events if event_name == "progress"]
        self.assertTrue(progress)
        self.assertEqual(
            [item["sequence"] for item in progress], list(range(1, len(progress) + 1))
        )
        self.assertIn("planning", {item["stage"] for item in progress})
        self.assertIn("retrieve", {item["stage"] for item in progress})
        self.assertIn("completed", {item["stage"] for item in progress})

        allowed_progress_fields = {
            "stage",
            "message",
            "run_id",
            "sequence",
            "timestamp",
            "task_id",
            "task",
            "tasks",
            "status",
            "queries",
            "evidence_ids",
            "facts",
            "claims",
            "evaluation",
            "round",
            "counts",
        }
        self.assertTrue(
            all(set(item).issubset(allowed_progress_fields) for item in progress)
        )
        self.assertTrue(all(isinstance(item["message"], str) for item in progress))

        result = events[-1][1]
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["answer"])
        self.assertTrue(result["claims"])
        self.assertTrue(result["graph"]["nodes"])

    def test_stream_error_is_sanitized_and_has_no_result_event(self) -> None:
        service = create_agent(
            model_profiles={"test": _FailingDemoModel()},
            default_model_profile="test",
            knowledge_base=JsonKnowledgeBase([]),
        )
        server, thread, base_url = self._start_server(service)
        try:
            _content_type, events = self._sse_request(
                base_url,
                {"message": "Trigger the failing test model."},
            )
        finally:
            self._stop_server(server, thread)

        self.assertEqual(events[-1][0], "error")
        self.assertNotIn("result", [event_name for event_name, _ in events])
        error = events[-1][1]
        self.assertEqual(set(error), {"message"})
        self.assertTrue(error["message"])
        self.assertNotIn("test-key-must-not-be-exposed", json.dumps(events, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
