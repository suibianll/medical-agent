"""Black-box contracts for report templates, run archive APIs and safe logs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from threading import Thread
import unittest
from urllib.request import Request, urlopen


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_agent.demo_model import DemoModelAdapter
from medical_agent.audit_log import SafeAuditLogger
from medical_agent.report import render_report
from medical_agent.retrieval import JsonKnowledgeBase
from medical_agent.run_archive import InMemoryRunArchive
from medical_agent.server import MedicalAgentRequestHandler, ThreadingHTTPServer
from medical_agent.service import MedicalAgentService


class _CaptureLogger:
    """Minimal logger that lets the HTTP test inspect access-log safety."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)


class _TwoSentenceDemoModel(DemoModelAdapter):
    """Deterministic multi-sentence claims for sentence-level citation tests."""

    def synthesize(
        self,
        *,
        task: dict,
        request: str,
        facts: list[dict[str, str]],
        evidence: list[dict[str, str]],
        upstream: dict[int, object],
    ) -> dict:
        patient = next((item for item in evidence if item["id"].startswith("P")), None)
        knowledge = next((item for item in evidence if item["id"].startswith("K")), None)
        if not knowledge:
            return {"claims": [], "unknowns": ["No knowledge evidence was retrieved."]}
        refs = [knowledge["id"]]
        if patient:
            refs.insert(0, patient["id"])
        task_id = task["id"]
        return {
            "claims": [
                {
                    "text": (
                        f"Task {task_id} first conclusion sentence. "
                        f"Task {task_id} second conclusion sentence."
                    ),
                    "refs": refs,
                }
            ],
            "unknowns": [],
        }


class ReportTemplateAndArchiveApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        knowledge_base = JsonKnowledgeBase([])
        knowledge_base.import_text(
            name="archive-guide.txt",
            content=(
                "ARCHIVE-ORBIT-42 requires traceable evidence citations in every "
                "clinical support conclusion and a human review boundary."
            ),
        )
        service = MedicalAgentService(
            model=_TwoSentenceDemoModel(), knowledge_base=knowledge_base, max_workers=1
        )

        class TestHandler(MedicalAgentRequestHandler):
            pass

        TestHandler.service = service
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        cls.logger = _CaptureLogger()
        cls.server.logger = cls.logger  # type: ignore[attr-defined]
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    @classmethod
    def _json_request(
        cls, path: str, payload: dict | None = None
    ) -> tuple[int, dict]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
        request = Request(
            f"{cls.base_url}{path}",
            data=body,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json; charset=utf-8"} if body else {},
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - loopback test server
            return response.status, json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _claim_sentences(claim: dict) -> list[str]:
        text = " ".join(str(claim["text"]).split())
        return [
            sentence.strip()
            for sentence in re.findall(r"[^。！？.!?]+[。！？.!?]|[^。！？.!?]+$", text)
            if sentence.strip()
        ]

    @classmethod
    def _assert_claim_sentences_are_cited(cls, payload: dict) -> None:
        """Every rendered conclusion must finish with the refs of its source claim."""

        report = payload["report"]
        for rendered_text in (report["text"], report["markdown"]):
            for claim in payload["claims"]:
                expected_citations = "".join(f"[{ref}]" for ref in claim["refs"])
                assert expected_citations, f"claim {claim['id']} must have references"
                for sentence in cls._claim_sentences(claim):
                    position = rendered_text.find(sentence)
                    assert position >= 0, (
                        f"report must render sentence {sentence!r} from {claim['id']}"
                    )
                    following = rendered_text[position + len(sentence) :].lstrip()
                    assert following.startswith(expected_citations), (
                        f"sentence {sentence!r} from {claim['id']} must end with citations"
                    )

    @staticmethod
    def _all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(
                *(ReportTemplateAndArchiveApiTests._all_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(
                *(ReportTemplateAndArchiveApiTests._all_keys(item) for item in value)
            ) if value else set()
        return set()

    def _create_run(self, report_template: str | dict) -> dict:
        status, result = self._json_request(
            "/api/runs",
            {
                "request": "Evaluate ARCHIVE-ORBIT-42 evidence requirements.",
                "patientRecord": "Patient has a medication review request and needs follow-up.",
                "reportTemplate": report_template,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["report"]["text"])
        return result

    def test_builtin_templates_render_cited_conclusion_sentences(self) -> None:
        default_report = self._create_run("evidence_summary")
        task_trace_report = self._create_run("task_trace")
        handoff_report = self._create_run("handoff")

        for expected_template, result in (
            ("evidence_summary", default_report),
            ("task_trace", task_trace_report),
            ("handoff", handoff_report),
        ):
            self._assert_claim_sentences_are_cited(result)
            self.assertNotEqual(result["report"]["text"], result["report"]["markdown"])
            self.assertEqual(result["report"]["template"]["name"], expected_template)

    def test_object_template_is_accepted_by_run_and_chat(self) -> None:
        template = {"name": "handoff", "title": "Unit Test Handoff"}
        run_result = self._create_run(template)
        self._assert_claim_sentences_are_cited(run_result)
        self.assertIn("Unit Test Handoff", run_result["report"]["text"])

        status, chat_result = self._json_request(
            "/api/chat",
            {
                "message": "What does ARCHIVE-ORBIT-42 require?",
                "reportTemplate": template,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(chat_result["status"], "passed")
        self._assert_claim_sentences_are_cited(chat_result)
        self.assertIn("Unit Test Handoff", chat_result["report"]["text"])
        for claim in chat_result["claims"]:
            citations = "".join(f"[{ref}]" for ref in claim["refs"])
            for sentence in self._claim_sentences(claim):
                position = chat_result["answer"].find(sentence)
                self.assertGreaterEqual(position, 0)
                self.assertTrue(
                    chat_result["answer"][position + len(sentence) :]
                    .lstrip()
                    .startswith(citations),
                    msg=f"chat sentence {sentence!r} must carry its refs",
                )

    def test_completed_run_can_be_retrieved_with_safe_event_summaries(self) -> None:
        marker = "PRIVATE-RUNTIME-EVENT-MARKER-42"
        status, created = self._json_request(
            "/api/runs",
            {
                "request": f"Evaluate evidence requirements for {marker}.",
                "patientRecord": f"Patient record includes {marker}.",
                "reportTemplate": "evidence_summary",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(created["status"], "passed")
        run_id = created["run"]["id"]

        status, archived = self._json_request(f"/api/runs/{run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(archived["run"]["id"], run_id)
        self.assertIn("archive", archived)
        self.assertEqual(archived["report"]["text"], created["report"]["text"])

        status, events_by_path = self._json_request(f"/api/runs/{run_id}/events")
        self.assertEqual(status, 200)
        status, events_by_query = self._json_request(f"/api/events?run_id={run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(events_by_path["run_id"], run_id)
        self.assertEqual(events_by_query["run_id"], run_id)
        self.assertTrue(events_by_path["events"])
        self.assertEqual(events_by_path["events"], events_by_query["events"])

        forbidden_keys = {
            "api_key",
            "authorization",
            "base_url",
            "patient_record",
            "raw_model_response",
            "reasoning",
            "chain_of_thought",
            "request",
            "query",
            "queries",
            "facts",
        }
        self.assertTrue(forbidden_keys.isdisjoint(self._all_keys(events_by_path)))
        self.assertNotIn(marker, json.dumps(events_by_path, ensure_ascii=False))
        for event in events_by_path["events"]:
            self.assertNotIn("queries", event)
            self.assertNotIn("facts", event)
            self.assertNotIn("claims", event)
            if "task" in event:
                self.assertTrue(set(event["task"]).issubset({"id", "deps"}))
            if "tasks" in event:
                self.assertTrue(
                    all(set(task).issubset({"id", "deps"}) for task in event["tasks"])
                )
            if "counts" in event:
                self.assertNotIn("claims", event["counts"])
                self.assertTrue(
                    set(event["counts"]).issubset(
                        {"tasks", "evidence", "claim_count"}
                    )
                )
        self.assertTrue(
            all(
                isinstance(event.get("stage"), str)
                and isinstance(event.get("message"), str)
                for event in events_by_path["events"]
            )
        )

    def test_access_log_does_not_echo_patient_record_or_request_body(self) -> None:
        marker = "PATIENT-PRIVATE-MARKER-ARCHIVE-42"
        self._create_run("evidence_summary")
        self._json_request(
            "/api/runs",
            {
                "request": f"Request containing {marker}",
                "patientRecord": f"Patient record containing {marker}",
                "reportTemplate": "evidence_summary",
            },
        )

        self.assertTrue(self.logger.messages)
        self.assertNotIn(marker, "\n".join(self.logger.messages))


class ReportRendererCitationTests(unittest.TestCase):
    def test_uncited_multisentence_claim_marks_every_sentence_as_uncited(self) -> None:
        report = render_report(
            request="Unit report request",
            task_states={
                1: {
                    "task": {"id": 1, "goal": "Unit task", "deps": []},
                    "status": "completed",
                    "result": {},
                }
            },
            claims=[
                {
                    "id": "C1",
                    "text": "First unsupported sentence. Second unsupported sentence.",
                    "refs": [],
                }
            ],
            evidence=[],
            evaluation={"issues": []},
            status="needs_human_review",
            template="evidence_summary",
        )

        for text in (report["text"], report["markdown"]):
            self.assertIn("First unsupported sentence. [无引用]", text)
            self.assertIn("Second unsupported sentence. [无引用]", text)


class RunArchiveRedactionTests(unittest.TestCase):
    def test_persisted_audit_event_keeps_only_safe_summaries(self) -> None:
        archive = InMemoryRunArchive(max_runs=2, ttl_seconds=60)
        run_id = "run-safe-42"
        marker = "PRIVATE-PATIENT-AND-MODEL-TEXT-42"
        archive.start(run_id)
        archive.append_event(
            {
                "run_id": run_id,
                "stage": "retrieve",
                "message": marker,
                "status": "completed",
                "task": {"id": 3, "goal": marker, "deps": [1, 2]},
                "queries": [marker],
                "facts": [{"ref": "P1", "summary": marker}],
                "claims": [{"refs": ["P1", "K1"], "summary": marker}],
                "evidence_ids": ["P1", "K1", marker],
                "counts": {"tasks": 3, "claims": 1, "evidence": 2, "other": 99},
            }
        )

        events = archive.get_events(run_id)
        self.assertIsNotNone(events)
        event = events["events"][0]
        self.assertEqual(event["run_id"], run_id)
        self.assertEqual(event["stage"], "retrieve")
        self.assertEqual(event["task"], {"id": 3, "deps": [1, 2]})
        self.assertEqual(event["fact_refs"], ["P1"])
        self.assertEqual(event["claim_refs"], [["P1", "K1"]])
        self.assertEqual(event["counts"], {"tasks": 3, "claims": 1, "evidence": 2})
        self.assertNotIn(marker, json.dumps(events, ensure_ascii=False))
        self.assertTrue(
            {
                "queries",
                "facts",
                "goal",
                "summary",
            }.isdisjoint(ReportTemplateAndArchiveApiTests._all_keys(events))
        )


class SafeAuditLoggerTests(unittest.TestCase):
    def test_runtime_log_line_is_redacted_before_reaching_logger(self) -> None:
        capture = _CaptureLogger()
        logger = SafeAuditLogger(capture)  # type: ignore[arg-type]
        marker = "PRIVATE-RUNTIME-LOGGER-MARKER-42"

        logger.record(
            {
                "run_id": "run-log-42",
                "stage": "synthesize",
                "status": "completed",
                "message": marker,
                "queries": [marker],
                "facts": [{"ref": "P1", "summary": marker}],
                "claims": [{"refs": ["P1", "K1"], "summary": marker}],
                "counts": {"tasks": 2, "claims": 1, "evidence": 2},
            }
        )

        self.assertEqual(len(capture.messages), 1)
        line = capture.messages[0]
        self.assertTrue(line.startswith("medical_agent_audit "))
        self.assertNotIn(marker, line)
        payload = json.loads(line.partition(" ")[2])
        self.assertEqual(payload["run_id"], "run-log-42")
        self.assertEqual(payload["stage"], "synthesize")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["reference_count"], 3)
        self.assertEqual(payload["counts"], {"tasks": 2, "claims": 1, "evidence": 2})
        self.assertTrue(
            set(payload).issubset(
                {
                    "run_id",
                    "stage",
                    "status",
                    "task_id",
                    "reference_count",
                    "round",
                    "counts",
                    "evaluation",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
