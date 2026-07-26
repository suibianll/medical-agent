"""Dependency-free local HTTP server for the Medical Agent MVP."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from mimetypes import guess_type
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .retrieval import KnowledgeImportError
from .service import MedicalAgentService

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = ROOT / "public"
MAX_BODY_BYTES = 2_000_000
REJECTED_RUN_STATUSES = frozenset({"rejected", "plan_rejected"})

SAMPLE_PAYLOAD = {
    "patientRecord": "患者，68岁。近期乏力，正在服用多种药物。病历记录 eGFR 约为 42 mL/min/1.73m²，既往有药物过敏史，近期肾功能尚未复查。",
    "request": "请评估当前情况中需要重点核实的用药安全问题，并给出下一步信息补全建议。",
}


class MedicalAgentRequestHandler(BaseHTTPRequestHandler):
    service: MedicalAgentService

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Avoid echoing request bodies or patient content into console logs.
        self.server.logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        """Start a no-buffer Server-Sent Events response."""

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.wfile.flush()

    def _write_sse(self, event: str, payload: dict[str, Any]) -> None:
        """Write one SSE frame; callers never pass model/provider raw text."""

        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        frame = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(frame)
        self.wfile.flush()

    def _serve_chat_stream(self, payload: dict[str, Any]) -> None:
        """Run chat off the request thread and stream safe audit milestones.

        Worker threads only enqueue events.  The HTTP request thread is the
        sole writer to ``wfile``, preventing interleaved/corrupt SSE frames
        when DAG tasks execute in parallel.
        """

        events: Queue[tuple[str, dict[str, Any] | None]] = Queue()

        def on_progress(event: dict[str, Any]) -> None:
            events.put(("progress", event))

        def run_chat() -> None:
            try:
                result = self.service.chat(
                    message=payload.get("message", ""),
                    patient_record=payload.get("patientRecord", ""),
                    history=payload.get("history", []),
                    report_template=payload.get("reportTemplate"),
                    model_profile=payload.get("modelProfile"),
                    on_progress=on_progress,
                )
                events.put(("result", result))
            except Exception:  # noqa: BLE001 - never stream provider diagnostics
                events.put(
                    (
                        "error",
                        {
                            "message": "模型或任务执行服务暂时不可用，请稍后重试。",
                        },
                    )
                )
            finally:
                events.put(("done", None))

        self._start_sse()
        Thread(target=run_chat, daemon=True, name="medical-agent-chat-stream").start()
        try:
            while True:
                try:
                    event, event_payload = events.get(timeout=15)
                except Empty:
                    # Keeps a browser/proxy connection alive during a slow real
                    # model call without adding a user-visible trace entry.
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                if event == "done":
                    return
                if event_payload is not None:
                    self._write_sse(event, event_payload)
        except OSError:
            # The task can finish independently; no request data is logged.
            return

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("无效的 Content-Length") from None
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("请求体为空或超过大小限制")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("请求体必须是 UTF-8 JSON") from None
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def _send_run_result(self, result: dict[str, Any]) -> None:
        """Map service-level validation failures to one consistent HTTP status."""

        status = (
            HTTPStatus.UNPROCESSABLE_ENTITY
            if result.get("status") in REJECTED_RUN_STATUSES
            else HTTPStatus.OK
        )
        self._send_json(result, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        run_path = parsed.path.rstrip("/")
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "medical-agent-mvp",
                    "model": self.service.model_metadata(),
                    "models": self.service.model_catalog(),
                }
            )
            return
        if parsed.path == "/api/sample":
            self._send_json(SAMPLE_PAYLOAD)
            return
        if parsed.path == "/api/knowledge":
            documents = self.service.list_knowledge()
            self._send_json({"documents": documents, "count": len(documents)})
            return
        if run_path == "/api/events":
            parameters = parse_qs(parsed.query)
            run_id = parameters.get("run_id", [""])[0]
            events = self.service.get_run_events(unquote(run_id))
            if events is None:
                self._send_json(
                    {
                        "error": "RUN_NOT_FOUND",
                        "message": "运行不存在、已过期或不在当前服务进程中。",
                    },
                    HTTPStatus.NOT_FOUND,
                )
                return
            self._send_json(events)
            return
        if run_path.startswith("/api/runs/"):
            segments = [
                unquote(part)
                for part in run_path[len("/api/runs/") :].split("/")
                if part
            ]
            if len(segments) == 1:
                result = self.service.get_run(segments[0])
                if result is None:
                    self._send_json(
                        {
                            "error": "RUN_NOT_FOUND",
                            "message": "运行不存在、已过期或不在当前服务进程中。",
                        },
                        HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json(result)
                return
            if len(segments) == 2 and segments[1] == "events":
                events = self.service.get_run_events(segments[0])
                if events is None:
                    self._send_json(
                        {
                            "error": "RUN_NOT_FOUND",
                            "message": "运行不存在、已过期或不在当前服务进程中。",
                        },
                        HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json(events)
                return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/runs":
                result = self.service.run(
                    request=payload.get("request", ""),
                    patient_record=payload.get("patientRecord", ""),
                    plan=payload.get("plan"),
                    report_template=payload.get("reportTemplate"),
                    model_profile=payload.get("modelProfile"),
                )
                self._send_run_result(result)
                return

            if parsed.path == "/api/chat/stream":
                self._serve_chat_stream(payload)
                return

            if parsed.path == "/api/chat":
                result = self.service.chat(
                    message=payload.get("message", ""),
                    patient_record=payload.get("patientRecord", ""),
                    history=payload.get("history", []),
                    report_template=payload.get("reportTemplate"),
                    model_profile=payload.get("modelProfile"),
                )
                self._send_run_result(result)
                return

            if parsed.path == "/api/knowledge/import":
                imported = self.service.import_knowledge(
                    name=payload.get("name", "导入资料.txt"),
                    content=payload.get("content", ""),
                )
                documents = self.service.list_knowledge()
                self._send_json(
                    {"imported": imported, "documents": documents, "count": len(documents)},
                    HTTPStatus.CREATED,
                )
                return

            self._send_json({"error": "NOT_FOUND"}, HTTPStatus.NOT_FOUND)
        except KnowledgeImportError as exc:
            self._send_json(
                {"error": "KNOWLEDGE_IMPORT_INVALID", "message": str(exc)},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except ValueError as exc:
            self._send_json({"error": "BAD_REQUEST", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:  # noqa: BLE001 - do not leak patient/context details
            self._send_json(
                {"error": "INTERNAL_ERROR", "message": "执行失败，请检查服务端日志。"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _serve_static(self, requested_path: str) -> None:
        relative = "index.html" if requested_path in {"", "/"} else requested_path.lstrip("/")
        candidate = (PUBLIC_DIR / relative).resolve()
        try:
            candidate.relative_to(PUBLIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "NOT_FOUND"}, HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self._send_json({"error": "NOT_FOUND"}, HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        mime, _ = guess_type(str(candidate))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def create_request_handler(
    service: MedicalAgentService,
) -> type[MedicalAgentRequestHandler]:
    """Bind one service instance without mutating global handler state."""

    class ConfiguredMedicalAgentRequestHandler(MedicalAgentRequestHandler):
        pass

    ConfiguredMedicalAgentRequestHandler.service = service
    return ConfiguredMedicalAgentRequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Medical Agent MVP locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = MedicalAgentService.from_environment()
    request_handler = create_request_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), request_handler)
    server.logger = logging.getLogger("medical_agent")  # type: ignore[attr-defined]
    print(f"Medical Agent MVP is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
