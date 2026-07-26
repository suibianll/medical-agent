"""Transport boundary validation and resource-budget contracts."""

from __future__ import annotations

import unittest

from medical_agent.bootstrap import create_agent
from medical_agent.server import MedicalAgentHTTPServer, is_loopback_host
from medical_agent.transport.validation import (
    MAX_HISTORY_ITEMS,
    MAX_REQUEST_CHARS,
    validate_chat_payload,
    validate_run_payload,
)


class TransportBoundaryTests(unittest.TestCase):
    def test_only_loopback_bind_addresses_are_supported(self) -> None:
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertFalse(is_loopback_host("0.0.0.0"))
        self.assertFalse(is_loopback_host("192.168.1.10"))

    def test_run_and_chat_fields_are_bounded_before_dispatch(self) -> None:
        with self.assertRaises(ValueError):
            validate_run_payload({"request": "x" * (MAX_REQUEST_CHARS + 1)})
        with self.assertRaises(ValueError):
            validate_chat_payload({"message": 42})
        with self.assertRaises(ValueError):
            validate_chat_payload(
                {"message": "ok", "history": [{}] * (MAX_HISTORY_ITEMS + 1)}
            )

    def test_expensive_runs_have_a_global_concurrency_budget(self) -> None:
        server = MedicalAgentHTTPServer(
            ("127.0.0.1", 0), create_agent(), max_active_runs=1
        )
        try:
            self.assertTrue(server.try_acquire_run())
            self.assertFalse(server.try_acquire_run())
            server.release_run()
            self.assertTrue(server.try_acquire_run())
            server.release_run()
        finally:
            server.server_close()

    def test_server_port_cannot_be_shared_with_a_second_instance(self) -> None:
        first = MedicalAgentHTTPServer(("127.0.0.1", 0), create_agent())
        try:
            with self.assertRaises(OSError):
                MedicalAgentHTTPServer(
                    ("127.0.0.1", first.server_port), create_agent()
                )
        finally:
            first.server_close()
