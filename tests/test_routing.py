from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from medical_agent.infrastructure.model_config import load_model_configuration
from medical_agent.bootstrap import create_agent
from medical_agent.demo_model import DemoModelAdapter
from medical_agent.retrieval.knowledge import JsonKnowledgeBase
from medical_agent.risk import (
    ExternalApiDecisionRouter,
    PatternDecisionRouter,
    route_decision,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class RoutingTests(unittest.TestCase):
    def test_config_parses_routing_mode_and_external_secret(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.local.json"
            path.write_text(
                json.dumps(
                    {
                        "routing": {
                            "mode": "rules",
                            "api_key_env": "ROUTER_KEY",
                            "rules": [
                                {
                                    "outcome": "emergency_escalation",
                                    "patterns": ["configured signal"],
                                    "emergency_signal": True,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            configuration = load_model_configuration(
                {"MEDICAL_AGENT_CONFIG": str(path), "ROUTER_KEY": "secret"}
            )
        self.assertEqual(configuration.routing.mode, "rules")
        self.assertEqual(configuration.routing.api_key, "secret")
        self.assertEqual(configuration.routing.rules[0]["patterns"], ["configured signal"])

    def test_configured_rule_can_escalate_without_code_keywords(self) -> None:
        router = PatternDecisionRouter(
            [
                {
                    "outcome": "emergency_escalation",
                    "patterns": [r"configured signal"],
                    "risk_level": "high",
                    "emergency_signal": True,
                    "reason": "configured_policy",
                }
            ]
        )
        decision = router.decide(
            request="configured signal",
            patient_record="",
            claims=[],
            evaluation={"pass": False, "issues": [{"code": "NO_REF"}]},
        )
        self.assertEqual(decision["outcome"], "emergency_escalation")
        self.assertEqual(decision["router"], "configured_rules")

    @patch("medical_agent.risk.urlopen")
    def test_external_router_is_structured_and_keeps_local_defer_gate(self, urlopen) -> None:
        urlopen.return_value = _Response(
            {
                "outcome": "answer",
                "risk_level": "standard",
                "emergency_signal": False,
            }
        )
        router = ExternalApiDecisionRouter(
            endpoint="https://router.example.invalid/v1/route",
            api_key="secret",
        )
        decision = router.decide(
            request="request",
            patient_record="patient",
            claims=[{"id": "C1"}],
            evaluation={"pass": False, "issues": [{"code": "NO_REF"}]},
        )
        self.assertEqual(decision["outcome"], "defer")
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("patient_record", payload)
        self.assertEqual(payload["context"]["issue_codes"], ["NO_REF"])

    def test_route_decision_compatibility_entry_point_uses_evidence_policy(self) -> None:
        decision = route_decision(
            request="arbitrary text",
            patient_record="",
            claims=[],
            evaluation={"pass": True, "issues": []},
        )
        self.assertEqual(decision["outcome"], "ask_clarification")
        self.assertEqual(decision["router"], "evidence")

    def test_workflow_uses_router_injected_by_composition_root(self) -> None:
        class _FixedRouter:
            def decide(self, **_kwargs):
                return {
                    "outcome": "defer",
                    "risk_level": "high",
                    "emergency_signal": True,
                    "reasons": ["test_router"],
                    "missing": [],
                    "issue_codes": [],
                    "router": "test",
                }

        service = create_agent(
            model_profiles={"demo": DemoModelAdapter()},
            knowledge_base=JsonKnowledgeBase(
                [{"id": "K1", "title": "guide", "text": "support"}]
            ),
            decision_router=_FixedRouter(),
        )
        result = service.chat(message="support")

        self.assertEqual(result["run"]["decision"]["router"], "test")


if __name__ == "__main__":
    unittest.main()
