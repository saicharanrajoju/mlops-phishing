"""Tests for the agentic investigation, uncertainty flagging, and feedback loop."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as appmod
from ai_analyst.threat_agent import ThreatHuntingAgent


def test_confidence_levels():
    assert appmod._confidence(0.95)["confidence"] == "HIGH"
    assert appmod._confidence(0.95)["needs_review"] is False
    low = appmod._confidence(0.52)
    assert low["confidence"] == "LOW"
    assert low["needs_review"] is True
    assert appmod._confidence(0.70)["confidence"] == "MEDIUM"


def test_agent_offline_investigation():
    agent = ThreatHuntingAgent(classify_fn=lambda u: {"label": "phishing", "phishing_probability": 0.9})
    agent.active = False  # force offline path regardless of environment
    res = agent.investigate("http://bad@1.2.3.4")
    assert res["agentic"] is False
    assert "offline" in res["report"].lower()
    assert res["classification"]["label"] == "phishing"
    assert res["tools_used"] == []


class _Stub:
    classes_ = np.array([0, 1])

    def predict_proba(self, df):
        return np.array([[0.9, 0.1]])


@pytest.fixture
def client():
    with TestClient(appmod.app) as c:
        appmod.model_holder.model = _Stub()
        appmod.model_holder.source = "test"
        appmod.threat_agent.active = False  # keep investigation offline/hermetic in tests
        yield c


def test_investigate_endpoint(client):
    r = client.post("/api/investigate", json={"url": "http://bad@1.2.3.4"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "report" in body
    assert "tools_used" in body


def test_feedback_endpoint_records(client):
    r = client.post(
        "/api/feedback",
        json={"url": "http://x@1.2.3.4", "correct_label": 0, "predicted_label": 1},
    )
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_feedback_rejects_invalid_label(client):
    r = client.post("/api/feedback", json={"url": "http://x", "correct_label": 5})
    assert r.status_code == 400
