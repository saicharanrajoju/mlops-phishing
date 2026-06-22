"""API smoke tests using FastAPI's TestClient with a stubbed model.

These run fully offline: we inject a deterministic fake model so the scan path
does not depend on a trained registry artifact.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as appmod


class _StubModel:
    """Always predicts phishing (class 0) with high probability."""

    classes_ = np.array([0, 1])

    def predict_proba(self, df):
        return np.array([[0.92, 0.08]])


@pytest.fixture
def client():
    with TestClient(appmod.app) as c:
        # Override whatever was loaded at startup with a deterministic stub.
        appmod.model_holder.model = _StubModel()
        appmod.model_holder.source = "test-stub"
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_ready" in body


def test_scan_returns_phishing_verdict(client):
    resp = client.post("/api/scan", json={"url": "http://secure-paypal@127.0.0.1/login"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["prediction"] == 0  # 0 = phishing
    assert body["probability"] == pytest.approx(0.92, abs=1e-3)
    assert len(body["features"]) == 30
    assert "ai_analysis" in body


def test_scan_rejects_empty_url(client):
    resp = client.post("/api/scan", json={"url": "   "})
    assert resp.status_code == 400
