"""Tests for the AI analyst fallback + structured report rendering."""

from ai_analyst.gemini_analyst import GeminiAnalyst
from ai_analyst.schemas import ThreatReport


def test_threat_report_renders_markdown():
    report = ThreatReport(
        risk_level="HIGH",
        summary="Looks malicious.",
        structural_signals=["having_IP_Address", "having_At_Symbol"],
        typosquatting_risk="High brand-impersonation risk.",
        recommendations=["Do not enter credentials."],
    )
    md = report.to_markdown(probability=0.91)
    assert "### " in md
    assert "`HIGH`" in md
    assert "91.0%" in md
    assert "having_IP_Address" in md


def test_fallback_report_used_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    analyst = GeminiAnalyst()
    assert analyst.active is False  # no live client

    features = {"having_IP_Address": -1, "Prefix_Suffix": -1, "SSLfinal_State": 1}
    md = analyst.generate_threat_report("http://bad@1.2.3.4", prediction=0, probability=0.8, features=features)
    assert "L2 AI Threat Report" in md
    assert "80.0%" in md
    # Flagged (-1) features should surface as structural signals
    assert "having_IP_Address" in md


def test_chatbot_offline_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    analyst = GeminiAnalyst()
    reply = analyst.ask_chatbot("what is phishing?")
    assert "offline" in reply.lower()
