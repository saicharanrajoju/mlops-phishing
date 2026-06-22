"""Pydantic schemas for structured Gemini output.

Treating the LLM's output as an untrusted, schema-validated payload (rather than
free text we string-parse) is the core "modern AI integration" pattern here.
"""

from pydantic import BaseModel, Field


class ThreatReport(BaseModel):
    """Structured L2 threat analysis returned by Gemini in JSON mode."""

    risk_level: str = Field(description="One of: CRITICAL, HIGH, MEDIUM, LOW, SAFE")
    summary: str = Field(description="A 2-3 sentence executive summary of the verdict and why.")
    structural_signals: list[str] = Field(
        default_factory=list,
        description="Specific URL/structural features that drove the verdict.",
    )
    typosquatting_risk: str = Field(
        default="",
        description="Short assessment of brand-impersonation / typosquatting risk.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable defense steps for a user encountering this URL.",
    )

    def to_markdown(self, probability: float) -> str:
        """Render the structured report into the markdown the dashboard expects."""
        signals = "\n".join(f"* {s}" for s in self.structural_signals) or "* None detected."
        recs = "\n".join(f"* {r}" for r in self.recommendations) or "* Stay vigilant."
        return (
            f"### 🛡️ L2 AI Threat Report\n"
            f"**Verdict**: `{self.risk_level}` (ML Risk Score: `{probability * 100:.1f}%`)\n\n"
            f"{self.summary}\n\n"
            f"**Structural Signals**\n{signals}\n\n"
            f"**Typosquatting / Impersonation**\n{self.typosquatting_risk or 'Not assessed.'}\n\n"
            f"**Defense Recommendations**\n{recs}\n"
        )
