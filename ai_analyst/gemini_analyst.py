"""
Gemini L2 threat analyst.

Design principles (treat the LLM as an unreliable network dependency):
  * Structured output  - Gemini returns JSON validated against `ThreatReport`.
  * Caching            - identical scans reuse the report (free + instant).
  * Graceful fallback  - if the key is missing or the API errors/validation fails,
                         we return a deterministic templated report from the
                         features. A scan must NEVER fail because the LLM hiccupped.
  * Timeout            - a slow API can't hang the request path.
"""

import hashlib
import json
import os
from collections import OrderedDict

from dotenv import load_dotenv

from ai_analyst.schemas import ThreatReport
from phishsentinel.logging.logger import logging

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "20000"))
CACHE_MAX_SIZE = 256


class GeminiAnalyst:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.active = False
        self._report_cache: OrderedDict[str, str] = OrderedDict()

        # Treat obvious placeholder keys as "no key" so demos default to fallback mode.
        if self.api_key and "your-" not in self.api_key:
            try:
                from google import genai
                from google.genai import types

                try:
                    self.client = genai.Client(
                        api_key=self.api_key,
                        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                    )
                except Exception:
                    # Older/newer SDKs may differ on HttpOptions; fall back to default client.
                    self.client = genai.Client(api_key=self.api_key)
                self._types = types
                self.active = True
                logging.info("Google GenAI (Gemini) client initialized.")
            except Exception as e:
                logging.error(f"Error initializing Gemini client: {e}")
        else:
            logging.warning("GEMINI_API_KEY not set; AI analyst runs in deterministic fallback mode.")

    # ------------------------------------------------------------------ #
    # Caching helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(prediction: int, features: dict) -> str:
        payload = json.dumps({"p": prediction, "f": features}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str):
        if key in self._report_cache:
            self._report_cache.move_to_end(key)
            return self._report_cache[key]
        return None

    def _cache_put(self, key: str, value: str):
        self._report_cache[key] = value
        self._report_cache.move_to_end(key)
        while len(self._report_cache) > CACHE_MAX_SIZE:
            self._report_cache.popitem(last=False)

    # ------------------------------------------------------------------ #
    # Threat report
    # ------------------------------------------------------------------ #
    def generate_threat_report(self, url: str, prediction: int, probability: float, features: dict) -> str:
        cache_key = self._cache_key(prediction, features)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logging.info("Threat report served from cache.")
            return cached

        if not self.active:
            return self._fallback_report(prediction, probability, features)

        try:
            verdict = "PHISHING / SUSPICIOUS" if prediction == 0 else "LEGITIMATE / SAFE"
            prompt = (
                "You are a Level-2 Security Threat Analyst (MLOps Phishing AI). Analyse the URL "
                "and produce a concise, professional threat report.\n\n"
                f"URL: {url}\n"
                f"ML classifier verdict: {verdict}\n"
                f"ML phishing probability: {probability * 100:.2f}%\n"
                f"Structural features (1=safe, 0=suspicious, -1=phishing): {features}\n\n"
                "Assess typosquatting/impersonation risk and explain which structural signals "
                "drove the verdict. Be specific; no filler."
            )
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ThreatReport,
                ),
            )
            report: ThreatReport = response.parsed
            markdown = report.to_markdown(probability)
            self._cache_put(cache_key, markdown)
            return markdown
        except Exception as e:
            logging.error(f"Gemini threat report failed ({e}); using fallback.")
            return self._fallback_report(prediction, probability, features)

    @staticmethod
    def _fallback_report(prediction: int, probability: float, features: dict) -> str:
        """Deterministic report built from features when Gemini is unavailable."""
        verdict = "PHISHING / SUSPICIOUS" if prediction == 0 else "LEGITIMATE / SAFE"
        flagged = [k for k, v in features.items() if v in (-1, 0)]
        report = ThreatReport(
            risk_level="HIGH" if prediction == 0 else "SAFE",
            summary=(
                f"This URL was classified as {verdict} by the MLOps phishing-detection model "
                f"with a {probability * 100:.1f}% phishing probability. "
                "This is an offline heuristic summary (set GEMINI_API_KEY for live AI forensics)."
            ),
            structural_signals=flagged or ["No suspicious structural signals detected."],
            typosquatting_risk=(
                "Potential brand-impersonation indicators present."
                if prediction == 0
                else "No strong impersonation indicators."
            ),
            recommendations=(
                [
                    "Do not enter credentials or personal data on this site.",
                    "Verify the domain against the official source.",
                    "Report the URL to your security team.",
                ]
                if prediction == 0
                else ["No action required, but stay vigilant for look-alike domains."]
            ),
        )
        return report.to_markdown(probability)

    # ------------------------------------------------------------------ #
    # Chatbot
    # ------------------------------------------------------------------ #
    def ask_chatbot(self, user_message: str, chat_history: list = None) -> str:
        if not self.active:
            return (
                "I'm in offline mode (no `GEMINI_API_KEY` configured). Add a Gemini key to "
                "`.env` to chat with the live MLOps cybersecurity assistant."
            )
        try:
            context = (
                "You are 'SecBot', the MLOps (Phishing) cybersecurity assistant. You explain "
                "security terms, phishing patterns, web-security tips, and the MLOps pipeline "
                "behind this app. Be informative, professional, and conversational."
            )
            convo = context + "\n\n"
            for msg in chat_history or []:
                role = "User" if msg.get("role") == "user" else "Assistant"
                convo += f"{role}: {msg.get('content', '')}\n"
            convo += f"User: {user_message}\nAssistant:"

            response = self.client.models.generate_content(model=MODEL_NAME, contents=convo)
            return response.text
        except Exception as e:
            logging.error(f"Gemini chatbot error: {e}")
            return "Sorry, I couldn't reach the AI assistant right now. Please try again shortly."
