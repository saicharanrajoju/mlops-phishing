"""
Agentic threat-hunter.

Unlike `GeminiAnalyst` (which writes a one-shot caption over the ML output), this
is a real *agent*: Gemini is given a set of tools and autonomously decides which
to call, in what order, reasons over the results, and then produces a verdict.

Built on the `google-genai` SDK's automatic function calling — the SDK executes
the Python tool functions and loops until the model is done, capturing the full
tool-call trace so the UI can show *how* the agent reached its conclusion.
"""

import os
import socket
import urllib.parse

from dotenv import load_dotenv

from ai_analyst.feature_extractor import URLFeatureExtractor
from phishsentinel.logging.logger import logging

load_dotenv()

AGENT_MODEL = os.getenv("GEMINI_AGENT_MODEL", "gemini-2.5-flash")
MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "6"))

SYSTEM_INSTRUCTION = (
    "You are the MLOps (Phishing) threat-hunting agent, an autonomous Level-2 analyst. "
    "Investigate the given URL for phishing/malicious intent. ALWAYS gather evidence "
    "with your tools before concluding: inspect the URL structure, run the ML "
    "classifier, check DNS, and review prior scan history. Then write a concise, "
    "professional threat report in markdown with: a clear verdict and risk level, the "
    "specific signals that drove it, and actionable recommendations. Cite the tool "
    "findings you relied on. Do not invent data you did not retrieve."
)


class ThreatHuntingAgent:
    def __init__(self, classify_fn=None, history_fn=None):
        """
        classify_fn(url) -> dict : runs the live ML model (injected by the app).
        history_fn(url)  -> list : returns prior scans for the URL (injected by the app).
        """
        self._classify_fn = classify_fn
        self._history_fn = history_fn
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.active = False

        if self.api_key and "your-" not in self.api_key:
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
                self._genai_types = __import__("google.genai.types", fromlist=["types"])
                self.active = True
                logging.info("ThreatHuntingAgent: Gemini agent online.")
            except Exception as e:
                logging.error(f"ThreatHuntingAgent init failed: {e}")
        else:
            logging.warning("ThreatHuntingAgent: no GEMINI_API_KEY; agent runs in offline mode.")

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    def investigate(self, url: str) -> dict:
        if not self.active:
            return self._offline_investigation(url)

        types = self._genai_types

        # --- Tools the agent may call (clean signatures + docstrings = the schema) ---
        def analyze_url_structure(url: str) -> dict:
            """Extract the 30 structural phishing features from a URL.

            Values are 1 (safe), 0 (suspicious) or -1 (phishing-like).
            """
            return URLFeatureExtractor(url).get_features_dict()

        def classify_url(url: str) -> dict:
            """Run the trained ML model on the URL.

            Returns the prediction (0=phishing, 1=safe), the phishing probability,
            and a confidence label.
            """
            if not self._classify_fn:
                return {"error": "classifier unavailable"}
            return self._classify_fn(url)

        def lookup_domain_dns(url: str) -> dict:
            """Check whether the URL's domain resolves via DNS. Unresolvable domains are suspicious."""
            host = urllib.parse.urlparse(url if "://" in url else "http://" + url).hostname or ""
            try:
                ip = socket.gethostbyname(host)
                return {"domain": host, "resolves": True, "ip": ip}
            except Exception:
                return {"domain": host, "resolves": False, "ip": None}

        def check_scan_history(url: str) -> dict:
            """Look up whether this URL has been scanned before and what the past verdicts were."""
            if not self._history_fn:
                return {"previous_scans": 0}
            rows = self._history_fn(url)
            verdicts = ["phishing" if r.get("prediction") == 0 else "safe" for r in rows]
            return {"previous_scans": len(rows), "past_verdicts": verdicts[:5]}

        tools = [analyze_url_structure, classify_url, lookup_domain_dns, check_scan_history]

        try:
            config = types.GenerateContentConfig(
                tools=tools,
                system_instruction=SYSTEM_INSTRUCTION,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=MAX_TOOL_CALLS),
            )
            response = self.client.models.generate_content(
                model=AGENT_MODEL,
                contents=f"Investigate this URL and deliver a threat assessment: {url}",
                config=config,
            )
            return {
                "agentic": True,
                "report": response.text,
                "tools_used": self._tools_used(response),
            }
        except Exception as e:
            logging.error(f"Agentic investigation failed ({e}); falling back.")
            return self._offline_investigation(url, error=str(e))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tools_used(response) -> list:
        """Extract the ordered list of tools the agent actually invoked."""
        used = []
        for content in getattr(response, "automatic_function_calling_history", None) or []:
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    used.append(fc.name)
        return used

    def _offline_investigation(self, url: str, error: str = "") -> dict:
        """Deterministic result when the agent can't run live (no key / API error)."""
        result = self._classify_fn(url) if self._classify_fn else {}
        label = result.get("label", "unknown")
        prob = result.get("phishing_probability", 0.0)
        note = f"\n\n> Agent error: `{error}`" if error else ""
        report = (
            "### 🧠 Agentic Investigation (offline)\n"
            f"**ML verdict**: `{label.upper()}` (phishing probability `{prob * 100:.1f}%`).\n\n"
            "Live multi-step agent reasoning requires a `GEMINI_API_KEY`. With a key, the agent "
            "autonomously inspects URL structure, runs the classifier, checks DNS, and reviews "
            "scan history before concluding." + note
        )
        return {"agentic": False, "report": report, "tools_used": [], "classification": result}
