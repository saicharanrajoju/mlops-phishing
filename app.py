"""
MLOps (Phishing) — FastAPI serving layer.

This process is *serving only*. It loads the champion model once at startup from
the MLflow Model Registry (falling back to a local pickle), and serves
predictions + the Gemini threat analyst. Training is intentionally NOT run inside
this process — the /api/pipeline/run endpoint launches `main.py` as a separate
OS process so training never competes with the web server for the event loop.
"""

import os
import subprocess
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ai_analyst.feature_extractor import URLFeatureExtractor
from ai_analyst.gemini_analyst import GeminiAnalyst
from ai_analyst.threat_agent import ThreatHuntingAgent
from phishsentinel.constant.training_pipeline import SCHEMA_COLUMNS, TARGET_COLUMN
from phishsentinel.database.supabase_client import DatabaseClient
from phishsentinel.logging.logger import logging
from phishsentinel.registry.model_registry import get_production_info, load_production_model
from phishsentinel.utils.main_utils.utils import load_object

LOCAL_MODEL_PATH = os.path.join("final_model", "model.pkl")
FEATURE_COLUMNS = [c for c in SCHEMA_COLUMNS if c != TARGET_COLUMN]


class ModelHolder:
    """Loads and holds the live model so it is loaded once, not per-request."""

    def __init__(self):
        self.model = None
        self.source = "none"

    def load(self) -> bool:
        """Prefer the registry champion; fall back to a local pickle for offline use."""
        model = load_production_model()
        if model is not None:
            self.model, self.source = model, "registry"
        elif os.path.exists(LOCAL_MODEL_PATH):
            self.model, self.source = load_object(LOCAL_MODEL_PATH), "local_pickle"
        else:
            self.model, self.source = None, "none"
        logging.info(f"Serving model source: {self.source}")
        return self.model is not None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def phishing_probability(self, df: pd.DataFrame) -> float:
        """Return P(phishing). Phishing is class 0; locate its column from classes_."""
        proba = self.model.predict_proba(df)[0]
        classes = list(getattr(self.model, "classes_", [0, 1]))
        try:
            idx = classes.index(0)  # 0 == 0.0, works for int or float labels
        except ValueError:
            idx = 0
        return float(proba[idx])


model_holder = ModelHolder()
db_client = DatabaseClient()
ai_analyst = GeminiAnalyst()


def _confidence(prob: float) -> dict:
    """Turn a phishing probability into a confidence label + human-review flag."""
    margin = abs(prob - 0.5)
    level = "LOW" if margin < 0.10 else "MEDIUM" if margin < 0.25 else "HIGH"
    return {"confidence": level, "needs_review": margin < 0.10}


def _classify_url(url: str) -> dict:
    """Live ML classification, shared by /api/scan and the agent's `classify_url` tool."""
    if not model_holder.ready:
        return {"error": "model not loaded"}
    ext = URLFeatureExtractor(url)
    df = pd.DataFrame([ext.get_features_list()], columns=FEATURE_COLUMNS)
    prob = model_holder.phishing_probability(df)
    pred = 0 if prob > 0.50 else 1
    return {
        "prediction": pred,
        "label": "phishing" if pred == 0 else "safe",
        "phishing_probability": round(prob, 4),
        **_confidence(prob),
    }


def _url_history(url: str) -> list:
    """Prior scans for a URL, used by the agent's `check_scan_history` tool."""
    scans = db_client.get_scan_history(limit=100)
    return [s for s in scans if s.get("url") == url]


# Agentic threat-hunter: Gemini autonomously calls the tools above to reach a verdict.
threat_agent = ThreatHuntingAgent(classify_fn=_classify_url, history_fn=_url_history)

# Handle to a detached training subprocess (if one is running) + last-seen state
_training_proc: subprocess.Popen | None = None
_was_training = False


def _training_running() -> bool:
    return _training_proc is not None and _training_proc.poll() is None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_holder.load()
    yield


app = FastAPI(title="MLOps (Phishing) — Threat-Hunting Portal", lifespan=lifespan)

# CORS: explicit origins by default (a security product shouldn't ship `*`).
_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")


class ScanRequest(BaseModel):
    url: str


class InvestigateRequest(BaseModel):
    url: str


class FeedbackRequest(BaseModel):
    url: str
    correct_label: int  # 0 = phishing, 1 = safe
    predicted_label: int | None = None


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = []


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_exists": model_holder.ready},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": model_holder.ready,
        "model_source": model_holder.source,
        "production": get_production_info(),
        "training_running": _training_running(),
    }


@app.post("/api/scan")
async def scan_url(request: ScanRequest):
    try:
        url = request.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="URL cannot be empty")

        logging.info(f"Received scan request for URL: {url}")

        if not model_holder.ready:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "detail": "No model is loaded yet. Trigger pipeline training first.",
                },
            )

        # 1. Extract 30 structural features from the URL
        extractor = URLFeatureExtractor(url)
        features_dict = extractor.get_features_dict()
        features_list = extractor.get_features_list()

        # 2. ML prediction (model loaded once at startup)
        df = pd.DataFrame([features_list], columns=FEATURE_COLUMNS)
        phishing_prob = model_holder.phishing_probability(df)
        prediction = 0 if phishing_prob > 0.50 else 1  # 0 = phishing, 1 = safe
        conf = _confidence(phishing_prob)  # uncertainty awareness

        # 3. Gemini L2 analyst report (never hard-fails the scan)
        ai_report = ai_analyst.generate_threat_report(
            url=url, prediction=prediction, probability=phishing_prob, features=features_dict
        )

        # 4. Persist the scan
        db_client.log_scan(
            url=url,
            prediction=prediction,
            probability=phishing_prob,
            features=features_dict,
            ai_analysis=ai_report,
        )

        return {
            "success": True,
            "url": url,
            "prediction": prediction,
            "probability": phishing_prob,
            "features": features_dict,
            "ai_analysis": ai_report,
            **conf,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error scanning URL: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/investigate")
async def investigate_url(request: InvestigateRequest):
    """Agentic investigation: Gemini autonomously calls tools and reasons to a verdict."""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    try:
        result = threat_agent.investigate(url)
        return {"success": True, "url": url, **result}
    except Exception as e:
        logging.error(f"Error in agentic investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Human-in-the-loop correction, stored for active learning / retraining."""
    if request.correct_label not in (0, 1):
        raise HTTPException(status_code=400, detail="correct_label must be 0 (phishing) or 1 (safe)")
    try:
        features = URLFeatureExtractor(request.url).get_features_dict()
        db_client.log_feedback(
            url=request.url,
            predicted_label=request.predicted_label if request.predicted_label is not None else -1,
            correct_label=request.correct_label,
            features=features,
        )
        return {"success": True, "message": "Feedback recorded — thank you, this improves future models."}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat")
async def chat_with_assistant(request: ChatRequest):
    try:
        reply = ai_analyst.ask_chatbot(request.message, request.history)
        return {"reply": reply}
    except Exception as e:
        logging.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/history")
async def get_history():
    try:
        scans = db_client.get_scan_history(limit=20)
        return {"scans": scans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/pipeline/run")
async def trigger_pipeline():
    """Launch training in a SEPARATE process so it never blocks the web server."""
    global _training_proc
    if _training_running():
        return {"message": "Pipeline training is already running."}
    _training_proc = subprocess.Popen([sys.executable, "main.py"], cwd=os.getcwd())
    logging.info(f"Launched training subprocess pid={_training_proc.pid}")
    return {"message": "Pipeline training triggered in a separate process."}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    global _was_training
    running = _training_running()
    # When a run transitions from running -> finished, hot-reload the new champion.
    if _was_training and not running:
        logging.info("Training finished; reloading champion model from registry.")
        model_holder.load()
    _was_training = running

    try:
        runs = db_client.get_pipeline_runs(limit=10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "is_training": running,
        "last_run_status": "Training in progress..." if running else "Idle",
        "error_message": "",
        "model_source": model_holder.source,
        "runs": runs,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("app:app", host=host, port=port, reload=True)
