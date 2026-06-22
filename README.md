# 🛡️ MLOps (Phishing)

**An end-to-end MLOps pipeline for phishing / network-security detection, with a Gemini-powered L2 threat analyst.**

MLOps (Phishing) takes a raw URL, extracts 30 structural features, classifies it with a
registry-managed ML model, and has **Google Gemini** write a human-readable security
report explaining the verdict — all behind a FastAPI app with a glassmorphic dashboard.

![CI](https://github.com/saicharanrajoju/mlops-phishing/actions/workflows/ci.yml/badge.svg)

---

## Why this project

Most "ML projects" stop at a trained `.pkl`. This project is built as a **closed MLOps
loop** — every stage of the lifecycle is a real, working component:

```
            ┌──────────────── TRAINING PLANE (offline) ───────────────┐
  data ──►  ingest ─► validate ─► transform ─► train (GridSearch)     │
   ▲        └────────────────────────────┬────────────────────────────┘
   │                                      │ register version
   │                                      ▼
   │                         ┌──────────────────────────┐
   │                         │   MLflow Model Registry   │  champion/challenger:
   │                         │   phishsentinel-detector  │  promote to @production
   │                         └─────────────┬────────────┘  only if F1 improves
   │                                        │ load @production
   │            ┌──────────── SERVING PLANE (online) ─────────────┐
   │            │  FastAPI  ─►  model (loaded once at startup)     │
   │            │     │                                            │
   │            │     ▼                                            │
   │            │  Gemini L2 analyst (structured + cached + fallback)
   │            └──────────────────────┬─────────────────────────┘
   │                                   │ scans logged
   │                                   ▼
   └──── retrain  ◄──── drift detected (KS + PSI) ◄──── MONITORING
```

The arrow that makes it "end-to-end": **monitoring → drift → retrain.**

---

## Tech stack

| Concern | Tool |
|---|---|
| Modeling | scikit-learn, XGBoost (GridSearch over 5 classifiers) |
| Experiment tracking & **registry** | MLflow (DB-backed, alias-based promotion) |
| Serving | FastAPI + Uvicorn |
| Generative AI | Google Gemini (`google-genai`) with Pydantic structured output |
| **Agentic AI** | Gemini **automatic function calling** — the agent calls tools & reasons |
| Active learning | human-in-the-loop feedback loop for retraining |
| Data validation | schema checks + Kolmogorov–Smirnov drift |
| Monitoring | KS-test + Population Stability Index (PSI) |
| Persistence | Supabase / PostgreSQL → local SQLite fallback |
| Quality | pytest, Ruff, pre-commit |
| Packaging / CI | Docker + docker-compose, GitHub Actions |

---

## Project layout

```
ML-Ops/
├── phishsentinel/              # Core package
│   ├── components/             # ingestion → validation → transformation → trainer
│   ├── pipeline/               # training DAG
│   ├── registry/               # MLflow registry: register / promote / load champion
│   ├── monitoring/             # KS + PSI drift monitor (closes the loop)
│   ├── entity/ constant/ utils/ database/ exception/ logging/
├── ai_analyst/
│   ├── feature_extractor.py    # URL → 30 features
│   ├── gemini_analyst.py       # Gemini client: structured output + cache + fallback
│   └── schemas.py              # Pydantic ThreatReport
├── tests/                      # pytest: extractor, analyst, drift, API smoke
├── templates/index.html        # dashboard
├── app.py                      # FastAPI serving layer (serving only)
├── main.py                     # training entrypoint
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Quickstart (local)

```bash
# 1. Install
pip install -r requirements.txt          # add -dev for tests/lint: requirements-dev.txt

# 2. (optional) configure secrets
cp .env.example .env                      # set GEMINI_API_KEY for live AI reports

# 3. Train — runs the pipeline and registers + promotes the model to @production
python main.py

# 4. Serve
python app.py                             # http://127.0.0.1:8000

# 5. (optional) MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then scan a URL in the dashboard, or:

```bash
curl -X POST http://127.0.0.1:8000/api/scan -H "Content-Type: application/json" \
  -d '{"url":"http://secure-paypal-login.com@1.2.3.4/verify"}'
```

## Quickstart (Docker)

Runs a real two-service stack — an MLflow registry server + the app pointing at it:

```bash
cp .env.example .env
docker compose up --build
# app:    http://localhost:8000   (click "Trigger Training Pipeline")
# mlflow: http://localhost:5000
```

---

## How the key pieces work

### Model Registry (the source of truth)
Training composes the fitted preprocessor + best classifier into one native sklearn
`Pipeline`, logs it to MLflow, and registers a new **version**. A champion/challenger
gate (`promote_if_better`) only moves the `@production` alias if the new test-F1 beats
the current champion. Serving loads `models:/phishsentinel-phishing-detector@production`
**once at startup** — with a local-pickle fallback so it still runs offline.

### Gemini L2 analyst (treated as an unreliable dependency)
- **Structured output** — Gemini returns JSON validated against a Pydantic `ThreatReport`.
- **Caching** — identical scans reuse the report (free + instant).
- **Graceful fallback** — missing key / API error / bad JSON → a deterministic templated
  report from the features. *A scan never fails because the LLM hiccupped.*

### Agentic investigation (the AI uses tools, not just text)
`POST /api/investigate` runs a **Gemini agent** (automatic function calling). Instead of
captioning the ML output, the agent **decides which tools to call** and reasons in steps:

| Tool | What it gives the agent |
|---|---|
| `analyze_url_structure` | the 30 structural features |
| `classify_url` | the ML model's verdict + probability + confidence |
| `lookup_domain_dns` | live DNS resolution signal |
| `check_scan_history` | whether the URL was seen before |

The response includes `tools_used` — the **ordered trace** of what the agent actually
called — surfaced in the dashboard so you can see *how* it reached the verdict.

### Built-in intelligence
- **Uncertainty awareness** — `/api/scan` returns a `confidence` level and a
  `needs_review` flag when a verdict sits near the 0.5 decision boundary.
- **Active-learning loop** — `/api/feedback` records human corrections (👍/👎 in the UI)
  to the `feedback` table, ready to seed the next retraining run.

### Drift monitoring (closes the loop)
```bash
python -m phishsentinel.monitoring.drift_monitor --reference data/phisingData.csv --current new.csv
```
Per-feature **KS-test + PSI**, an overall dataset-drift verdict, and a JSON + HTML report —
the signal that should trigger retraining.

---

## Testing, linting, CI

```bash
pytest -q          # unit + API smoke tests (offline, no network/model needed)
ruff check .       # lint
ruff format .      # format
pre-commit install # run both automatically on commit
```
GitHub Actions runs lint + format-check + tests on every push/PR.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | enables live AI reports/chatbot | _(fallback mode)_ |
| `MLFLOW_TRACKING_URI` | tracking + registry store | `sqlite:///mlflow.db` |
| `SUPABASE_URL` / `SUPABASE_KEY` | cloud scan logging | _(SQLite fallback)_ |
| `CORS_ALLOW_ORIGINS` | allowed API origins | localhost only |

---

## Known limitations & future work

- **Feature extraction is heuristic.** Live URL features are computed on-the-fly and
  approximate some dataset signals (e.g. WHOIS/page-rank), so borderline synthetic URLs
  can score near the threshold. Swapping in real WHOIS/HTML probes would tighten this.
- **Orchestration** is a subprocess + CI cron; Prefect/Dagster/ZenML would be the
  production upgrade.
- **Monitoring** uses KS + PSI; **Evidently AI** is the richer drop-in for dashboards.
- **Registry** runs on SQLite; point `MLFLOW_TRACKING_URI` at Postgres for a team setup.
- **Agent** uses Gemini function calling; next steps: **RAG** over a threat-intel
  knowledge base to ground reports, **Langfuse** for agent tracing/cost, and porting to
  the **Google ADK** framework for multi-agent workflows.
