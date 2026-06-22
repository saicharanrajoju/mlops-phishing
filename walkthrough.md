# MLOps (Phishing) — Build & Verification Log

A record of what was built and the checks that were actually run. See
[README.md](README.md) for full setup and architecture.

---

## What was built

1. **Modular training pipeline** (`phishsentinel/`): data ingestion → schema/KS
   validation → KNNImputer + RobustScaler transformation → GridSearch over Random
   Forest, Decision Tree, Gradient Boosting, AdaBoost and XGBoost.
2. **MLflow Model Registry** (`phishsentinel/registry/`): training composes the
   preprocessor + best classifier into one sklearn `Pipeline`, registers a new
   **version**, and a champion/challenger gate promotes it to the `@production`
   alias only if test-F1 improves. Backed by a DB-backed tracking store (sqlite).
3. **Serving layer** (`app.py`): FastAPI loads the `@production` model **once at
   startup** (local-pickle fallback). Training is launched as a **separate process**,
   never a thread inside the web server. New `/health` endpoint exposes model source
   and champion metadata.
4. **AI L2 analyst** (`ai_analyst/`): Gemini with **Pydantic structured output**,
   an in-memory **cache**, a request **timeout**, and a deterministic **fallback**
   report so scans never fail when the LLM is unavailable.
5. **Drift monitoring** (`phishsentinel/monitoring/`): KS-test + PSI per feature with
   JSON/HTML reports — the signal that closes the retraining loop.
6. **Quality & delivery**: pytest suite, Ruff lint+format, pre-commit, Dockerfile +
   docker-compose (app + MLflow server), and a GitHub Actions CI pipeline.

---

## Verification results

| Check | Result |
|---|---|
| `python main.py` (full pipeline) | Best model **XGBoost**, **Test F1 ≈ 0.968** |
| Model registered | `phishsentinel-phishing-detector` **version 1** |
| `@production` alias promoted | tagged `test_f1=0.968` |
| Serving loads from registry | source = `registry`, native `predict_proba` |
| `/api/scan` (google.com) | prediction = 1 (safe), phishing prob ≈ 0.02 |
| AI fallback report (no key) | deterministic markdown dossier rendered |
| Drift monitor (self-check) | 0/30 features drifted |
| `pytest` | all tests pass (extractor, analyst, drift, API smoke) |
| `ruff check` / `ruff format --check` | clean |

---

## Run it

```bash
pip install -r requirements.txt
python main.py        # train + register + promote
python app.py         # serve at http://127.0.0.1:8000
```

Or the full stack with Docker: `docker compose up --build`.
