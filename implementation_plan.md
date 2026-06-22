# Autonomous Threat-Hunting MLOps Pipeline (2026 Plan)

This project implements a production-style, modular MLOps pipeline for **phishing / network-security detection**, built around **Google GenAI SDK (Gemini)**, **Supabase**, and **MLflow**.

---

## 1. System Architecture

```
                    ┌────────────────────────┐
                    │  Modern Web Frontend   │ (Glassmorphic Dashboard)
                    └───────────┬────────────┘
                                │
                        (FastAPI App)
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
   ┌──────────────────┐                  ┌──────────────────┐
   │    ML Engine     │                  │  AI L2 Analyst   │
   │  (XGBoost model) │                  │ (Google GenAI)   │
   └─────────┬────────┘                  └─────────┬────────┘
             │                                     │
   ┌─────────▼────────┐                            │
   │  MLflow Metrics  │                            │
   └──────────────────┘                            │
             │                                     │
             ▼                                     ▼
     ┌──────────────┐                       ┌──────────────┐
     │   Database   │◄──────────────────────┤   Database   │
     │  (Supabase)  │  (Save scan history)  │  (Supabase)  │
     └──────────────┘                       └──────────────┘
```

The system coordinates traditional machine learning classification with LLM reasoning:
1. **Raw URL Input**: The user inputs a URL (e.g. `http://bank-verification.com/login`).
2. **Feature Extraction**: Python utilities parse the URL into a 30-feature vector expected by the model.
3. **ML Prediction**: A trained tabular model (XGBoost/Random Forest) classifies the vector.
4. **Google GenAI L2 Report**: The Google GenAI SDK calls Gemini with the domain, extracted features, and ML prediction to write a cybersecurity breakdown.
5. **Log Scan to Database**: Saves results (URL, probability, features, AI explanation) to Supabase (with a zero-configuration SQLite local database fallback).

---

## 2. Technical Stack

*   **Google GenAI SDK**: `google-genai` (utilizing Gemini models) for explainable security diagnostics and the cybersecurity assistant chatbot.
*   **Database**: Supabase / PostgreSQL for scan logging, metadata tracking, and training runs (with a zero-configuration SQLite local database fallback).
*   **ML & Preprocessing**: `xgboost`, `scikit-learn`, `pandas`, `numpy`, and `dill` for pipeline components.
*   **MLOps Platform**: MLflow for experiment tracking and a **Model Registry** with champion/challenger promotion (`@production` alias).
*   **Monitoring**: KS-test + Population Stability Index (PSI) drift detection that signals retraining.
*   **Backend Server**: FastAPI & Uvicorn serving static client files and API endpoints.

---

## 3. Directory Layout & Core Components

```
ML-Ops/
├── data/                       # Local raw phishing CSV datasets
├── phishsentinel/            # Core package
│   ├── constant/               # Directory paths, schemas, hyperparameters
│   ├── database/               # Supabase & SQLite connections
│   ├── entity/                 # Input/output config and artifact dataclasses
│   ├── logging/                # Logger configuration
│   ├── exception/              # Customized exceptions
│   ├── components/             # Ingestion, Validation, Transformation, Model Trainer
│   ├── pipeline/               # Orchestrated training and retraining workflows
│   ├── registry/               # MLflow registry: register / promote / load champion
│   ├── monitoring/             # KS + PSI data-drift monitor
│   └── utils/                  # Common file/pickle helpers
├── ai_analyst/                 # AI & LLM Components
│   ├── feature_extractor.py    # URL parser into 30 features
│   └── gemini_analyst.py       # Google GenAI LLM interface
├── templates/                  # Frontend html assets
├── app.py                      # FastAPI server
├── requirements.txt            # Dependency configuration
└── .env                        # Credentials (GEMINI_API_KEY, SUPABASE_URL)
```

---

## 4. Verification Plan

### Automated Tests
1. Run `python main.py` to trigger the pipeline locally (Ingestion -> Validation -> Transformation -> Model Trainer). Verify that MLflow registers metrics and saved artifacts (`model.pkl`, `preprocessor.pkl`) exist.
2. Run testing suites validating the feature extraction rules.

### Manual Verification
1. Boot the API and UI: `python app.py`
2. Access `http://127.0.0.1:8000` and scan test URLs (e.g. `https://google.com` vs known mock phishing links).
3. Query the cybersecurity chatbot to verify Gemini's responsiveness.
