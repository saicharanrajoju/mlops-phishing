# Architecture — per-stage flow

Detailed diagrams for each stage of the **MLOps (Phishing)** pipeline. GitHub renders
these Mermaid diagrams natively. For the high-level loop, see the [README](../README.md).

---

## 1. Data Ingestion

```mermaid
flowchart TD
    CFG["Data Ingestion Config<br/>dirs · paths · split ratio · collection"] --> INIT["Initiate ingestion"]
    DB[("Dataset (CSV / Mongo)")] --> FS["Export to Feature Store<br/>raw.csv"]
    INIT --> FS
    FS --> SPLIT["Train / Test split"]
    SPLIT --> TR["train.csv"]
    SPLIT --> TE["test.csv"]
    TR --> ART["Data Ingestion Artifact"]
    TE --> ART
```

## 2. Data Validation

```mermaid
flowchart TD
    A["Ingestion Artifact<br/>train.csv · test.csv"] --> READ["Read data"]
    SCHEMA["schema.yaml"] --> COLS["Validate # of columns"]
    READ --> COLS
    COLS --> NUM["Numerical columns exist?"]
    NUM --> DRIFT["Detect dataset drift<br/>KS-test per column"]
    DRIFT --> STATUS{"Validation status"}
    STATUS -->|True| OK["valid/ train.csv · test.csv"]
    STATUS -->|False| ERR["Validation error"]
    DRIFT --> REPORT["drift report.yaml"]
    OK --> ART["Data Validation Artifact"]
    REPORT --> ART
```

## 3. Data Transformation

```mermaid
flowchart TD
    A["Validated train / test"] --> READ["Read data"]
    READ --> SPLITX["Split features / target<br/>map -1 → 0 (phishing), 1 (safe)"]
    SPLITX --> PIPE["Preprocessor pipeline<br/>KNNImputer → RobustScaler"]
    PIPE --> FIT["fit_transform train · transform test"]
    FIT --> NPY["train.npy · test.npy"]
    PIPE --> PKL["preprocessor.pkl"]
    NPY --> ART["Data Transformation Artifact"]
    PKL --> ART
```

## 4. Model Trainer + Registry

```mermaid
flowchart TD
    A["train.npy · test.npy"] --> LOAD["Load arrays → X / y"]
    LOAD --> GS["GridSearch:<br/>RandomForest · DecisionTree · GradientBoosting · AdaBoost · XGBoost"]
    GS --> BEST["Select best by test-F1"]
    BEST --> GATE{"F1 ≥ threshold?"}
    GATE -->|No| FAIL["Raise: no acceptable model"]
    GATE -->|Yes| WRAP["Pipeline(preprocessor + classifier)"]
    WRAP --> MLF["Log run + params + metrics → MLflow"]
    MLF --> REGV["Register new model version"]
    REGV --> PROMO{"Beats current champion?"}
    PROMO -->|Yes| PROD["Set @production alias"]
    PROMO -->|No| KEEP["Keep current champion"]
```

## 5. Serving + Agent

```mermaid
flowchart TD
    START["App startup"] --> LOADM["Load models:/...@production<br/>(fallback: local pickle)"]

    URL["POST /api/scan"] --> FEAT["Extract 30 features"]
    FEAT --> PRED["model.predict_proba"]
    PRED --> CONF["confidence + needs_review"]
    PRED --> AIRPT["Gemini report<br/>structured · cached · fallback"]
    AIRPT --> RESP["JSON response + log scan"]

    INV["POST /api/investigate"] --> AGENT["🧠 Gemini agent"]
    AGENT -->|tool| T1["analyze_url_structure"]
    AGENT -->|tool| T2["classify_url"]
    AGENT -->|tool| T3["lookup_domain_dns"]
    AGENT -->|tool| T4["check_scan_history"]
    AGENT --> VERD["verdict + report + tools_used"]
```

## 6. Drift Monitoring (closes the loop)

```mermaid
flowchart LR
    REF["Reference dataset"] --> CMP["Per-feature compare"]
    CUR["Current dataset"] --> CMP
    CMP --> KS["KS-test p-value"]
    CMP --> PSI["PSI"]
    KS --> FLAG{"feature drifted?"}
    PSI --> FLAG
    FLAG --> RPT["JSON + HTML drift report"]
    RPT --> DEC{"dataset drift?"}
    DEC -->|Yes| RETRAIN["Trigger retraining"]
    DEC -->|No| NOOP["No action"]
```
