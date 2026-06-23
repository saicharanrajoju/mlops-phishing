# Screenshots

Drop your captured PNGs here with these exact filenames so the README renders them:

| Filename | What to capture | How |
|---|---|---|
| `dashboard.png` | The dashboard after scanning a URL (gauge + verdict + AI dossier + features) | `python app.py` → open http://127.0.0.1:8000 → scan a URL |
| `agent.png` | An agentic investigation result + the "Agent tools called" chips | click **🧠 INVESTIGATE** on a URL (needs `GEMINI_API_KEY`) |
| `mlflow.png` | MLflow runs / registered model versions | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000 |
| `drift.png` | The drift report | open `monitoring_reports/drift_report.html` after running the drift monitor |

Tip: on Windows use **Win + Shift + S** to snip, then save into this folder.

Once added, they appear automatically in the main [README](../../README.md#-demo).
