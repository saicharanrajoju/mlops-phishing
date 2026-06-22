import json
import os
import sqlite3
import sys

from dotenv import load_dotenv

from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


class DatabaseClient:
    def __init__(self):
        self.use_supabase = False
        self.supabase_client = None
        self.sqlite_db_path = os.path.join(os.getcwd(), "phishsentinel.db")

        # Try to connect to Supabase if credentials exist
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                from supabase import create_client

                self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                self.use_supabase = True
                logging.info("Successfully connected to Supabase Database.")
                print("Database: Connected to Supabase Cloud Database.")
            except Exception as e:
                logging.warning(f"Could not connect to Supabase ({e}). Falling back to SQLite.")
                print("Database: Supabase connection failed. Falling back to local SQLite.")

        if not self.use_supabase:
            logging.info(f"Using local SQLite database at {self.sqlite_db_path}")
            print(f"Database: Running local-first using SQLite ({self.sqlite_db_path}).")
            self._init_sqlite_tables()

    def _init_sqlite_tables(self):
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()

            # Create scan_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    features TEXT NOT NULL, -- JSON string of 30 features
                    ai_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create pipeline_runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    accuracy REAL,
                    f1_score REAL,
                    model_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create feedback table — human-in-the-loop corrections for active learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    predicted_label INTEGER,
                    correct_label INTEGER NOT NULL, -- 0 = phishing, 1 = safe
                    features TEXT,                  -- JSON, so corrections can seed retraining
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
            logging.info("SQLite tables verified/created successfully.")
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def log_scan(self, url: str, prediction: int, probability: float, features: dict, ai_analysis: str = ""):
        """Logs a URL threat scan to the database (Supabase or SQLite)"""
        try:
            features_json = json.dumps(features)

            if self.use_supabase:
                data = {
                    "url": url,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "features": features_json,
                    "ai_analysis": ai_analysis,
                }
                # Supabase table insertion
                response = self.supabase_client.table("scan_history").insert(data).execute()
                logging.info(f"Logged scan to Supabase: {response.data}")
            else:
                conn = sqlite3.connect(self.sqlite_db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO scan_history (url, prediction, probability, features, ai_analysis) VALUES (?, ?, ?, ?, ?)",
                    (url, int(prediction), float(probability), features_json, ai_analysis),
                )
                conn.commit()
                conn.close()
                logging.info(f"Logged scan to SQLite for URL: {url}")
        except Exception as e:
            logging.error(f"Error logging scan: {e}")
            raise PhishSentinelException(e, sys) from e

    def log_feedback(self, url: str, predicted_label: int, correct_label: int, features: dict = None):
        """Record a human correction (active-learning signal) to the database."""
        try:
            features_json = json.dumps(features or {})
            if self.use_supabase:
                self.supabase_client.table("feedback").insert(
                    {
                        "url": url,
                        "predicted_label": int(predicted_label),
                        "correct_label": int(correct_label),
                        "features": features_json,
                    }
                ).execute()
            else:
                conn = sqlite3.connect(self.sqlite_db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO feedback (url, predicted_label, correct_label, features) VALUES (?, ?, ?, ?)",
                    (url, int(predicted_label), int(correct_label), features_json),
                )
                conn.commit()
                conn.close()
            logging.info(f"Logged feedback for URL: {url} (correct_label={correct_label})")
        except Exception as e:
            logging.error(f"Error logging feedback: {e}")
            raise PhishSentinelException(e, sys) from e

    def get_feedback(self, limit: int = 100):
        """Retrieve human corrections (for retraining / active learning)."""
        try:
            if self.use_supabase:
                resp = (
                    self.supabase_client.table("feedback")
                    .select("*")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                return resp.data
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logging.error(f"Error retrieving feedback: {e}")
            raise PhishSentinelException(e, sys) from e

    def log_pipeline_run(self, run_id: str, accuracy: float, f1_score: float, model_path: str):
        """Logs a model training run to the database (Supabase or SQLite)"""
        try:
            if self.use_supabase:
                data = {
                    "run_id": run_id,
                    "accuracy": float(accuracy),
                    "f1_score": float(f1_score),
                    "model_path": model_path,
                }
                response = self.supabase_client.table("pipeline_runs").insert(data).execute()
                logging.info(f"Logged pipeline run to Supabase: {response.data}")
            else:
                conn = sqlite3.connect(self.sqlite_db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO pipeline_runs (run_id, accuracy, f1_score, model_path) VALUES (?, ?, ?, ?)",
                    (run_id, float(accuracy), float(f1_score), model_path),
                )
                conn.commit()
                conn.close()
                logging.info(f"Logged pipeline run to SQLite: {run_id}")
        except Exception as e:
            logging.error(f"Error logging pipeline run: {e}")
            raise PhishSentinelException(e, sys) from e

    def get_scan_history(self, limit: int = 50):
        """Retrieves past scans history"""
        try:
            if self.use_supabase:
                response = (
                    self.supabase_client.table("scan_history")
                    .select("*")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                # Parse features JSON inside result
                records = response.data
                for r in records:
                    if r.get("features"):
                        try:
                            r["features"] = json.loads(r["features"])
                        except Exception:
                            pass
                return records
            else:
                conn = sqlite3.connect(self.sqlite_db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM scan_history ORDER BY created_at DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
                records = []
                for row in rows:
                    rec = dict(row)
                    try:
                        rec["features"] = json.loads(rec["features"])
                    except Exception:
                        pass
                    records.append(rec)
                conn.close()
                return records
        except Exception as e:
            logging.error(f"Error retrieving scan history: {e}")
            raise PhishSentinelException(e, sys) from e

    def get_pipeline_runs(self, limit: int = 10):
        """Retrieves past training pipeline runs"""
        try:
            if self.use_supabase:
                response = (
                    self.supabase_client.table("pipeline_runs")
                    .select("*")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                return response.data
            else:
                conn = sqlite3.connect(self.sqlite_db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
                records = [dict(row) for row in rows]
                conn.close()
                return records
        except Exception as e:
            logging.error(f"Error retrieving pipeline runs: {e}")
            raise PhishSentinelException(e, sys) from e
