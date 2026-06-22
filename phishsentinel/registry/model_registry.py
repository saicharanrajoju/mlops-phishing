"""
MLflow Model Registry helpers for MLOps (Phishing).

The registry is the single source of truth for "which model is live". Training
registers a new *version*; a quality gate promotes it to the ``@production``
alias only if it beats the current champion. Serving loads ``models:/<name>@production``.

Note: the MLflow Model Registry requires a database-backed tracking store
(sqlite/postgres), not the bare ``./mlruns`` file store. We default to the local
``mlflow.db`` sqlite file, overridable via ``MLFLOW_TRACKING_URI``.
"""

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from phishsentinel.logging.logger import logging

REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL", "phishsentinel-phishing-detector")
PRODUCTION_ALIAS = "production"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "phishsentinel-phishing"


def setup_mlflow() -> str:
    """Point MLflow at a DB-backed store (so the registry works) and set the experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    return tracking_uri


def register_model(model, input_example=None) -> str | None:
    """
    Log a fitted sklearn pipeline as an MLflow model and register a new version.
    Must be called inside an active MLflow run. Returns the new version string,
    or None if registration failed (training still succeeds via the local fallback).
    """
    try:
        signature = None
        if input_example is not None:
            from mlflow.models.signature import infer_signature

            signature = infer_signature(input_example, model.predict(input_example))

        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )
        version = getattr(info, "registered_model_version", None)
        if version is None:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            version = max(int(v.version) for v in versions)
        logging.info(f"Registered '{REGISTERED_MODEL_NAME}' version {version}")
        return str(version)
    except Exception as e:
        logging.error(f"Model registry logging failed: {e}")
        return None


def promote_if_better(version: str, new_f1: float) -> bool:
    """
    Champion/challenger gate: tag the new version with its test F1, then promote
    it to ``@production`` only if it is at least as good as the current champion
    (or if there is no champion yet).
    """
    client = MlflowClient()
    try:
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "test_f1", f"{new_f1:.6f}")
    except Exception as e:
        logging.warning(f"Could not tag model version: {e}")

    current_f1 = -1.0
    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        current_f1 = float(current.tags.get("test_f1", "0") or 0.0)
    except Exception:
        current = None  # no production model yet

    if current_f1 < 0 or new_f1 >= current_f1:
        try:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version)
            logging.info(f"Promoted version {version} to @{PRODUCTION_ALIAS} (F1={new_f1:.4f})")
            return True
        except Exception as e:
            logging.error(f"Promotion failed: {e}")
            return False

    logging.info(f"Challenger F1={new_f1:.4f} did not beat champion F1={current_f1:.4f}; " "production unchanged.")
    return False


def load_production_model():
    """
    Load the current ``@production`` sklearn pipeline from the registry.
    Returns the native sklearn model (with ``predict_proba``) or None if the
    registry is unavailable / empty — callers should fall back to a local model.
    """
    try:
        setup_mlflow()
        uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
        model = mlflow.sklearn.load_model(uri)
        logging.info(f"Loaded production model from registry: {uri}")
        return model
    except Exception as e:
        logging.warning(f"Could not load model from registry ({e}); using local fallback.")
        return None


def get_production_info() -> dict | None:
    """Lightweight metadata about the current champion, for dashboards/health checks."""
    try:
        setup_mlflow()
        client = MlflowClient()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        return {
            "name": REGISTERED_MODEL_NAME,
            "version": mv.version,
            "test_f1": mv.tags.get("test_f1"),
            "run_id": mv.run_id,
        }
    except Exception:
        return None
