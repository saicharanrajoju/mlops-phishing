import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from phishsentinel.constant.training_pipeline import SCHEMA_COLUMNS, TARGET_COLUMN
from phishsentinel.database.supabase_client import DatabaseClient
from phishsentinel.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from phishsentinel.entity.config_entity import ModelTrainerConfig
from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging
from phishsentinel.registry.model_registry import promote_if_better, register_model, setup_mlflow
from phishsentinel.utils.main_utils.utils import load_object, save_object
from phishsentinel.utils.ml_utils.model.estimator import PhishSentinelModel


class ModelTrainer:
    def __init__(
        self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.db_client = DatabaseClient()
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def evaluate_models(self, x_train, y_train, x_test, y_test, models: dict, param: dict) -> dict:
        """Runs custom hyperparameter grid search and returns score dictionary"""
        try:
            report = {}
            for name, model in models.items():
                logging.info(f"Tuning and training model: {name}")

                # Check hyperparams tuning
                params = param.get(name, {})
                best_model = model

                # Basic grid search implementation
                from sklearn.model_selection import GridSearchCV

                gs = GridSearchCV(model, params, cv=3)
                gs.fit(x_train, y_train)

                best_model.set_params(**gs.best_params_)
                best_model.fit(x_train, y_train)

                # Predict
                y_train_pred = best_model.predict(x_train)
                y_test_pred = best_model.predict(x_test)

                # Metrics
                train_f1 = f1_score(y_train, y_train_pred, average="weighted")
                test_f1 = f1_score(y_test, y_test_pred, average="weighted")

                logging.info(f"Model {name} - Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

                report[name] = {
                    "model": best_model,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "best_params": gs.best_params_,
                }
            return report
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating Model Trainer phase...")

            # Load transformed npy files
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            logging.info("Splitting feature and target variables from numpy arrays...")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Models definition
            models = {
                "RandomForest": RandomForestClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            }

            # Hyperparameter grids
            params = {
                "DecisionTree": {
                    "criterion": ["gini", "entropy"],
                },
                "RandomForest": {
                    "n_estimators": [50, 100],
                },
                "GradientBoosting": {"learning_rate": [0.1, 0.05], "n_estimators": [50, 100]},
                "AdaBoost": {"n_estimators": [50, 100]},
                "XGBoost": {"learning_rate": [0.1, 0.2], "n_estimators": [50, 100]},
            }

            # Run grid search evaluation
            model_report = self.evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params
            )

            # Find best model
            best_model_name = max(model_report, key=lambda k: model_report[k]["test_f1"])
            best_model_info = model_report[best_model_name]
            best_model_obj = best_model_info["model"]
            best_f1_score = best_model_info["test_f1"]
            best_params = best_model_info["best_params"]

            logging.info(f"Best model selected: {best_model_name} with Test F1: {best_f1_score:.4f}")

            if best_f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No model met the expected minimum F1-score threshold of {self.model_trainer_config.expected_accuracy}"
                )

            # Load preprocessor
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Package combined model (preprocessor + model)
            network_model = PhishSentinelModel(preprocessor=preprocessor, model=best_model_obj)

            # MLflow tracking + registry (DB-backed store so the Model Registry works)
            setup_mlflow()

            logging.info(f"Starting MLflow Run for model: {best_model_name}")
            with mlflow.start_run() as run:
                # Log params and metrics
                mlflow.log_param("best_model_name", best_model_name)
                for pk, pv in best_params.items():
                    mlflow.log_param(f"param_{pk}", pv)

                # Predictions for full metric tracking
                y_train_pred = best_model_obj.predict(x_train)
                y_test_pred = best_model_obj.predict(x_test)

                # Evaluate training metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                train_prec = precision_score(y_train, y_train_pred, average="weighted")
                train_rec = recall_score(y_train, y_train_pred, average="weighted")
                train_f1 = f1_score(y_train, y_train_pred, average="weighted")

                # Evaluate test metrics
                test_acc = accuracy_score(y_test, y_test_pred)
                test_prec = precision_score(y_test, y_test_pred, average="weighted")
                test_rec = recall_score(y_test, y_test_pred, average="weighted")
                test_f1 = f1_score(y_test, y_test_pred, average="weighted")

                # Log metrics to MLflow
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("train_f1", train_f1)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("test_f1", test_f1)

                # Local fallback artifact: dill-pickled wrapper, used when the registry is unreachable
                model_save_path = self.model_trainer_config.trained_model_file_path
                save_object(file_path=model_save_path, obj=network_model)
                logging.info(f"Saved PhishSentinelModel wrapper to: {model_save_path}")

                # Register a native sklearn pipeline (preprocessor + classifier) to the MLflow
                # Model Registry, then promote it to @production only if it beats the champion.
                full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_model_obj)])
                feature_cols = [c for c in SCHEMA_COLUMNS if c != TARGET_COLUMN]
                input_example = pd.DataFrame([[1] * len(feature_cols)], columns=feature_cols)
                registered_version = register_model(full_pipeline, input_example=input_example)
                if registered_version:
                    promote_if_better(registered_version, test_f1)

                # Check for overfitting
                f1_diff = abs(train_f1 - test_f1)
                if f1_diff > self.model_trainer_config.overfitting_threshold:
                    logging.warning(f"Warning: Overfitting detected! Train/Test F1 gap: {f1_diff:.4f}")

                # Log run stats to Supabase/SQLite
                try:
                    self.db_client.log_pipeline_run(
                        run_id=run.info.run_id, accuracy=test_acc, f1_score=test_f1, model_path=model_save_path
                    )
                except Exception as db_err:
                    logging.error(f"Error logging to DB: {db_err}")

            # Return artifacts
            train_metrics = ClassificationMetricArtifact(
                f1_score=train_f1, precision_score=train_prec, recall_score=train_rec, accuracy_score=train_acc
            )
            test_metrics = ClassificationMetricArtifact(
                f1_score=test_f1, precision_score=test_prec, recall_score=test_rec, accuracy_score=test_acc
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_save_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
            )

            logging.info("Model Trainer completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e
