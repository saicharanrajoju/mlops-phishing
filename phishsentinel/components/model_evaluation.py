"""
Model Evaluation gate.

Before a freshly-trained challenger is promoted, this component re-evaluates BOTH
the challenger and the current ``@production`` champion on the *same* held-out test
set and only promotes the challenger if it beats the champion (by at least
``min_improvement`` F1). This is stricter than comparing stored metrics: the champion
is scored on the current data, so a regression can never be silently promoted.
"""

import sys

import pandas as pd
from sklearn.metrics import f1_score

from phishsentinel.constant.training_pipeline import TARGET_COLUMN
from phishsentinel.entity.artifact_entity import ModelEvaluationArtifact
from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging
from phishsentinel.registry.model_registry import (
    load_model_version,
    load_production_model,
    set_production,
)


class ModelEvaluation:
    def __init__(self, model_eval_config, valid_test_file_path: str, candidate_version: str):
        self.config = model_eval_config
        self.valid_test_file_path = valid_test_file_path
        self.candidate_version = candidate_version

    @staticmethod
    def _f1(model, x, y) -> float:
        return float(f1_score(y, model.predict(x), average="weighted"))

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initiating Model Evaluation phase...")
            df = pd.read_csv(self.valid_test_file_path)
            y = df[TARGET_COLUMN].replace(-1, 0)  # align with training label mapping
            x = df.drop(columns=[TARGET_COLUMN])

            candidate = load_model_version(self.candidate_version)
            if candidate is None:
                raise Exception(f"Could not load candidate model version {self.candidate_version}")
            candidate_f1 = self._f1(candidate, x, y)

            champion = load_production_model()
            champion_f1 = self._f1(champion, x, y) if champion is not None else None

            if champion_f1 is None:
                accepted, improvement = True, candidate_f1
                logging.info(f"No current champion; accepting candidate (F1={candidate_f1:.4f}).")
            else:
                improvement = candidate_f1 - champion_f1
                accepted = improvement >= self.config.min_improvement
                logging.info(
                    f"Candidate F1={candidate_f1:.4f} vs champion F1={champion_f1:.4f} "
                    f"(delta={improvement:+.4f}); accepted={accepted}"
                )

            if accepted:
                set_production(self.candidate_version, candidate_f1)
                logging.info(f"Promoted version {self.candidate_version} to @production.")
            else:
                logging.info("Challenger rejected; @production unchanged.")

            return ModelEvaluationArtifact(
                is_model_accepted=accepted,
                candidate_f1=candidate_f1,
                champion_f1=champion_f1 if champion_f1 is not None else -1.0,
                improvement=improvement,
                promoted_version=self.candidate_version if accepted else None,
            )
        except Exception as e:
            raise PhishSentinelException(e, sys) from e
