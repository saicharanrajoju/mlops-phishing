import sys

from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging


class PhishSentinelModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def predict(self, x):
        try:
            logging.info("Transforming input using preprocessor...")
            x_transformed = self.preprocessor.transform(x)
            logging.info("Running prediction on transformed inputs...")
            y_pred = self.model.predict(x_transformed)
            return y_pred
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def predict_proba(self, x):
        try:
            logging.info("Transforming input using preprocessor...")
            x_transformed = self.preprocessor.transform(x)
            logging.info("Running predict_proba on transformed inputs...")
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_transformed)
            else:
                # If model doesn't support predict_proba, fallback to decision function or mock probabilities
                preds = self.model.predict(x_transformed)
                # mock probability
                import numpy as np

                return np.array([[1 - p, p] for p in preds])
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def __repr__(self):
        return f"PhishSentinelModel(model={type(self.model).__name__})"
