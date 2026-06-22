import os
import shutil
import sys

from phishsentinel.logging.logger import logging
from phishsentinel.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        logging.info("Initiating baseline training run...")
        print("MLOps: Initiating baseline training run...")

        pipeline = TrainingPipeline()
        model_trainer_artifact = pipeline.run_pipeline()

        # Save model to a predictable local path for final serving
        final_model_dir = "final_model"
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_path = os.path.join(final_model_dir, "model.pkl")

        # Copy the trained pickle to the final serving directory
        shutil.copy(model_trainer_artifact.trained_model_file_path, final_model_path)
        logging.info(f"Model saved to persistent deployment directory: {final_model_path}")
        print(f"MLOps: Model trained successfully and deployed to '{final_model_path}'!")

        # Output evaluation metrics
        train_metrics = model_trainer_artifact.train_metric_artifact
        test_metrics = model_trainer_artifact.test_metric_artifact
        print(f"  Train Accuracy: {train_metrics.accuracy_score:.4f} | F1: {train_metrics.f1_score:.4f}")
        print(f"  Test Accuracy:  {test_metrics.accuracy_score:.4f} | F1: {test_metrics.f1_score:.4f}")

    except Exception as e:
        logging.error(f"Execution failed: {e}")
        print(f"MLOps Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
