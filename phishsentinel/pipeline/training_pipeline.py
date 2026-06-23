import sys

from phishsentinel.components.data_ingestion import DataIngestion
from phishsentinel.components.data_transformation import DataTransformation
from phishsentinel.components.data_validation import DataValidation
from phishsentinel.components.model_evaluation import ModelEvaluation
from phishsentinel.components.model_trainer import ModelTrainer
from phishsentinel.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from phishsentinel.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact, model_trainer_config=model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            model_eval_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvaluation(
                model_eval_config=model_eval_config,
                valid_test_file_path=data_validation_artifact.valid_test_file_path,
                candidate_version=model_trainer_artifact.registered_version,
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            logging.info("===== Starting Training Pipeline Execution =====")

            # 1. Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # 2. Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            if not data_validation_artifact.validation_status:
                raise Exception("Data validation failed. Halting training pipeline.")

            # 3. Transformation
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)

            # 4. Trainer (registers a new model version)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            # 5. Evaluation gate (promotes to @production only if it beats the champion)
            if model_trainer_artifact.registered_version:
                eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
                logging.info(
                    f"Model Evaluation: accepted={eval_artifact.is_model_accepted}, "
                    f"candidate_f1={eval_artifact.candidate_f1:.4f}, champion_f1={eval_artifact.champion_f1:.4f}"
                )
            else:
                logging.warning("No registered version; skipping evaluation/promotion.")

            logging.info("===== Training Pipeline Completed Successfully =====")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error in Training Pipeline: {e}")
            raise PhishSentinelException(e, sys) from e
