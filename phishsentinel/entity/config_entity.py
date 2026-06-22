import os
from datetime import datetime

from phishsentinel.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.timestamp = timestamp
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_dir = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        os.makedirs(self.artifact_dir, exist_ok=True)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, "phisingData.csv"
        )
        self.train_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.DATA_INGESTION_TRAIN_FILE_NAME,
        )
        self.test_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.DATA_INGESTION_TEST_FILE_NAME,
        )
        self.train_test_split_ratio: float = 0.2


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path = os.path.join(self.valid_data_dir, "train.csv")
        self.valid_test_file_path = os.path.join(self.valid_data_dir, "test.csv")
        self.invalid_train_file_path = os.path.join(self.invalid_data_dir, "train.csv")
        self.invalid_test_file_path = os.path.join(self.invalid_data_dir, "test.csv")
        self.schema_file_path = os.path.join("data_schema", "schema.yaml")
        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR, "train.npy"
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR, "test.npy"
        )
        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_PREPROCESSOR_DIR,
            training_pipeline.DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME,
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_ACCURACY
        self.overfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVER_FITTING_THRESHOLD
