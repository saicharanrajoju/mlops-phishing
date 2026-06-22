import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from phishsentinel.constant.training_pipeline import TARGET_COLUMN
from phishsentinel.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from phishsentinel.entity.config_entity import DataTransformationConfig
from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging
from phishsentinel.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(
        self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def get_data_transformer_object(cls) -> Pipeline:
        """Creates and returns the data preprocessing pipeline"""
        try:
            logging.info("Creating Preprocessing Pipeline: KNNImputer -> RobustScaler")
            imputer = KNNImputer(n_neighbors=3, weights="uniform")
            scaler = RobustScaler()

            preprocessor = Pipeline(steps=[("imputer", imputer), ("scaler", scaler)])
            return preprocessor
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating Data Transformation phase...")

            # Load validated datasets
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            logging.info("Splitting target column from features...")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Replaces target values: some phishing datasets use -1 for phishing and 1 for safe,
            # or 0/1. Let's make sure it's 0 (phishing) and 1 (safe), or let's keep original label structure.
            # Usually target contains -1 (phishing) and 1 (safe). Let's convert to 0 (phishing) and 1 (safe) for XGBoost compatibility!
            # XGBoost requires target labels to be [0, 1] instead of [-1, 1].
            # Let's map: -1 -> 0, 1 -> 1
            target_feature_train_df = target_feature_train_df.replace(-1, 0)
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # Get transformer pipeline
            preprocessor_pipeline = self.get_data_transformer_object()

            logging.info("Fitting and transforming datasets...")
            # Fit and transform train, transform test
            input_feature_train_arr = preprocessor_pipeline.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_pipeline.transform(input_feature_test_df)

            # Combine transformed features and target values into single numpy arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save transformed arrays
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr
            )
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path, obj=preprocessor_pipeline
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info("Data Transformation completed successfully.")
            return data_transformation_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e
