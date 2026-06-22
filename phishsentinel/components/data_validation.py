import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

from phishsentinel.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from phishsentinel.entity.config_entity import DataValidationConfig
from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging
from phishsentinel.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(self.data_validation_config.schema_file_path)
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Expected number of columns: {number_of_columns}")
            logging.info(f"Actual columns count: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        """Detects column drift using Kolmogorov-Smirnov test"""
        try:
            report = {}
            drift_detected = False

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                # Run KS test
                is_same_dist = ks_2samp(d1, d2)

                # If p-value < threshold, we reject the null hypothesis -> distributions differ (drift detected)
                p_value = float(is_same_dist.pvalue)
                if p_value < threshold:
                    report[column] = {"p_value": p_value, "drift_status": True}
                    drift_detected = True
                else:
                    report[column] = {"p_value": p_value, "drift_status": False}

            # Save the drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)

            logging.info(f"Drift report written to {drift_report_file_path}")

            # If drift is detected on columns, log warning
            if drift_detected:
                logging.warning("Data drift detected in some columns!")

            return not drift_detected  # Returns validation status (False if drift detected)
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating Data Validation phase...")
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Validate number of columns
            train_columns_ok = self.validate_number_of_columns(train_df)
            test_columns_ok = self.validate_number_of_columns(test_df)

            if not train_columns_ok:
                logging.error("Train data columns mismatch.")
            if not test_columns_ok:
                logging.error("Test data columns mismatch.")

            validation_status = train_columns_ok and test_columns_ok

            # Check for data drift (informational — drift is surfaced, not used to block here)
            drift_free = self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            logging.info(f"Data drift check — distributions stable: {drift_free}")

            # Write valid files
            if validation_status:
                os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                logging.info("Copied validated datasets to valid directory.")
            else:
                os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                logging.error("Copied invalid datasets to invalid directory.")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path if validation_status else "",
                valid_test_file_path=self.data_validation_config.valid_test_file_path if validation_status else "",
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path
                if not validation_status
                else "",
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path
                if not validation_status
                else "",
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data Validation completed. Status: {validation_status}")
            return data_validation_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e
