import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from phishsentinel.entity.artifact_entity import DataIngestionArtifact
from phishsentinel.entity.config_entity import DataIngestionConfig
from phishsentinel.exception.exception import PhishSentinelException
from phishsentinel.logging.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """Reads raw phishing dataset from local repository and exports to feature store"""
        try:
            logging.info("Reading raw dataset from data/phisingData.csv...")
            raw_data_path = os.path.join("data", "phisingData.csv")
            if not os.path.exists(raw_data_path):
                raise Exception(f"Phishing dataset not found at: {raw_data_path}")

            df = pd.read_csv(raw_data_path)

            # Save to feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Successfully exported data to feature store: {feature_store_file_path}")
            return df
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def split_data_as_train_test(self, df: pd.DataFrame) -> DataIngestionArtifact:
        """Splits data into train/test sets and saves them to artifacts"""
        try:
            logging.info("Splitting dataset into train and test...")
            train_set, test_set = train_test_split(
                df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42
            )

            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            # Save files
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)

            logging.info(f"Saved training dataset to: {train_file_path}")
            logging.info(f"Saved test dataset to: {test_file_path}")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=train_file_path, test_file_path=test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating Data Ingestion phase...")
            df = self.export_data_into_feature_store()
            data_ingestion_artifact = self.split_data_as_train_test(df)
            logging.info("Data Ingestion completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            raise PhishSentinelException(e, sys) from e
