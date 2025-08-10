import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from config.paths_config import *


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        pass

    def split_data(self):
        try:
            logger.info("Starting the splitting process")

            houses = fetch_california_housing()
            x = houses.data
            y = houses.target

            df_data = pd.DataFrame(houses.data, columns=houses.feature_names)
            train_data, test_data = train_test_split(df_data, test_size = 1 - self.train_test_ratio, random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH, index = False)
            test_data.to_csv(TEST_FILE_PATH, index = False)

            logger.info(f"Train data is saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data is saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while splitting the data")
            raise CustomException("Failed to split the data into train test", e)

    def run(self):
        try:
            logger.info("Starting date ingestion process")
            self.split_data()
            logger.info("Data Ingestion completed successfully")

        except Exception as ce:
            logger.error(f"Custom Exception :", {str(ce)})
        
        finally:
            logger.info("Data Ingestion completed finally")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
