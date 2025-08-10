# Adding code is WIP
import os
import pandas as pd
from src.logger import get_logger
from src.custom_Exception import CustomException
from config.paths_config import *
from utils.common_fucntions import read_yaml, load_data
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)

class DataPreprocessing:
    
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path =  train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config =read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def preprorcess_data(self, df):
        try:
            logger.info("Starting our Data Processing step")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            # Convert back to DataFrame with same column names
            df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
                       
            return df_scaled
        
        except Exception as e:
            logger.error(f"Error while preprocessing step {e}")
            raise CustomException("Error while preprocess data", e)
        
    def save_data(self,df, file_path):
        try:
            logger.info("Saving data in pre processed folder")

            df.to_csv(file_path, index = False)

            logger.info(f"Data saved to {file_path}")

        except Exception as e:
            logger.error(f"Error while Saving proccessed data {e}")
            raise CustomException("Error while proccessed data", e)
        
    def process(self):
        try:
            logger.info("Loading data from raw directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprorcess_data(train_df)
            test_df = self.preprorcess_data(test_df)

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data Preprocessing completed Successfully")

        except Exception as e:
            logger.error(f"Error while data preprocessing pipeline {e}")
            raise CustomException("Error while data preprocessing pipeline", e)

if __name__ == '__main__':
    processor = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()