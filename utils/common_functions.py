import os
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError("File is not present in the given path")
        
        with open(file_path, "r") as yaml_file:
            config =yaml.safe_load(yaml_file) # load the cotent of the yaml file
            logger.info("Successfully read the yaml file")
            return config
        
    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read the YAML file", e)


def load_data(path):
    try:
        logger.info("Loading the data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading the data {e}")
        raise CustomException("Failed to load the data", e)