# Main purpose of creating logger is to store the logs

import logging
import os
from datetime import datetime # datetime is required to store the logs at specific time


LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok = True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename = LOG_FILE,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger