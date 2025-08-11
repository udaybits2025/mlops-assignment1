import logging

# Updating logging configuration to log to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_logs.log')
    ]
)

log = logging.getLogger(__name__)
